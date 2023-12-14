#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <pthread.h>
#include <cassert>

#include "classifier.h"
#include "util.h"


/** SECTION: Constants and hyperparameters **/
#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

#define NUM_PARAMETER (OFFSET21 + 4)
#define POP_BATCH_SIZE 16
#define COMPUTE_BATCH_SIZE 1

static int mpi_size, mpi_rank;
static char processor_name[MPI_MAX_PROCESSOR_NAME];
static int iam_root;


/** SECTION: GPU manipulation **/
#define NGPU    4
static cudaStream_t streams[NGPU];
static float *a_input_gpu[NGPU], *a_conv1_gpu[NGPU];
static float *a_pool1_gpu[NGPU], *a_conv2_gpu[NGPU];
static float *a_pool2_gpu[NGPU], *a_conv3_gpu[NGPU];
static float *a_conv4_gpu[NGPU];
static float *a_conv5_gpu[NGPU];
static float *a_conv6_gpu[NGPU];
static float *a_collapse_gpu[NGPU];
static float *a_linear1_gpu[NGPU];
static float *a_linear2_gpu[NGPU];
static float *a_linear3_gpu[NGPU];

static float *w_conv1_gpu[NGPU], *b_conv1_gpu[NGPU];
static float *w_conv2_gpu[NGPU], *b_conv2_gpu[NGPU];
static float *w_conv3_gpu[NGPU], *b_conv3_gpu[NGPU];
static float *w_conv4_gpu[NGPU], *b_conv4_gpu[NGPU];
static float *w_conv5_gpu[NGPU], *b_conv5_gpu[NGPU];
static float *w_conv6_gpu[NGPU], *b_conv6_gpu[NGPU];
static float *w_fc1_gpu[NGPU], *b_fc1_gpu[NGPU];
static float *w_fc2_gpu[NGPU], *b_fc2_gpu[NGPU];
static float *w_fc3_gpu[NGPU], *b_fc3_gpu[NGPU];


/** SECTION: DEBUGGING **/
#define DEBUG 0
#if DEBUG == 1
#define DEBUG_PRINT(...) do { \
  printf("(%s|rank=%d) ", processor_name, mpi_rank); \
  printf(__VA_ARGS__); \
} while (0)
#else
#define DEBUG_PRINT(...)
#endif

int checksum(float *buf, int N) {
  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += (int) buf[i];

  return sum;
}


/** SECTION: Tensor **/

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  int num_elem();
  void fill_zeros();

  float *buf = nullptr;
  int ndim = 0;
  int shape[4];
};

Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
  for (int n = 0; n < N_; ++n) { buf[n] = buf_[n]; }
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

int Tensor::num_elem() {
  int sz = 1;
  for (int i = 0; i < ndim; ++i) { sz *= shape[i]; }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n = 0; n < N_; ++n) { buf[n] = 0.0; }
}

/** SECTION: Kernels **/

#define KERNEL_SIZE_3 3
#define KERNEL_SIZE_7 7
#define C1D_K3_BM 32
#define C1D_K3_BN 32
#define C1D_K3_BK 32

static __global__ void conv1d_k3_cuda(
  float *input, float *weight, float *bias, float *output,
  int num_batch, int len_output, int in_channels, int out_channels,
  int relu
) {
  /** PARAMS **/
  // input: float[batch_size, in_channels, len_input]
  // weight: float[out_channels, in_channels, kernel_size]
  // bias: float[out_channels]
  // output: float[batch_size, out_channels, len_output]

  /** CONSTS **/
  const int BLOCK_SIZE = C1D_K3_BM;
  const int KERNEL_SIZE = KERNEL_SIZE_3;
  
  /** VARS **/
  int block_channel_idx = blockIdx.x * BLOCK_SIZE;
  int block_output_idx = blockIdx.y * BLOCK_SIZE;
  int block_channel_len = min(BLOCK_SIZE, out_channels - block_channel_idx);
  int block_output_len = min(BLOCK_SIZE, len_output - block_output_idx);

  int thread_channel_idx = block_channel_idx + threadIdx.x;
  int thread_output_idx = block_output_idx + threadIdx.y;

  int len_input = len_output + KERNEL_SIZE - 1;

  /** SMEM **/
  __shared__ float input_buf[BLOCK_SIZE][BLOCK_SIZE + KERNEL_SIZE - 1];
  __shared__ float kernel_buf[BLOCK_SIZE][BLOCK_SIZE][KERNEL_SIZE];

  /** REGS **/
  float val = 0.0f;

  /** ITER THROUGH K **/
  for (int b_inchan_idx = 0; b_inchan_idx < in_channels; b_inchan_idx += BLOCK_SIZE) {
    // Load input
    int thread_inchan_idx = b_inchan_idx + threadIdx.x;
    if (thread_inchan_idx < in_channels && thread_output_idx < len_input) {
      // Activate only valid threads
      input_buf[threadIdx.x][threadIdx.y] = input[
        thread_inchan_idx * len_input + thread_output_idx
      ];
      // Load extra input
      if (threadIdx.y == block_output_len - 1) {
        input_buf[threadIdx.x][threadIdx.y + 1] = input[
          thread_inchan_idx * len_input + thread_output_idx + 1
        ];
        input_buf[threadIdx.x][threadIdx.y + 2] = input[
          thread_inchan_idx * len_input + thread_output_idx + 2
        ];
      }
    }

    // Load kernel
    // Every threads are valid
    float *weight_ptr = &weight[
      thread_channel_idx * in_channels * KERNEL_SIZE + (b_inchan_idx + threadIdx.y) * KERNEL_SIZE
    ];
    for (int ks = 0; ks < KERNEL_SIZE; ks++) {
      kernel_buf[threadIdx.x][threadIdx.y][ks] = weight_ptr[ks];
    }

    __syncthreads();
    // No need to select - store will be selective
    for (int k=0; k < BLOCK_SIZE; k++) {
      /** COMPUTE **/
      for (int ks = 0; ks < KERNEL_SIZE; ks++) {
        val += kernel_buf[threadIdx.x][k][ks] * input_buf[k][threadIdx.y + ks];
      }
    }

    __syncthreads();
  }

  /** STORE **/
  if (thread_channel_idx < out_channels && thread_output_idx < len_output) {
    val += bias[thread_channel_idx];
    if (relu && val < 0.0f) val = 0.0f;
    output[thread_channel_idx * len_output + thread_output_idx] = val;
  }
}

static __global__ void conv1d_cuda(
  float *input, float *weight, float *bias, float *output,
  int num_batch, int len_output, int in_channels, int out_channels, int kernel_size,
  int relu
) {
  /** PARAMS **/
  // input: float[batch_size, in_channels, len_input]
  // weight: float[out_channels, in_channels, kernel_size]
  // bias: float[out_channels]
  // output: float[batch_size, out_channels, len_output]

  /** CONSTS **/
  const int BLOCK_SIZE = C1D_K3_BM;
  
  /** VARS **/
  int block_channel_idx = blockIdx.x * BLOCK_SIZE;
  int block_output_idx = blockIdx.y * BLOCK_SIZE;
  int block_channel_len = min(BLOCK_SIZE, out_channels - block_channel_idx);
  int block_output_len = min(BLOCK_SIZE, len_output - block_output_idx);

  int thread_channel_idx = block_channel_idx + threadIdx.x;
  int thread_output_idx = block_output_idx + threadIdx.y;

  int len_input = len_output + kernel_size - 1;

  if (thread_output_idx < len_output) {
    /** COMPUTE **/
    float val = 0.0f;
    for (int k = 0; k < in_channels; k++) {
      float *input_ptr = &input[k * len_input + thread_output_idx];
      float *weight_ptr = &weight[thread_channel_idx * in_channels * kernel_size + k * kernel_size];
      for (int ks = 0; ks < kernel_size; ks++) {
        val += weight_ptr[ks] * input_ptr[ks];
      }
    }

    /** STORE **/
    val += bias[thread_channel_idx];
    if (relu && val < 0.0f) val = 0.0f;
    output[thread_channel_idx * len_output + thread_output_idx] = val;
  }
}

#define LIN_BM 32
#define LIN_BN 32
#define LIN_BK 32

static __global__ void linear_cuda(
  float *input, float *weight, float *bias, float *output,
  int num_batch, int in_channels, int out_channels,
  int relu
) {
  /** PARAMS **/
  // input: float[batch_size, in_channels]
  // weight: float[out_channels, in_channels]
  // bias: float[out_channels]
  // output: float[batch_size, out_channels]

  /** CONSTS **/
  const int BLOCK_SIZE = LIN_BM;

  /** VARS **/
  int block_batch_idx = blockIdx.x * BLOCK_SIZE;
  int block_outchan_idx = blockIdx.y * BLOCK_SIZE;
  int block_batch_len = min(BLOCK_SIZE, num_batch - block_batch_idx);
  int block_outchan_len = min(BLOCK_SIZE, out_channels - block_outchan_idx);

  int thread_batch_idx = block_batch_idx + threadIdx.x;
  int thread_outchan_idx = block_outchan_idx + threadIdx.y;

  if (thread_outchan_idx < out_channels && thread_batch_idx < num_batch) {
    /** COMPUTE **/
    float val = 0.0f;
    for (int k = 0; k < in_channels; k++) {
      val += weight[thread_outchan_idx * in_channels + k] * input[thread_batch_idx * in_channels + k];
    }

    /** STORE **/
    val += bias[thread_outchan_idx];
    if (relu && val < 0.0f) val = 0.0f;
    output[thread_batch_idx * out_channels + thread_outchan_idx] = val;
  }
}


/** SECTION: Operators **/

void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride, int relu) {
  int B = input->shape[0];
  int IL = input->shape[2];
  int OC = output->shape[1];
  int OL = output->shape[2];

  int single_input_size = input->num_elem() / B;
  int single_output_size = output->num_elem() / B;

  for (int batch = 0; batch < B; batch++) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int ol = 0; ol < OL; ++ol) {
        float mx = -1e99;
        for (int ks = 0; ks < kernel_size; ++ks) {
          float val = input->buf[batch * single_input_size + oc * IL + ks + ol * stride];
          if (val > mx) mx = val;
        }
        if (relu && mx < 0.0f) mx = 0.0f;
        output->buf[batch * single_output_size + oc * OL + ol] = mx;
      }
    }
  }
}

void collapse(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i];
  }
}

void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias) {
  int B = input->shape[0];
  int IC = input->shape[1];
  int OC = output->shape[1];

  int single_input_size = input->num_elem() / B;
  int single_output_size = output->num_elem() / B;

  for (int batch = 0; batch < B; batch++) {
    for (int oc = 0; oc < OC; ++oc) {
      float val = 0.0;
      for (int ic = 0; ic < IC; ++ic) {
        val += input->buf[batch * single_input_size + ic] * weight->buf[oc * IC + ic];
      }
      if (has_bias) val += bias->buf[oc];
      output->buf[batch * single_output_size + oc] = val;
    }
  }
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  int B = input->shape[0];

  int single_input_size = input->num_elem() / B;
  int single_output_size = output->num_elem() / B;

  for (int batch = 0; batch < B; batch++) {
    // E[X], E[X^2]
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < single_input_size; ++i) {
        sum1 += input->buf[batch * single_input_size + i];
        sum2 += input->buf[batch * single_input_size + i] * input->buf[batch * single_input_size + i];
    }
    float mean1 = sum1 / (float)single_input_size;
    float mean2 = sum2 / (float)single_input_size;

    // V[X]
    float var = mean2 - mean1 * mean1; 

    // Normalization
    for (int i = 0; i < single_output_size; ++i) {
      output->buf[batch * single_output_size + i]
       = (input->buf[batch * single_input_size + i] - mean1)
       / sqrtf(var + 1e-5) * gamma->buf[i] + beta->buf[i];
    }
  }

}


/** SECTION: COMPUTE_ENGINE **/

struct ComputeEngine {
public:
  ComputeEngine(float *parameter, int num_input, int gpu_idx);
  ~ComputeEngine();

  void set_input(float *input_buf_);
  void set_output(float *output_buf_);

  void run();
  void join();

  void push(int num_input);

private:
  // Input
  float *input_buf;
  float *output_buf;
  int num_input;
  
  // GPU
  int gpu_idx;

  // Queue
  float *input_to_process;
  int num_input_ready;
  int num_input_processed;
  pthread_mutex_t mutex_queue;
  pthread_cond_t cond_queue;

  int pop();
  void inference(int num_input);

  // Runner
  static void *run_func(void *arg);
  pthread_t thread;

  // Parameters
  Tensor *w_fc1, *w_fc2, *w_fc3,
      *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

  // Activations
  Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
  Tensor *a_relu2, *a_pool2;
  Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
  Tensor *a_linear3;
};

ComputeEngine *compute_engines[NGPU];

ComputeEngine::ComputeEngine(float *parameter_, int num_input_, int gpu_idx_) {
  // Initialize member variables
  input_buf = nullptr;
  output_buf = nullptr;
  num_input = num_input_;

  gpu_idx = gpu_idx_;

  // Initialize queue
  num_input_ready = 0;
  num_input_processed = 0;
  pthread_mutex_init(&mutex_queue, NULL);
  pthread_cond_init(&cond_queue, NULL);

  // Initialize CUDA
  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  // Initialize parameters
  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&a_input_gpu[i], COMPUTE_BATCH_SIZE * 70 * 1014 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv1_gpu[i], COMPUTE_BATCH_SIZE * 256 * 1008 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_pool1_gpu[i], COMPUTE_BATCH_SIZE * 256 * 336 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv2_gpu[i], COMPUTE_BATCH_SIZE * 256 * 330 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_pool2_gpu[i], COMPUTE_BATCH_SIZE * 256 * 110 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv3_gpu[i], COMPUTE_BATCH_SIZE * 256 * 108 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv4_gpu[i], COMPUTE_BATCH_SIZE * 256 * 106 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv5_gpu[i], COMPUTE_BATCH_SIZE * 256 * 104 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_conv6_gpu[i], COMPUTE_BATCH_SIZE * 256 * 102 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_collapse_gpu[i], COMPUTE_BATCH_SIZE * 8704 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_linear1_gpu[i], COMPUTE_BATCH_SIZE * 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_linear2_gpu[i], COMPUTE_BATCH_SIZE * 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&a_linear3_gpu[i], COMPUTE_BATCH_SIZE * 4 * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&w_conv1_gpu[i], 256 * 70 * 7 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv1_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_conv2_gpu[i], 256 * 256 * 7 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv2_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_conv3_gpu[i], 256 * 256 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv3_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_conv4_gpu[i], 256 * 256 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv4_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_conv5_gpu[i], 256 * 256 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv5_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_conv6_gpu[i], 256 * 256 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_conv6_gpu[i], 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_fc1_gpu[i], 1024 * 8704 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_fc1_gpu[i], 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_fc2_gpu[i], 1024 * 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_fc2_gpu[i], 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w_fc3_gpu[i], 4 * 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_fc3_gpu[i], 4 * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(
      w_conv1_gpu[i], parameter_ + OFFSET0, 256 * 70 * 7 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv1_gpu[i], parameter_ + OFFSET1, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_conv2_gpu[i], parameter_ + OFFSET4, 256 * 256 * 7 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv2_gpu[i], parameter_ + OFFSET5, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_conv3_gpu[i], parameter_ + OFFSET6, 256 * 256 * 3 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv3_gpu[i], parameter_ + OFFSET7, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_conv4_gpu[i], parameter_ + OFFSET8, 256 * 256 * 3 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv4_gpu[i], parameter_ + OFFSET9, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_conv5_gpu[i], parameter_ + OFFSET10, 256 * 256 * 3 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv5_gpu[i], parameter_ + OFFSET11, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_conv6_gpu[i], parameter_ + OFFSET12, 256 * 256 * 3 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_conv6_gpu[i], parameter_ + OFFSET13, 256 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_fc1_gpu[i], parameter_ + OFFSET16, 1024 * 8704 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_fc1_gpu[i], parameter_ + OFFSET17, 1024 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_fc2_gpu[i], parameter_ + OFFSET18, 1024 * 1024 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_fc2_gpu[i], parameter_ + OFFSET19, 1024 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      w_fc3_gpu[i], parameter_ + OFFSET20, 4 * 1024 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    CHECK_CUDA(cudaMemcpyAsync(
      b_fc3_gpu[i], parameter_ + OFFSET21, 4 * sizeof(float),
      cudaMemcpyHostToDevice, streams[i]
    ));
    
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }
  gamma_conv1 = new Tensor({256, 1008}, parameter_ + OFFSET2);
  beta_conv1 = new Tensor({256, 1008}, parameter_ + OFFSET3);
  gamma_conv6 = new Tensor({256, 102}, parameter_ + OFFSET14);
  beta_conv6 = new Tensor({256, 102}, parameter_ + OFFSET15);
  w_fc1 = new Tensor({1024, 8704}, parameter_ + OFFSET16);
  b_fc1 = new Tensor({1024}, parameter_ + OFFSET17);
  w_fc2 = new Tensor({1024, 1024}, parameter_ + OFFSET18);
  b_fc2 = new Tensor({1024}, parameter_ + OFFSET19);
  w_fc3 = new Tensor({4, 1024}, parameter_ + OFFSET20);
  b_fc3 = new Tensor({4}, parameter_ + OFFSET21);

  // Initialize activations
  a_conv1 = new Tensor({COMPUTE_BATCH_SIZE, 256, 1008});
  a_layernorm1 = new Tensor({COMPUTE_BATCH_SIZE, 256, 1008});
  a_relu1 = new Tensor({COMPUTE_BATCH_SIZE, 256, 1008});
  a_pool1 = new Tensor({COMPUTE_BATCH_SIZE, 256, 336});
  a_relu2 = new Tensor({COMPUTE_BATCH_SIZE, 256, 330});
  a_pool2 = new Tensor({COMPUTE_BATCH_SIZE, 256, 110});
  a_conv6 = new Tensor({COMPUTE_BATCH_SIZE, 256, 102});
  a_layernorm6 = new Tensor({COMPUTE_BATCH_SIZE, 256, 102});
  a_relu6 = new Tensor({COMPUTE_BATCH_SIZE, 256, 102});
  a_pool6 = new Tensor({COMPUTE_BATCH_SIZE, 256, 34});
  a_linear3 = new Tensor({COMPUTE_BATCH_SIZE, 4});
}

ComputeEngine::~ComputeEngine() {
  pthread_mutex_destroy(&mutex_queue);
  pthread_cond_destroy(&cond_queue);

  delete w_fc1; delete b_fc1;
  delete w_fc2; delete b_fc2;
  delete w_fc3; delete b_fc3;
  
  delete gamma_conv1; delete gamma_conv6;
  delete beta_conv1; delete beta_conv6;
  delete a_conv1; delete a_layernorm1; delete a_relu1; delete a_pool1;
  delete a_relu2; delete a_pool2;
  delete a_conv6; delete a_layernorm6; delete a_relu6; delete a_pool6;
  delete a_linear3;
}

void ComputeEngine::set_input(float *input_buf_) {
  input_buf = input_buf_;
  input_to_process = input_buf;
}

void ComputeEngine::set_output(float *output_buf_) {
  output_buf = output_buf_;
}

void ComputeEngine::run() {
  pthread_create(&thread, NULL, ComputeEngine::run_func, this);
}

void ComputeEngine::join() {
  pthread_join(thread, NULL);
}

void ComputeEngine::push(int num_input) {
  pthread_mutex_lock(&mutex_queue);
  if (num_input_ready == 0) pthread_cond_signal(&cond_queue);
  num_input_ready += num_input;
  pthread_mutex_unlock(&mutex_queue);
}

int ComputeEngine::pop() {
  int num_input = 0;
  pthread_mutex_lock(&mutex_queue);
  if (num_input_ready == 0) pthread_cond_wait(&cond_queue, &mutex_queue);
  num_input = std::min(num_input_ready, POP_BATCH_SIZE);
  num_input_ready -= num_input;
  pthread_mutex_unlock(&mutex_queue);
  return num_input;
}

void ComputeEngine::inference(int num_input) {
  DEBUG_PRINT("Inference %d\n", num_input);

  CHECK_CUDA(cudaSetDevice(gpu_idx));
  
  for (int batch = 0; batch < num_input; batch+=COMPUTE_BATCH_SIZE) {
    DEBUG_PRINT("Inference %d/%d\n", num_input_processed+1, num_input);

    int now_batch_size = std::min(COMPUTE_BATCH_SIZE, num_input - batch);
    
    Tensor *input = new Tensor(
      {now_batch_size, VOCAB_SIZE, MAX_LENGTH}, 
      input_to_process + batch * VOCAB_SIZE * MAX_LENGTH
    );

    // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
    {
      CHECK_CUDA(cudaMemcpyAsync(
        a_input_gpu[gpu_idx], input->buf, now_batch_size * 70 * 1014 * sizeof(float),
        cudaMemcpyHostToDevice, streams[gpu_idx]
      ));

      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(1008, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_input_gpu[gpu_idx], w_conv1_gpu[gpu_idx], b_conv1_gpu[gpu_idx], a_conv1_gpu[gpu_idx],
        now_batch_size, 1008, 70, 256, 7,
        0
      );

      CHECK_CUDA(cudaMemcpyAsync(
        a_conv1->buf, a_conv1_gpu[gpu_idx], now_batch_size * 256 * 1008 * sizeof(float),
        cudaMemcpyDeviceToHost, streams[gpu_idx]
      ));

      CHECK_CUDA(cudaStreamSynchronize(streams[gpu_idx]));
      
      layernorm(a_conv1, gamma_conv1, beta_conv1, a_layernorm1);
      maxpool1d(a_layernorm1, a_pool1, 3, 3, 1);
    }

    // Conv block 2 : Conv1d + ReLU + MaxPool1d
    {
      CHECK_CUDA(cudaMemcpyAsync(
        a_pool1_gpu[gpu_idx], a_pool1->buf, now_batch_size * 256 * 336 * sizeof(float),
        cudaMemcpyHostToDevice, streams[gpu_idx]
      ));

      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(330, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_pool1_gpu[gpu_idx], w_conv2_gpu[gpu_idx], b_conv2_gpu[gpu_idx], a_conv2_gpu[gpu_idx],
        now_batch_size, 330, 256, 256, 7,
        1
      );

      CHECK_CUDA(cudaMemcpyAsync(
        a_relu2->buf, a_conv2_gpu[gpu_idx], now_batch_size * 256 * 330 * sizeof(float),
        cudaMemcpyDeviceToHost, streams[gpu_idx]
      ));

      CHECK_CUDA(cudaStreamSynchronize(streams[gpu_idx]));

      maxpool1d(a_relu2, a_pool2, 3, 3, 0);
    }

    // Conv block 3 : Conv1d + ReLU
    {
      CHECK_CUDA(cudaMemcpyAsync(
        a_pool2_gpu[gpu_idx], a_pool2->buf, now_batch_size * 256 * 110 * sizeof(float),
        cudaMemcpyHostToDevice, streams[gpu_idx]
      ));

      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(108, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_pool2_gpu[gpu_idx], w_conv3_gpu[gpu_idx], b_conv3_gpu[gpu_idx], a_conv3_gpu[gpu_idx],
        now_batch_size, 108, 256, 256,
        1
      );

    }

    // Conv block 4 : Conv1d + ReLU
    {

      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(106, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_conv3_gpu[gpu_idx], w_conv4_gpu[gpu_idx], b_conv4_gpu[gpu_idx], a_conv4_gpu[gpu_idx],
        now_batch_size, 106, 256, 256,
        1
      );
    }

    // Conv block 5 : Conv1d + ReLU
    {
      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(104, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_conv4_gpu[gpu_idx], w_conv5_gpu[gpu_idx], b_conv5_gpu[gpu_idx], a_conv5_gpu[gpu_idx],
        now_batch_size, 104, 256, 256,
        1
      );
    }


    // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
    {
      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(102, C1D_K3_BN));
      dim3 block(C1D_K3_BM, C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_conv5_gpu[gpu_idx], w_conv6_gpu[gpu_idx], b_conv6_gpu[gpu_idx], a_conv6_gpu[gpu_idx],
        now_batch_size, 102, 256, 256,
        0
      );

      CHECK_CUDA(cudaMemcpyAsync(
        a_conv6->buf, a_conv6_gpu[gpu_idx], now_batch_size * 256 * 102 * sizeof(float),
        cudaMemcpyDeviceToHost, streams[gpu_idx]
      ));

      CHECK_CUDA(cudaStreamSynchronize(streams[gpu_idx]));
    }
    layernorm(a_conv6, gamma_conv6, beta_conv6, a_layernorm6);
    maxpool1d(a_layernorm6, a_pool6, 3, 3, 1);

    // FC block 1 : Linear + ReLU
    {
      CHECK_CUDA(cudaMemcpyAsync(
        a_collapse_gpu[gpu_idx], a_pool6->buf, now_batch_size * 8704 * sizeof(float),
        cudaMemcpyHostToDevice, streams[gpu_idx]
      ));

      dim3 grid(CEIL_DIV(now_batch_size, LIN_BM), CEIL_DIV(1024, LIN_BN));
      dim3 block(LIN_BM, LIN_BN);
      linear_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_collapse_gpu[gpu_idx], w_fc1_gpu[gpu_idx], b_fc1_gpu[gpu_idx], a_linear1_gpu[gpu_idx],
        now_batch_size, 8704, 1024,
        1
      );
    }

    // FC block 2 : Linear + ReLU
    {
      dim3 grid(CEIL_DIV(now_batch_size, LIN_BM), CEIL_DIV(1024, LIN_BN));
      dim3 block(LIN_BM, LIN_BN);
      linear_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_linear1_gpu[gpu_idx], w_fc2_gpu[gpu_idx], b_fc2_gpu[gpu_idx], a_linear2_gpu[gpu_idx],
        now_batch_size, 1024, 1024,
        1
      );
    }

    // FC block 3 : Linear
    {
      dim3 grid(CEIL_DIV(now_batch_size, LIN_BM), CEIL_DIV(4, LIN_BN));
      dim3 block(LIN_BM, LIN_BN);
      linear_cuda<<<grid, block, 0, streams[gpu_idx]>>>(
        a_linear2_gpu[gpu_idx], w_fc3_gpu[gpu_idx], b_fc3_gpu[gpu_idx], a_linear3_gpu[gpu_idx],
        now_batch_size, 1024, 4,
        0
      );

      CHECK_CUDA(cudaMemcpyAsync(
        a_linear3->buf, a_linear3_gpu[gpu_idx], now_batch_size * 4 * sizeof(float),
        cudaMemcpyDeviceToHost, streams[gpu_idx]
      ));

      CHECK_CUDA(cudaStreamSynchronize(streams[gpu_idx]));
    }

    int single_logit_size = a_linear3->num_elem() / now_batch_size;

    for (int b = 0; b < now_batch_size; b++) {
      float max_val = -1e99f;
      int max_idx = 0;
      for (int i = 0; i < single_logit_size; ++i) {
        if (a_linear3->buf[b * single_logit_size + i] > max_val) {
          max_val = a_linear3->buf[b * single_logit_size + i];
          max_idx = i;
        }
      }
      output_buf[num_input_processed++] = (float)max_idx;
    }

  }

  input_to_process += num_input * VOCAB_SIZE * MAX_LENGTH;
}

void *ComputeEngine::run_func(void *arg) {
  ComputeEngine *engine = (ComputeEngine *) arg;
  while (engine->num_input_processed < engine->num_input) {
    int num_input = engine->pop();
    engine->inference(num_input);
  }
  return NULL;
}


/** SECTION: Classifier interface **/

void initialize_classifier(float *parameter, int N) {
  int len_name;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Get_processor_name(processor_name, &len_name);
  iam_root = (mpi_rank == 0);

  // Broadcast parameters
  if (parameter == nullptr)
    parameter = (float *) calloc(NUM_PARAMETER, sizeof(float));

  MPI_Bcast(parameter, NUM_PARAMETER, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx)
    compute_engines[ce_idx] = new ComputeEngine(parameter, N / mpi_size / NGPU, ce_idx);
}

void classifier(float *input_, float *output_, int N) {
  // Scatter input & initialize memory
  DEBUG_PRINT("Scatter input\n");
  if (iam_root) {
    MPI_Scatter(
      input_, N * VOCAB_SIZE * MAX_LENGTH / mpi_size, MPI_FLOAT,
      MPI_IN_PLACE, N * VOCAB_SIZE * MAX_LENGTH / mpi_size, MPI_FLOAT, 
      0, MPI_COMM_WORLD
    );
  }
  else {
    if (input_ == nullptr) 
      input_ = (float *) calloc(
        N * VOCAB_SIZE * MAX_LENGTH / mpi_size, 
        sizeof(float)
      );
    if (output_ == nullptr) 
      output_ = (float *) calloc(N / mpi_size, sizeof(float));

    MPI_Scatter(
      MPI_IN_PLACE, N * VOCAB_SIZE * MAX_LENGTH / mpi_size, MPI_FLOAT,
      input_, N * VOCAB_SIZE * MAX_LENGTH / mpi_size, MPI_FLOAT, 
      0, MPI_COMM_WORLD
    );
  }
  DEBUG_PRINT("Scatter input done\n");
  

  // Compute
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->set_input(input_ + ce_idx * N / mpi_size / NGPU * VOCAB_SIZE * MAX_LENGTH);
    compute_engines[ce_idx]->set_output(output_ + ce_idx * N / mpi_size / NGPU);
    compute_engines[ce_idx]->run();
    compute_engines[ce_idx]->push(N / mpi_size / NGPU);
  }

  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->join();
  }

  // Gather output
  if (iam_root) {
    MPI_Gather(MPI_IN_PLACE, N / mpi_size, MPI_FLOAT,
               output_, N / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Gather(output_, N / mpi_size, MPI_FLOAT,
               MPI_IN_PLACE, N / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

}

void finalize_classifier() {
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    delete compute_engines[ce_idx];
  }
}

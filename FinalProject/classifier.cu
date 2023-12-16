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

static int mpi_size, mpi_rank;
static char processor_name[MPI_MAX_PROCESSOR_NAME];
static int iam_root;


/** SECTION: GPU manipulation **/
#define NGPU    4


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


/** SECTION: Hyperparams **/
#define MAX_MPI_SIZE 4

#define PUSH_BATCH_SIZE 16
#define POP_BATCH_SIZE 16
#define COMPUTE_BATCH_SIZE 4

#define C1D_K3_BM 16
#define C1D_K3_BN 8
#define C1D_K3_BK 8

#define C1D_K7_BM 8
#define C1D_K7_BN 18
#define C1D_K7_BK 4

#define LIN_NAIVE_BM 4
#define LIN_NAIVE_BN 16

#define LIN_REG_BM 4
#define LIN_REG_BN 16
#define LIN_REG_BK 32

#define LNORM_CHAN 256  // NEVER CHANGE!
#define LNORM_CHPT 2
#define LNORM_TPB  (LNORM_CHAN/LNORM_CHPT)  // NEVER CHANGE!


/** SECTION: Kernels **/

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
  const int BB = COMPUTE_BATCH_SIZE;
  const int BM = C1D_K3_BM;
  const int BN = C1D_K3_BN;
  const int BK = C1D_K3_BK;

  const int KERNEL_SIZE = 3;
  const int len_input = len_output + KERNEL_SIZE - 1;
  const int single_input_size = in_channels * len_input;
  const int single_output_size = out_channels * len_output;

  /** ASSERTION **/
  #if DEBUG == 1
  if (BM * BN < BM * BK * KERNEL_SIZE) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      // printf("conv1d_k3_cuda: num of threads are insufficient for kernel load!\n");
      // return;
    }
  }
  #endif
  
  /** VARS **/
  float val[BB] = {0.0f};
  
  // output block
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;
  int len_oblock_m = min(BM, out_channels - oblock_m_offset);
  int len_oblock_n = min(BN, len_output - oblock_n_offset);
  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;
  
  /** SMEM **/
  __shared__ float input_buf[BB][BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float weight_buf[BM][BK][KERNEL_SIZE + 4];

  /** LOOP OVER K **/
  for (int bk = 0; bk < in_channels; bk += BK) {
    // load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, in_channels - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, len_input - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid) {
      for (int bb = 0; bb < num_batch; bb++) {
        input_buf[bb][ithread_k_offset][ithread_n_offset] = input[
          bb * single_input_size
           + (iblock_k_offset + ithread_k_offset) * len_input
           + iblock_n_offset + ithread_n_offset
        ];
      }
    }

    // load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, out_channels - wblock_m_offset);
    int len_wblock_k = min(BK, in_channels - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if (wthread_valid) {
      for (int ks = 0; ks < KERNEL_SIZE; ks++) {
        weight_buf[wthread_m_offset][wthread_k_offset][ks] = weight[
          (wblock_m_offset + wthread_m_offset) * in_channels * KERNEL_SIZE
          + (wblock_k_offset + wthread_k_offset) * KERNEL_SIZE + ks
        ];
      }
    }

    __syncthreads();

    // compute
    if (othread_valid) {
      for (int bb = 0; bb < num_batch; bb++) {
        for (int k = 0; k < BK; k++) {
          for (int ks = 0; ks < KERNEL_SIZE; ks++) {
            val[bb] += weight_buf[othread_m_offset][k][ks] * input_buf[bb][k][othread_n_offset + ks];
          }
        }
      }
    }

    __syncthreads();
  }

  /** STORE **/
  if (othread_valid) {
    for (int bb = 0; bb < num_batch; bb++) {
      val[bb] += bias[oblock_m_offset + othread_m_offset];
      if (relu && val[bb] < 0.0f) val[bb] = 0.0f;
      output[
        bb * single_output_size
         + (oblock_m_offset + othread_m_offset) * len_output
         + oblock_n_offset + othread_n_offset
      ] = val[bb];
    }
  }
}

static __global__ void conv1d_k7_cuda(
  float *input, float *weight, float *bias, float *output,
  int num_batch, int len_output, int in_channels, int out_channels,
  int relu, int mpool3=0, float *pooled_output=nullptr
) {
  /** PARAMS **/
  // input: float[batch_size, in_channels, len_input]
  // weight: float[out_channels, in_channels, kernel_size]
  // bias: float[out_channels]
  // output: float[batch_size, out_channels, len_output]

  /** CONSTS **/
  const int BB = COMPUTE_BATCH_SIZE;
  const int BM = C1D_K7_BM;
  const int BN = C1D_K7_BN;
  const int BK = C1D_K7_BK;

  const int KERNEL_SIZE = 7;
  const int len_input = len_output + KERNEL_SIZE - 1;

  const int single_input_size = in_channels * len_input;
  const int single_output_size = out_channels * len_output;

  /** ASSERTION **/
  #if DEBUG == 1
  if (BM * BN < BM * BK * KERNEL_SIZE) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      // printf("conv1d_k3_cuda: num of threads are insufficient for kernel load!\n");
      // return;
    }
  }
  #endif
  
  /** VARS **/
  float val[BB] = {0.0f};
  
  // output block
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;
  int len_oblock_m = min(BM, out_channels - oblock_m_offset);
  int len_oblock_n = min(BN, len_output - oblock_n_offset);
  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;
  
  /** SMEM **/
  __shared__ float input_buf[BB][BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float weight_buf[BM][BK][KERNEL_SIZE + 4];

  /** LOOP OVER K **/
  for (int bk = 0; bk < in_channels; bk += BK) {
    // load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, in_channels - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, len_input - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid) {
      for (int bb = 0; bb < num_batch; bb++) {
        input_buf[bb][ithread_k_offset][ithread_n_offset] = input[
          bb * single_input_size
           + (iblock_k_offset + ithread_k_offset) * len_input
           + iblock_n_offset + ithread_n_offset
        ];
      }
    }

    // load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, out_channels - wblock_m_offset);
    int len_wblock_k = min(BK, in_channels - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if (wthread_valid) {
      for (int ks = 0; ks < KERNEL_SIZE; ks++) {
        weight_buf[wthread_m_offset][wthread_k_offset][ks] = weight[
          (wblock_m_offset + wthread_m_offset) * in_channels * KERNEL_SIZE
          + (wblock_k_offset + wthread_k_offset) * KERNEL_SIZE + ks
        ];
      }
    }

    __syncthreads();

    // compute
    if (othread_valid) {
      for (int bb = 0; bb < num_batch; bb++) {
        for (int k = 0; k < BK; k++) {
          for (int ks = 0; ks < KERNEL_SIZE; ks++) {
            val[bb] += weight_buf[othread_m_offset][k][ks] * input_buf[bb][k][othread_n_offset + ks];
          }
        }
      }
    }

    __syncthreads();
  }

  /** STORE **/
  if (othread_valid) {
    for (int bb = 0; bb < num_batch; bb++) {
      val[bb] += bias[oblock_m_offset + othread_m_offset];
      if (relu && val[bb] < 0.0f) val[bb] = 0.0f;
      output[
        bb * single_output_size
         + (oblock_m_offset + othread_m_offset) * len_output
         + oblock_n_offset + othread_n_offset
      ] = val[bb];
    }
  }
}

static __global__ void linear_naive_cuda(
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
  const int BM = LIN_NAIVE_BM;
  const int BN = LIN_NAIVE_BN;

  /** VARS **/
  int block_batch_idx = blockIdx.x * BM;
  int block_outchan_idx = blockIdx.y * BN;
  int block_batch_len = min(BM, num_batch - block_batch_idx);
  int block_outchan_len = min(BN, out_channels - block_outchan_idx);

  int thread_batch_idx = block_batch_idx + threadIdx.x;
  int thread_outchan_idx = block_outchan_idx + threadIdx.y;

  if (thread_batch_idx < num_batch && thread_outchan_idx < out_channels) {
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

static __global__ void linear_reg_cuda(
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
  const int BM = LIN_REG_BM;
  const int BN = LIN_REG_BN;
  const int BK = LIN_REG_BK;
  const int LDPT_INPUT = BK / BN;
  const int LDPT_WEIGHT = BK / BM;

  /** VARS **/
  float val = 0.0f;

  int oblock_m = blockIdx.x * BM;
  int oblock_n = blockIdx.y * BN;

  /** SMEM **/
  __shared__ float input_buf[BM][BK];
  __shared__ float weight_buf[BK][BN];

  /** LOOP OVER K **/
  for (int bk = 0; bk < in_channels; bk += BK) {
    // load input
    for (int ld_input = 0; ld_input < LDPT_INPUT; ld_input++) {
      input_buf[threadIdx.x][threadIdx.y * LDPT_INPUT + ld_input] = input[
        in_channels * (oblock_m + threadIdx.x)
        + bk + threadIdx.y * LDPT_INPUT + ld_input
      ];
    }

    // load weight
    for (int ld_weight = 0; ld_weight < LDPT_WEIGHT; ld_weight++) {
      weight_buf[threadIdx.x * LDPT_WEIGHT + ld_weight][threadIdx.y] = weight[
        in_channels * (oblock_n + threadIdx.y)
        + bk + threadIdx.x * LDPT_WEIGHT + ld_weight
      ];
    }

    __syncthreads();

    // compute
    for (int k = 0; k < BK; k++) {
      val += weight_buf[k][threadIdx.y] * input_buf[threadIdx.x][k];
    }

    __syncthreads();
  }

  /** STORE **/
  val += bias[oblock_n + threadIdx.y];
  if (relu && val < 0.0f) val = 0.0f;
  output[out_channels * (oblock_m + threadIdx.x) + oblock_n + threadIdx.y] = val;

}

static __global__ void layernorm_cuda(
  float *input, float *gamma, float *beta, float *output,
  int num_batch, int num_channels, int len_input
) {

  int now_batch = blockIdx.x;
  int single_input_size = num_channels * len_input;
  int single_output_size = num_channels * len_input;

  int thread_stride = LNORM_CHPT * len_input;

  __shared__ float sum1[LNORM_TPB];
  __shared__ float sum2[LNORM_TPB];

  float psum1 = 0.0f, psum2 = 0.0f;
  for (int i = 0; i < thread_stride; ++i) {
    psum1 += input[now_batch * single_input_size + threadIdx.x * thread_stride + i];
    psum2 += input[now_batch * single_input_size + threadIdx.x * thread_stride + i]
      * input[now_batch * single_input_size + threadIdx.x * thread_stride + i];
  }

  __syncthreads();

  sum1[threadIdx.x] = psum1;
  sum2[threadIdx.x] = psum2;
  
  if (threadIdx.x == 0) {
    for (int i = 1; i < LNORM_TPB; i++) {
      sum1[0] += sum1[i];
      sum2[0] += sum2[i];
    }
  }

  __syncthreads();

  float mean1 = sum1[0] / (float)single_input_size;
  float mean2 = sum2[0] / (float)single_input_size;

  float var = mean2 - mean1 * mean1;

  for (int i = 0; i < thread_stride; ++i) {
    output[now_batch * single_output_size + threadIdx.x * thread_stride + i]
    = (input[now_batch * single_input_size + threadIdx.x * thread_stride + i] - mean1)
    / sqrtf(var + 1e-5) * gamma[threadIdx.x * thread_stride + i] + beta[threadIdx.x * thread_stride + i];
  }
  
}

static __global__ void maxpool1d_k3_cuda(
  float *input, float *output,
  int num_batch, int num_channels, int len_input,
  int relu
) {
  int POOL_SIZE = 3;
  int single_input_size = num_channels * len_input;
  int single_output_size = num_channels * (len_input / POOL_SIZE);
  int len_output = len_input / POOL_SIZE;

  int now_batch = blockIdx.x;
  int now_ol = threadIdx.x;
  for (int oc = 0; oc < num_channels; ++oc) {
    float mx = -1e99;
    for (int ks = 0; ks < POOL_SIZE; ++ks) {
      float val = input[now_batch * single_input_size + oc * len_input + ks + now_ol * POOL_SIZE];
      if (val > mx) mx = val;
    }
    if (relu && mx < 0.0f) mx = 0.0f;
    output[now_batch * single_output_size + oc * len_output + now_ol] = mx;
  }
}

static __global__ void argmax_f4_inplace_cuda(float *input) {
  int now_batch = threadIdx.x;
  int single_input_size = 4;
  float *input_offset = input + now_batch * single_input_size;

  if (blockIdx.x == 0) {
    int max_idx = 0;
    float max_val = -1e99;
    for (int i = 0; i < 4; ++i) {
      if (input_offset[i] > max_val) {
        max_val = input_offset[i];
        max_idx = i;
      }
    }
    input_offset[0] = max_idx;
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
  cudaStream_t _gpu_stream;
  int _gpu_idx;

  // Queue
  float *input_to_process;
  int num_input_ready;
  int num_input_processed;
  pthread_mutex_t mutex_queue;
  pthread_cond_t cond_queue;

  float *pop();
  void inference(float *popped_input);

  // Runner
  static void *run_func(void *arg);
  pthread_t thread;

  // Activations
  float *a_input_gpu, *a_conv1_gpu;
  float *a_layernorm1_gpu;
  float *a_pool1_gpu, *a_conv2_gpu;
  float *a_pool2_gpu, *a_conv3_gpu;
  float *a_conv4_gpu;
  float *a_conv5_gpu;
  float *a_conv6_gpu;
  float *a_layernorm6_gpu;
  float *a_collapse_gpu;
  float *a_linear1_gpu;
  float *a_linear2_gpu;
  float *a_linear3_gpu;
  float *a_linear3;

  // Parameters
  float *w_conv1_gpu, *b_conv1_gpu;
  float *w_conv2_gpu, *b_conv2_gpu;
  float *w_conv3_gpu, *b_conv3_gpu;
  float *w_conv4_gpu, *b_conv4_gpu;
  float *w_conv5_gpu, *b_conv5_gpu;
  float *w_conv6_gpu, *b_conv6_gpu;
  float *w_fc1_gpu, *b_fc1_gpu;
  float *w_fc2_gpu, *b_fc2_gpu;
  float *w_fc3_gpu, *b_fc3_gpu;
  float *gamma_conv1_gpu, *beta_conv1_gpu;
  float *gamma_conv6_gpu, *beta_conv6_gpu;

  // Debugging
  #if DEBUG == 1
  int never_inferenced;
  #endif
};

ComputeEngine *compute_engines[NGPU];

ComputeEngine::ComputeEngine(float *parameter_, int num_input_, int gpu_idx_) {
  // Initialize member variables
  input_buf = nullptr;
  output_buf = nullptr;
  num_input = num_input_;

  // Initialize GPU info
  _gpu_idx = gpu_idx_;

  // Initialize queue
  num_input_ready = 0;
  num_input_processed = 0;
  pthread_mutex_init(&mutex_queue, NULL);
  pthread_cond_init(&cond_queue, NULL);

  // Initialize CUDA
  CHECK_CUDA(cudaSetDevice(_gpu_idx));
  CHECK_CUDA(cudaStreamCreate(&_gpu_stream));

  // Initialize parameters
  CHECK_CUDA(cudaSetDevice(_gpu_idx));
  CHECK_CUDA(cudaMalloc(&a_input_gpu, COMPUTE_BATCH_SIZE * 70 * 1014 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv1_gpu, COMPUTE_BATCH_SIZE * 256 * 1008 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_layernorm1_gpu, COMPUTE_BATCH_SIZE * 256 * 1008 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_pool1_gpu, COMPUTE_BATCH_SIZE * 256 * 336 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv2_gpu, COMPUTE_BATCH_SIZE * 256 * 330 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_pool2_gpu, COMPUTE_BATCH_SIZE * 256 * 110 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv3_gpu, COMPUTE_BATCH_SIZE * 256 * 108 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv4_gpu, COMPUTE_BATCH_SIZE * 256 * 106 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv5_gpu, COMPUTE_BATCH_SIZE * 256 * 104 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_conv6_gpu, COMPUTE_BATCH_SIZE * 256 * 102 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_layernorm6_gpu, COMPUTE_BATCH_SIZE * 256 * 102 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_collapse_gpu, COMPUTE_BATCH_SIZE * 8704 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_linear1_gpu, COMPUTE_BATCH_SIZE * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_linear2_gpu, COMPUTE_BATCH_SIZE * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&a_linear3_gpu, COMPUTE_BATCH_SIZE * 4 * sizeof(float)));
  
  CHECK_CUDA(cudaMalloc(&w_conv1_gpu, 256 * 70 * 7 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv1_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_conv2_gpu, 256 * 256 * 7 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv2_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_conv3_gpu, 256 * 256 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv3_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_conv4_gpu, 256 * 256 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv4_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_conv5_gpu, 256 * 256 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv5_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_conv6_gpu, 256 * 256 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_conv6_gpu, 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_fc1_gpu, 1024 * 8704 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_fc1_gpu, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_fc2_gpu, 1024 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_fc2_gpu, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&w_fc3_gpu, 4 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b_fc3_gpu, 4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gamma_conv1_gpu, 256 * 1008 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&beta_conv1_gpu, 256 * 1008 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gamma_conv6_gpu, 256 * 102 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&beta_conv6_gpu, 256 * 102 * sizeof(float)));

  CHECK_CUDA(cudaMemcpyAsync(
    w_conv1_gpu, parameter_ + OFFSET0, 256 * 70 * 7 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv1_gpu, parameter_ + OFFSET1, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    gamma_conv1_gpu, parameter_ + OFFSET2, 256 * 1008 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    beta_conv1_gpu, parameter_ + OFFSET3, 256 * 1008 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_conv2_gpu, parameter_ + OFFSET4, 256 * 256 * 7 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv2_gpu, parameter_ + OFFSET5, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_conv3_gpu, parameter_ + OFFSET6, 256 * 256 * 3 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv3_gpu, parameter_ + OFFSET7, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_conv4_gpu, parameter_ + OFFSET8, 256 * 256 * 3 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv4_gpu, parameter_ + OFFSET9, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_conv5_gpu, parameter_ + OFFSET10, 256 * 256 * 3 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv5_gpu, parameter_ + OFFSET11, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_conv6_gpu, parameter_ + OFFSET12, 256 * 256 * 3 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_conv6_gpu, parameter_ + OFFSET13, 256 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    gamma_conv6_gpu, parameter_ + OFFSET14, 256 * 102 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    beta_conv6_gpu, parameter_ + OFFSET15, 256 * 102 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_fc1_gpu, parameter_ + OFFSET16, 1024 * 8704 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_fc1_gpu, parameter_ + OFFSET17, 1024 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_fc2_gpu, parameter_ + OFFSET18, 1024 * 1024 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_fc2_gpu, parameter_ + OFFSET19, 1024 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    w_fc3_gpu, parameter_ + OFFSET20, 4 * 1024 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  CHECK_CUDA(cudaMemcpyAsync(
    b_fc3_gpu, parameter_ + OFFSET21, 4 * sizeof(float),
    cudaMemcpyHostToDevice, _gpu_stream
  ));
  
  CHECK_CUDA(cudaStreamSynchronize(_gpu_stream));
  
  // Initialize activations
  a_linear3 = (float *)calloc(COMPUTE_BATCH_SIZE * 4, sizeof(float));

  // Debugging
  #if DEBUG == 1
  never_inferenced = 1;
  #endif
}

ComputeEngine::~ComputeEngine() {
  pthread_mutex_destroy(&mutex_queue);
  pthread_cond_destroy(&cond_queue);

  free(a_linear3);
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

float *ComputeEngine::pop() {
  float *pop_input;
  pthread_mutex_lock(&mutex_queue);
  if (num_input_ready == 0) pthread_cond_wait(&cond_queue, &mutex_queue);
  pop_input = input_to_process;
  input_to_process += POP_BATCH_SIZE * VOCAB_SIZE * MAX_LENGTH;
  num_input_ready -= POP_BATCH_SIZE;
  pthread_mutex_unlock(&mutex_queue);
  return pop_input;
}

void ComputeEngine::inference(float *popped_input) {
  #if DEBUG == 1
  if (never_inferenced) {
    DEBUG_PRINT("GPU %d Inference started: %f\n", _gpu_idx, get_time());
    never_inferenced = 0;
  }
  #endif

  int num_input = POP_BATCH_SIZE;

  CHECK_CUDA(cudaSetDevice(_gpu_idx));
  
  for (int batch = 0; batch < num_input; batch+=COMPUTE_BATCH_SIZE) {

    int now_batch_size = COMPUTE_BATCH_SIZE;

    // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
    {
      CHECK_CUDA(cudaMemcpyAsync(
        a_input_gpu, popped_input + batch * VOCAB_SIZE * MAX_LENGTH,
        now_batch_size * 70 * 1014 * sizeof(float),
        cudaMemcpyHostToDevice, _gpu_stream
      ));

      dim3 grid(CEIL_DIV(256, C1D_K7_BM), CEIL_DIV(1008, C1D_K7_BN));
      dim3 block(C1D_K7_BM * C1D_K7_BN);
      conv1d_k7_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_input_gpu, w_conv1_gpu, b_conv1_gpu, a_conv1_gpu,
        now_batch_size, 1008, 70, 256,
        0
      );

      layernorm_cuda<<<now_batch_size, LNORM_TPB, 0, _gpu_stream>>>(
        a_conv1_gpu, gamma_conv1_gpu, beta_conv1_gpu, a_layernorm1_gpu,
        now_batch_size, 256, 1008
      );

      maxpool1d_k3_cuda<<<now_batch_size, 1008/3, 0, _gpu_stream>>>(
        a_layernorm1_gpu, a_pool1_gpu,
        now_batch_size, 256, 1008,
        1
      );
    }

    // Conv block 2 : Conv1d + ReLU + MaxPool1d
    {
      dim3 grid(CEIL_DIV(256, C1D_K7_BM), CEIL_DIV(330, C1D_K7_BN));
      dim3 block(C1D_K7_BM * C1D_K7_BN);
      conv1d_k7_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_pool1_gpu, w_conv2_gpu, b_conv2_gpu, a_conv2_gpu,
        now_batch_size, 330, 256, 256,
        1
      );

      maxpool1d_k3_cuda<<<now_batch_size, 330/3, 0, _gpu_stream>>>(
        a_conv2_gpu, a_pool2_gpu,
        now_batch_size, 256, 330,
        0
      );
    }

    // Conv block 3 : Conv1d + ReLU
    {
      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(108, C1D_K3_BN));
      dim3 block(C1D_K3_BM * C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_pool2_gpu, w_conv3_gpu, b_conv3_gpu, a_conv3_gpu,
        now_batch_size, 108, 256, 256,
        1
      );

    }

    // Conv block 4 : Conv1d + ReLU
    {

      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(106, C1D_K3_BN));
      dim3 block(C1D_K3_BM * C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_conv3_gpu, w_conv4_gpu, b_conv4_gpu, a_conv4_gpu,
        now_batch_size, 106, 256, 256,
        1
      );
    }

    // Conv block 5 : Conv1d + ReLU
    {
      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(104, C1D_K3_BN));
      dim3 block(C1D_K3_BM * C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_conv4_gpu, w_conv5_gpu, b_conv5_gpu, a_conv5_gpu,
        now_batch_size, 104, 256, 256,
        1
      );
    }


    // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
    {
      dim3 grid(CEIL_DIV(256, C1D_K3_BM), CEIL_DIV(102, C1D_K3_BN));
      dim3 block(C1D_K3_BM * C1D_K3_BN);
      conv1d_k3_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_conv5_gpu, w_conv6_gpu, b_conv6_gpu, a_conv6_gpu,
        now_batch_size, 102, 256, 256,
        0
      );

      layernorm_cuda<<<now_batch_size, LNORM_TPB, 0, _gpu_stream>>>(
        a_conv6_gpu, gamma_conv6_gpu, beta_conv6_gpu, a_layernorm6_gpu,
        now_batch_size, 256, 102
      );

      maxpool1d_k3_cuda<<<now_batch_size, 102/3, 0, _gpu_stream>>>(
        a_layernorm6_gpu, a_collapse_gpu,
        now_batch_size, 256, 102,
        1
      );

    }

    // FC block 1 : Linear + ReLU
    {
      dim3 grid(CEIL_DIV(now_batch_size, LIN_REG_BM), CEIL_DIV(1024, LIN_REG_BN));
      dim3 block(LIN_REG_BM, LIN_REG_BN);
      linear_reg_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_collapse_gpu, w_fc1_gpu, b_fc1_gpu, a_linear1_gpu,
        now_batch_size, 8704, 1024,
        1
      );
    }

    // FC block 2 : Linear + ReLU
    {
      dim3 grid(CEIL_DIV(now_batch_size, LIN_REG_BM), CEIL_DIV(1024, LIN_REG_BN));
      dim3 block(LIN_REG_BM, LIN_REG_BN);
      linear_reg_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_linear1_gpu, w_fc2_gpu, b_fc2_gpu, a_linear2_gpu,
        now_batch_size, 1024, 1024,
        1
      );
    }

    // FC block 3 : Linear
    {
      dim3 grid(CEIL_DIV(now_batch_size, LIN_NAIVE_BM), CEIL_DIV(4, LIN_NAIVE_BN));
      dim3 block(LIN_NAIVE_BM, LIN_NAIVE_BN);
      linear_naive_cuda<<<grid, block, 0, _gpu_stream>>>(
        a_linear2_gpu, w_fc3_gpu, b_fc3_gpu, a_linear3_gpu,
        now_batch_size, 1024, 4,
        0
      );

      argmax_f4_inplace_cuda<<<1, now_batch_size, 0, _gpu_stream>>>(a_linear3_gpu);

      CHECK_CUDA(cudaMemcpyAsync(
        a_linear3, a_linear3_gpu, now_batch_size * 4 * sizeof(float),
        cudaMemcpyDeviceToHost, _gpu_stream
      ));

      CHECK_CUDA(cudaStreamSynchronize(_gpu_stream));
    }

    for (int b = 0; b < now_batch_size; b++) {
      output_buf[num_input_processed + b] = a_linear3[b * 4];
    }

    num_input_processed = num_input_processed + now_batch_size;
  }
}

void *ComputeEngine::run_func(void *arg) {
  ComputeEngine *engine = (ComputeEngine *) arg;
  while (engine->num_input_processed < engine->num_input) {
    float *popped_input = engine->pop();
    engine->inference(popped_input);
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

void classifier_root(float *input_, float *output_, int N) {
  // Initialize CE
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->set_input(input_ + ce_idx * N / mpi_size / NGPU * VOCAB_SIZE * MAX_LENGTH);
    compute_engines[ce_idx]->set_output(output_ + ce_idx * N / mpi_size / NGPU);
    compute_engines[ce_idx]->run();
  }
  DEBUG_PRINT("Compute engines initialized: %f\n", get_time());

  // Compute
  for (int wl_pushed = 0; wl_pushed < N / mpi_size; wl_pushed += PUSH_BATCH_SIZE) {
    int wl_to_push = min(PUSH_BATCH_SIZE, N / mpi_size - wl_pushed);
    for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
      compute_engines[ce_idx]->push(wl_to_push / NGPU);
    }
  }

  // Scatter input & initialize memory
  DEBUG_PRINT("Scatter input: %f\n", get_time());
  int scv_displacements[MAX_MPI_SIZE];
  int scv_counts[MAX_MPI_SIZE];

  for (int node_idx = 0; node_idx < mpi_size; ++node_idx) {
    scv_displacements[node_idx] = node_idx * N / mpi_size * VOCAB_SIZE * MAX_LENGTH;
    scv_counts[node_idx] = PUSH_BATCH_SIZE * VOCAB_SIZE * MAX_LENGTH;
  }

  for (int scv_idx = 0; scv_idx < N / mpi_size / NGPU; scv_idx += PUSH_BATCH_SIZE) {
    for (int gpu_idx = 0; gpu_idx < NGPU; gpu_idx++) {
      float *now_input = input_
       + scv_idx * VOCAB_SIZE * MAX_LENGTH
       + gpu_idx * N / mpi_size / NGPU * VOCAB_SIZE * MAX_LENGTH;
      
      MPI_Scatterv(
        now_input, scv_counts, scv_displacements, MPI_FLOAT,
        MPI_IN_PLACE, scv_counts[mpi_rank], MPI_FLOAT,
        0, MPI_COMM_WORLD
      );
    }
  }
  DEBUG_PRINT("Scatter input done: %f\n", get_time());

  // Wait completion
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->join();
  }
  DEBUG_PRINT("Computation complete: %f\n", get_time());

  // Gather output
  MPI_Gather(
    MPI_IN_PLACE, N / mpi_size, MPI_FLOAT,
    output_, N / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD
  );
  DEBUG_PRINT("Gathered output: %f\n", get_time());

}

void classifier_nonroot(float *input_, float *output_, int N) {
  // Initialize memory
  DEBUG_PRINT("Init mem: %f\n", get_time());
  if (input_ == nullptr) 
    input_ = (float *) calloc(
      N * VOCAB_SIZE * MAX_LENGTH / mpi_size, 
      sizeof(float)
    );
  if (output_ == nullptr) 
    output_ = (float *) calloc(N / mpi_size, sizeof(float));

  // Start compute engines
  DEBUG_PRINT("Init CE: %f\n", get_time());
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->set_input(input_ + ce_idx * N / mpi_size / NGPU * VOCAB_SIZE * MAX_LENGTH);
    compute_engines[ce_idx]->set_output(output_ + ce_idx * N / mpi_size / NGPU);
    compute_engines[ce_idx]->run();
  }

  // Scatter input
  DEBUG_PRINT("Scatter input: %f\n", get_time());
  int scv_displacements[MAX_MPI_SIZE];
  int scv_counts[MAX_MPI_SIZE];

  for (int node_idx = 0; node_idx < mpi_size; ++node_idx) {
    scv_displacements[node_idx] = node_idx * N / mpi_size * VOCAB_SIZE * MAX_LENGTH;
    scv_counts[node_idx] = PUSH_BATCH_SIZE * VOCAB_SIZE * MAX_LENGTH;
  }

  for (int scv_idx = 0; scv_idx < N / mpi_size / NGPU; scv_idx += PUSH_BATCH_SIZE) {
    for (int gpu_idx = 0; gpu_idx < NGPU; gpu_idx++) {
      float *now_input = input_
       + scv_idx * VOCAB_SIZE * MAX_LENGTH
       + gpu_idx * N / mpi_size / NGPU * VOCAB_SIZE * MAX_LENGTH;
      
      MPI_Scatterv(
        MPI_IN_PLACE, scv_counts, scv_displacements, MPI_FLOAT,
        now_input, scv_counts[mpi_rank], MPI_FLOAT,
        0, MPI_COMM_WORLD
      );

      compute_engines[gpu_idx]->push(PUSH_BATCH_SIZE);
    }
  }
  DEBUG_PRINT("Scatter input done: %f\n", get_time());

  // Wait completion
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    compute_engines[ce_idx]->join();
  }
  DEBUG_PRINT("Computation complete: %f\n", get_time());

  // Gather output
  MPI_Gather(
    output_, N / mpi_size, MPI_FLOAT,
    MPI_IN_PLACE, N / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD
  );
  DEBUG_PRINT("Gathered output: %f\n", get_time());
}

void classifier(float *input_, float *output_, int N) {
  DEBUG_PRINT("Start classifier: %f\n", get_time());
  if (iam_root) classifier_root(input_, output_, N);
  else classifier_nonroot(input_, output_, N);
  DEBUG_PRINT("Finish classifier: %f\n", get_time());
}

void finalize_classifier() {
  for (int ce_idx = 0; ce_idx < NGPU; ++ce_idx) {
    delete compute_engines[ce_idx];
  }
}

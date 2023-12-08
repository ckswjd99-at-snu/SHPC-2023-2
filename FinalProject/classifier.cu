#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <pthread.h>
#include <cassert>

#include "classifier.h"
#include "util.h"


/** SECTION: Constants and hyperparameters **/
#define NUM_PARAMETER (OFFSET21 + 4)

static int mpi_size, mpi_rank;
static char processor_name[MPI_MAX_PROCESSOR_NAME];
static int iam_root;


/** SECTION: DEBUGGING **/
#define DEBUG 0
#if DEBUG == 1
#define DEBUG_PRINT(...) do { \
  fprintf(stderr, "(%s|rank=%d) ", processor_name, mpi_rank); \
  fprintf(stderr, __VA_ARGS__); \
} while (0)
#else
#define DEBUG_PRINT(...)
#endif


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


/** SECTION: Operators **/
void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride = 1, int padding = 0, int dilation = 1,
            bool has_bias = true) {
  int out_channels = weight->shape[0];
  int in_channels = weight->shape[1];
  int kernel_size = weight->shape[2];
  int input_length = input->shape[2];
  int output_length =
      (input->shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  for (int oc = 0; oc < out_channels; ++oc) {
    for (int ol = 0; ol < output_length; ++ol) {
      float val = 0.0f;
      int offset = ol;
      for (int ic = 0; ic < in_channels; ++ic) {
        for (int ks = 0; ks < kernel_size; ++ks) {
          val += weight->buf[oc * in_channels * kernel_size + ic * kernel_size + ks] *
                 input->buf[ic * input_length + ks + offset];
        }
      }
      if (has_bias) val += bias->buf[oc];
      output->buf[oc * output_length + ol] = val;
    }
  }
}

void relu(Tensor *input, Tensor *output) {
  for (int i = 0; i < input->num_elem(); ++i) {
    if (input->buf[i] > 0.0f)
      output->buf[i] = input->buf[i];
    else
      output->buf[i] = 0.0f;
  }
}

void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride) {
  int IL = input->shape[2];
  int OC = output->shape[1];
  int OL = output->shape[2];

  for (int oc = 0; oc < OC; ++oc) {
    for (int ol = 0; ol < OL; ++ol) {
      float mx = -1e99;
      for (int ks = 0; ks < kernel_size; ++ks) {
        float val = input->buf[oc * IL + ks + ol * stride];
        if (val > mx) mx = val;
      }
      output->buf[oc * OL + ol] = mx;
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
  int IC = input->shape[1];
  int OC = output->shape[1];

  for (int oc = 0; oc < OC; ++oc) {
    float val = 0.0;
    for (int ic = 0; ic < IC; ++ic) {
      val += input->buf[ic] * weight->buf[oc * IC + ic];
    }
    if (has_bias) val += bias->buf[oc];
    output->buf[oc] = val;
  }
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  // E[X], E[X^2]
  float sum1 = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < input->num_elem(); ++i) {
      sum1 += input->buf[i];
      sum2 += input->buf[i] * input->buf[i];
  }
  float mean1 = sum1 / (float)input->num_elem();
  float mean2 = sum2 / (float)input->num_elem();

  // V[X]
  float var = mean2 - mean1 * mean1; 

  // Normalization
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = (input->buf[i] - mean1) / sqrtf(var + 1e-5) * gamma->buf[i] + beta->buf[i];
  }
}


/** SECTION: COMPUTE_ENGINE **/
#define COMPUTE_BATCH_SIZE 10
#define COMPUTE_MAX_QUEUE_SIZE 16

struct ComputeEngine {
public:
  ComputeEngine(float *parameter, int num_input);
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
  Tensor *w_conv1, *w_conv2, *w_conv3, *w_conv4, *w_conv5, *w_conv6, *b_conv1,
      *b_conv2, *b_conv3, *b_conv4, *b_conv5, *b_conv6, *w_fc1, *w_fc2, *w_fc3,
      *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

  // Activations
  Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
  Tensor *a_conv2, *a_relu2, *a_pool2;
  Tensor *a_conv3, *a_relu3;
  Tensor *a_conv4, *a_relu4;
  Tensor *a_conv5, *a_relu5;
  Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
  Tensor *a_collapse;
  Tensor *a_linear1, *a_relu7;
  Tensor *a_linear2, *a_relu8;
  Tensor *a_linear3;
};

ComputeEngine *compute_engine;  // Singleton

ComputeEngine::ComputeEngine(float *parameter_, int num_input_) {
  if (compute_engine != nullptr) {
    printf("ComputeEngine is a singleton class\n");
    exit(1);
  }
  compute_engine = this;

  // Initialize member variables
  input_buf = nullptr;
  output_buf = nullptr;
  num_input = num_input_;

  // Initialize queue
  num_input_ready = 0;
  num_input_processed = 0;
  pthread_mutex_init(&mutex_queue, NULL);
  pthread_cond_init(&cond_queue, NULL);

  // Initialize parameters
  w_conv1 = new Tensor({256, 70, 7}, parameter_ + OFFSET0);
  b_conv1 = new Tensor({256}, parameter_ + OFFSET1);
  gamma_conv1 = new Tensor({256, 1008}, parameter_ + OFFSET2);
  beta_conv1 = new Tensor({256, 1008}, parameter_ + OFFSET3);
  w_conv2 = new Tensor({256, 256, 7}, parameter_ + OFFSET4);
  b_conv2 = new Tensor({256}, parameter_ + OFFSET5);
  w_conv3 = new Tensor({256, 256, 3}, parameter_ + OFFSET6);
  b_conv3 = new Tensor({256}, parameter_ + OFFSET7);
  w_conv4 = new Tensor({256, 256, 3}, parameter_ + OFFSET8);
  b_conv4 = new Tensor({256}, parameter_ + OFFSET9);
  w_conv5 = new Tensor({256, 256, 3}, parameter_ + OFFSET10);
  b_conv5 = new Tensor({256}, parameter_ + OFFSET11);
  w_conv6 = new Tensor({256, 256, 3}, parameter_ + OFFSET12);
  b_conv6 = new Tensor({256}, parameter_ + OFFSET13);
  gamma_conv6 = new Tensor({256, 102}, parameter_ + OFFSET14);
  beta_conv6 = new Tensor({256, 102}, parameter_ + OFFSET15);
  w_fc1 = new Tensor({1024, 8704}, parameter_ + OFFSET16);
  b_fc1 = new Tensor({1024}, parameter_ + OFFSET17);
  w_fc2 = new Tensor({1024, 1024}, parameter_ + OFFSET18);
  b_fc2 = new Tensor({1024}, parameter_ + OFFSET19);
  w_fc3 = new Tensor({4, 1024}, parameter_ + OFFSET20);
  b_fc3 = new Tensor({4}, parameter_ + OFFSET21);

  // Initialize activations
  a_conv1 = new Tensor({1, 256, 1008});
  a_layernorm1 = new Tensor({1, 256, 1008});
  a_relu1 = new Tensor({1, 256, 1008});
  a_pool1 = new Tensor({1, 256, 336});
  a_conv2 = new Tensor({1, 256, 330});
  a_relu2 = new Tensor({1, 256, 330});
  a_pool2 = new Tensor({1, 256, 110});
  a_conv3 = new Tensor({1, 256, 108});
  a_relu3 = new Tensor({1, 256, 108});
  a_conv4 = new Tensor({1, 256, 106});
  a_relu4 = new Tensor({1, 256, 106});
  a_conv5 = new Tensor({1, 256, 104});
  a_relu5 = new Tensor({1, 256, 104});
  a_conv6 = new Tensor({1, 256, 102});
  a_layernorm6 = new Tensor({1, 256, 102});
  a_relu6 = new Tensor({1, 256, 102});
  a_pool6 = new Tensor({1, 256, 34});
  a_collapse = new Tensor({1, 8704});
  a_linear1 = new Tensor({1, 1024});
  a_relu7 = new Tensor({1, 1024});
  a_linear2 = new Tensor({1, 1024});
  a_relu8 = new Tensor({1, 1024});
  a_linear3 = new Tensor({1, 4});
}

ComputeEngine::~ComputeEngine() {
  pthread_mutex_destroy(&mutex_queue);
  pthread_cond_destroy(&cond_queue);

  delete w_conv1; delete b_conv1;
  delete w_conv2; delete b_conv2;
  delete w_conv3; delete b_conv3;
  delete w_conv4; delete b_conv4;
  delete w_conv5; delete b_conv5;
  delete w_conv6; delete b_conv6;
  delete w_fc1; delete b_fc1;
  delete w_fc2; delete b_fc2;
  delete w_fc3; delete b_fc3;
  
  delete gamma_conv1; delete gamma_conv6;
  delete beta_conv1; delete beta_conv6;
  delete a_conv1; delete a_layernorm1; delete a_relu1; delete a_pool1;
  delete a_conv2; delete a_relu2; delete a_pool2;
  delete a_conv3; delete a_relu3;
  delete a_conv4; delete a_relu4;
  delete a_conv5; delete a_relu5;
  delete a_conv6; delete a_layernorm6; delete a_relu6; delete a_pool6;
  delete a_collapse;
  delete a_linear1; delete a_relu7;
  delete a_linear2; delete a_relu8;
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
  num_input = std::min(num_input_ready, COMPUTE_BATCH_SIZE);
  num_input_ready -= num_input;
  pthread_mutex_unlock(&mutex_queue);
  return num_input;
}

void ComputeEngine::inference(int num_input) {
  DEBUG_PRINT("Inference %d\n", num_input);
  
  for (int batch = 0; batch < num_input; batch++) {
    DEBUG_PRINT("Inference %d/%d\n", batch+1, num_input);
    Tensor *input = new Tensor({1, VOCAB_SIZE, MAX_LENGTH}, input_to_process + batch * VOCAB_SIZE * MAX_LENGTH);

    // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
    conv1d(input, w_conv1, b_conv1, a_conv1, 1, 0, 1, true);
    layernorm(a_conv1, gamma_conv1, beta_conv1, a_layernorm1);
    relu(a_layernorm1, a_relu1);
    maxpool1d(a_relu1, a_pool1, 3, 3);

    // Conv block 2 : Conv1d + ReLU + MaxPool1d
    conv1d(a_pool1, w_conv2, b_conv2, a_conv2, 1, 0, 1, true);
    relu(a_conv2, a_relu2);
    maxpool1d(a_relu2, a_pool2, 3, 3);

    // Conv block 3 : Conv1d + ReLU
    conv1d(a_pool2, w_conv3, b_conv3, a_conv3, 1, 0, 1, true);
    relu(a_conv3, a_relu3);

    // Conv block 4 : Conv1d + ReLU
    conv1d(a_relu3, w_conv4, b_conv4, a_conv4, 1, 0, 1, true);
    relu(a_conv4, a_relu4);

    // Conv block 5 : Conv1d + ReLU
    conv1d(a_relu4, w_conv5, b_conv5, a_conv5, 1, 0, 1, true);
    relu(a_conv5, a_relu5);

    // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
    conv1d(a_relu5, w_conv6, b_conv6, a_conv6, 1, 0, 1, true);
    layernorm(a_conv6, gamma_conv6, beta_conv6, a_layernorm6);
    relu(a_layernorm6, a_relu6);
    maxpool1d(a_relu6, a_pool6, 3, 3);

    // Collapse
    collapse(a_pool6, a_collapse);

    // FC block 1 : Linear + ReLU
    linear(a_collapse, w_fc1, b_fc1, a_linear1, true);
    relu(a_linear1, a_relu7);

    // FC block 2 : Linear + ReLU
    linear(a_relu7, w_fc2, b_fc2, a_linear2, true);
    relu(a_linear2, a_relu8);

    // FC block 3 : Linear
    linear(a_relu8, w_fc3, b_fc3, a_linear3, true);

    float max_val = -1e99f;
    int max_idx = 0;
    for (int i = 0; i < a_linear3->num_elem(); ++i) {
      if (a_linear3->buf[i] > max_val) {
        max_val = a_linear3->buf[i];
        max_idx = i;
      }
    }

    output_buf[num_input_processed++] = max_idx;
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

  compute_engine = new ComputeEngine(parameter, N / mpi_size);
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
  compute_engine->set_input(input_);
  compute_engine->set_output(output_);
  compute_engine->run();
  compute_engine->push(N / mpi_size);
  compute_engine->join();

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
  if (mpi_rank == 0) {
    delete compute_engine;
  }
}

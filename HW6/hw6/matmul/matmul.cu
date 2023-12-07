/******************************************************************************
 *  OPTIMIZATION NOTE
 * 
 *  [OPT TARGET]
 *  Total latency under 0.11 sec
 *  
 *  [OPT LOG]
 *  Last optimization: hyperparameter tuning
 *   - increase THREAD_M from 8 to 16
 *  Baseline latency:   0.098886 sec
 * 
 *  [LATENCY BREAKDOWN]
 *  0. total latency:       0.098886 sec (100.0%)
 *  1. MPI_Scatter:         0.000000 sec (  0.0%)
 *  2. MPI_Bcast:           0.000000 sec (  0.0%)
 *  3. cudaMemcpyAsync(B):  0.007043 sec (  7.1%)
 *  3. cudaMemcpyAsync(A):  0.025976 sec ( 26.3%)
 *  4. matmul_kernel:       0.053739 sec ( 54.4%)
 *  5. cudaMemcpyAsync(C):  0.000000 sec (  0.0%)
 *  6. MPI_Gather:          0.000000 sec (  0.0%)
 *  7. etc(error):          0.012128 sec ( 12.3%)
 * 
 *  => Plan: autotune hyperparameters
 *  
******************************************************************************/

#include "matmul.h"
#include "util.h"

#include <omp.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <pthread.h>

/** UTIL FUNCS **/

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


/** CUDA CONSTS **/
#define NGPU 4


/** KERNEL CONSTS **/
#define BLOCK_M     128
#define BLOCK_N     128
#define BLOCK_K     8
#define THREAD_M    16
#define THREAD_N    8
#define VEC_SIZE    4
#define NUM_TM      (BLOCK_M / THREAD_M)
#define NUM_TN      (BLOCK_N / THREAD_N)
#define NUM_THS     (NUM_TM * NUM_TN)
#define LDNK_STRD   (NUM_THS / (BLOCK_K / VEC_SIZE))
#define NUM_BDIM    ((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N))


/** GLOBALS **/
static int mpi_rank, mpi_world_size;
int M_node_start, M_node_end, M_node_size;
static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static cudaStream_t streams_mem[NGPU];
cudaEvent_t htod_event[NGPU], dtoh_event[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];


/** FUNCS **/
static __global__ void matmul_kernel(
  float *A, float *B, float *C, int M, int N, int K
) {
  // SMEM ALLOC
  __shared__ float Asub[BLOCK_K][BLOCK_M+4];
  __shared__ float Bsub[BLOCK_K][BLOCK_N+4];

  // REG ALLOC
  float sum[THREAD_M][THREAD_N] = {0.0};
  float tempA[THREAD_M];
  float tempB[THREAD_N];

  float *A_offset = A + K * blockIdx.x * BLOCK_M;
  float *B_offset = B + blockIdx.y * BLOCK_N;
  float *C_offset = C + N * blockIdx.x * BLOCK_M + blockIdx.y * BLOCK_N;

  int tx = threadIdx.x / NUM_TN;
  int ty = threadIdx.x % NUM_TN;

  int lnk_A = threadIdx.x / (BLOCK_K / VEC_SIZE);
  int lk_A = threadIdx.x % (BLOCK_K / VEC_SIZE);

  int lnk_B = threadIdx.x / BLOCK_K;
  int lk_B = threadIdx.x % BLOCK_K;

  // ITER THROUGH K
  for (int k=0; k<K; k+=BLOCK_K) {
    for (int lda=0; lda<BLOCK_M; lda+=LDNK_STRD) {
      float4 A_offset_temp = *reinterpret_cast<float4 *>(&A_offset[K * (lda + lnk_A) + lk_A * VEC_SIZE]);
      Asub[lk_A * VEC_SIZE + 0][lda + lnk_A] = A_offset_temp.x;
      Asub[lk_A * VEC_SIZE + 1][lda + lnk_A] = A_offset_temp.y;
      Asub[lk_A * VEC_SIZE + 2][lda + lnk_A] = A_offset_temp.z;
      Asub[lk_A * VEC_SIZE + 3][lda + lnk_A] = A_offset_temp.w;
    }
    for (int ldb=0; ldb<BLOCK_N; ldb+=LDNK_STRD) {
      *reinterpret_cast<float4 *>(&Bsub[lk_B][ldb + lnk_B * VEC_SIZE])
      = *reinterpret_cast<float4 *>(&B_offset[N * lk_B + ldb + lnk_B * VEC_SIZE]);
    }
    
    __syncthreads();

    A_offset += BLOCK_K;
    B_offset += BLOCK_K * N;

    for (int bk=0; bk<BLOCK_K; bk++) {
      for (int tm=0; tm<THREAD_M; tm+=VEC_SIZE) {
        *reinterpret_cast<float4 *>(&tempA[tm])
         = *reinterpret_cast<float4 *>(&Asub[bk][THREAD_M * tx + tm]);
      }
      for (int tn=0; tn<THREAD_N; tn+=VEC_SIZE) {
        *reinterpret_cast<float4 *>(&tempB[tn])
         = *reinterpret_cast<float4 *>(&Bsub[bk][THREAD_N * ty + tn]);
      }

      for (int tm=0; tm<THREAD_M; tm++) {
        for (int tn=0; tn<THREAD_N; tn++) {
          sum[tm][tn] += tempA[tm] * tempB[tn];
        }
      }
    }

    __syncthreads();
  }

  for (int tm=0; tm<THREAD_M; tm++) {
    for (int tn=0; tn<THREAD_N; tn+=VEC_SIZE) {
      *reinterpret_cast<float4 *>(&C_offset[N * (THREAD_M * tx + tm) + THREAD_N * ty + tn])
      = *reinterpret_cast<float4 *>(&sum[tm][tn]);
    }
  }

} 

void matmul(float *A, float *B, float *C, int M, int N, int K) {

  const int NUM_WORKLOAD = 1;

  // #pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaMemcpyAsync(
      B_gpu[i], B, K * N * sizeof(float),
      cudaMemcpyHostToDevice, streams_mem[i]
    ));
  }

  #pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));

    for (int wl=0; wl<NUM_WORKLOAD; wl++) {

      int Mbegin_wl = Mbegin[i] + wl * (Mend[i] - Mbegin[i]) / NUM_WORKLOAD;
      int Mend_wl   = Mbegin[i] + (wl + 1) * (Mend[i] - Mbegin[i]) / NUM_WORKLOAD;
      
      // Async memcpy H->D on each GPU
      CHECK_CUDA(cudaMemcpyAsync(
        A_gpu[i], &A[Mbegin_wl * K],
        (Mend_wl - Mbegin_wl) * K * sizeof(float),
        cudaMemcpyHostToDevice, streams_mem[i]
      ));

      CHECK_CUDA(cudaEventRecord(htod_event[i], streams_mem[i]));
      CHECK_CUDA(cudaStreamWaitEvent(streams[i], htod_event[i], 0));

      // Run kernels asynchronously on each GPU
      dim3 blockDim(NUM_BDIM);
      dim3 gridDim(CEIL_DIV(Mend_wl - Mbegin_wl, BLOCK_M), CEIL_DIV(N, BLOCK_N));
      matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
          A_gpu[i], B_gpu[i], C_gpu[i], Mend_wl - Mbegin_wl, N, K);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaEventRecord(dtoh_event[i], streams[i]));
      CHECK_CUDA(cudaStreamWaitEvent(streams_mem[i], dtoh_event[i], 0));

      // Async memcpy D->H on each GPU
      CHECK_CUDA(cudaMemcpyAsync(
        &C[Mbegin_wl * N], C_gpu[i],
        (Mend_wl - Mbegin_wl) * N * sizeof(float),
        cudaMemcpyDeviceToHost, streams_mem[i]
      ));
    }
  }

  // Wait for all async jobs to finish
  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

}


void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  printf("[rank %d] Number of devices: %d\n", mpi_rank, ngpu);
  cudaDeviceProp props[4];
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
    printf("[rank %d] device %d: %s\n", mpi_rank, i, props[i].name);
  }

  M_node_start = M * mpi_rank / mpi_world_size;
  M_node_end = M * (mpi_rank + 1) / mpi_world_size;
  M_node_size = M_node_end - M_node_start;

  for (int i = 0; i < ngpu; i++) {
    Mbegin[i] = M_node_start + M_node_size * i / ngpu;
    Mend[i] = M_node_start + M_node_size * (i + 1) / ngpu;
    if (i == ngpu - 1) Mend[i] = M_node_end;
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaStreamCreate(&streams_mem[i]));
    CHECK_CUDA(cudaEventCreate(&htod_event[i]));
    CHECK_CUDA(cudaEventCreate(&dtoh_event[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}


void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    CHECK_CUDA(cudaStreamDestroy(streams_mem[i]));
    CHECK_CUDA(cudaEventDestroy(htod_event[i]));
    CHECK_CUDA(cudaEventDestroy(dtoh_event[i]));
  }
}

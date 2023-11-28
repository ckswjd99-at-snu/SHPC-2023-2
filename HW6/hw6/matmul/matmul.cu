/******************************************************************************
 *  OPTIMIZATION NOTE
 *  
 *  [OPT LOG]
 *  Last optimization:  allocating multiple elements per thread (2D)
 *  Baseline latency:   0.238121 sec -> 0.222934 sec (-0.015187 sec)
 *  matmul_kernel:      0.024486 sec -> 0.009299 sec (-0.015187 sec)
 * 
 *  [LATENCY BREAKDOWN]
 *  0. total latency:   0.222934 sec (100.0%)
 *  1. MPI_Scatter:     0.080922 sec ( 36.3%)
 *  2. MPI_Bcast:       0.015000 sec (  6.7%)
 *  3. cudaMemcpyAsync: 0.027967 sec ( 12.5%)
 *  4. matmul_kernel:   0.009299 sec (  4.2%)
 *  5. cudaMemcpyAsync: 0.003006 sec (  1.3%)
 *  6. MPI_Gather:      0.062237 sec ( 27.9%)
 * 
 *  => Plan: stop optimizing kernel, fine-grain MPI
 *  
******************************************************************************/

#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

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

#define BLOCK_M     128
#define BLOCK_N     128
#define BLOCK_K     8
#define THREAD_M    8
#define THREAD_N    8
#define NUM_TM      (BLOCK_M / THREAD_M)
#define NUM_TN      (BLOCK_N / THREAD_N)
#define NUM_THS     (NUM_TM * NUM_TN)
#define LDNK_STRD   (NUM_THS / BLOCK_K)

#define NUM_BDIM    ((BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N))

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  /* SMEM ALLOC */
  __shared__ float Asub[BLOCK_M][BLOCK_K];
  __shared__ float Bsub[BLOCK_K][BLOCK_N];

  /* REG ALLOC */
  float sum[THREAD_M][THREAD_N] = {0.0};
  float tempA[THREAD_M];
  float tempB[THREAD_N];

  float *A_offset = A + K * blockIdx.x * BLOCK_M;
  float *B_offset = B + blockIdx.y * BLOCK_N;
  float *C_offset = C + N * blockIdx.x * BLOCK_M + blockIdx.y * BLOCK_N;

  int tx = threadIdx.x / NUM_TN;
  int ty = threadIdx.x % NUM_TN;

  int lnk = threadIdx.x / BLOCK_K;
  int lk = threadIdx.x % BLOCK_K;

  /* ITER THROUGH K */
  for (int k=0; k<K; k+=BLOCK_K) {
    for (int lda=0; lda<BLOCK_M; lda+=LDNK_STRD) {
      Asub[lda + lnk][lk] = A_offset[K * (lda + lnk) + lk];
    }
    for (int ldb=0; ldb<BLOCK_N; ldb+=LDNK_STRD) {
      Bsub[lk][ldb + lnk] = B_offset[N * lk + ldb + lnk];
    }
    
    __syncthreads();

    A_offset += BLOCK_K;
    B_offset += BLOCK_K * N;

    for (int bk=0; bk<BLOCK_K; bk++) {
      for (int tm=0; tm<THREAD_M; tm++) {
        tempA[tm] = Asub[THREAD_M * tx + tm][bk];
      }
      for (int tn=0; tn<THREAD_N; tn++) {
        tempB[tn] = Bsub[bk][THREAD_N * ty + tn];
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
    for (int tn=0; tn<THREAD_N; tn++) {
      C_offset[N * (THREAD_M * tx + tm) + THREAD_N * ty + tn] = sum[tm][tn];
    }
  }

}

#define NGPU 4

int M_node_start, M_node_end, M_node_size;
static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Scatter mat A
  float *Abuf = (float *)A;
  if (mpi_rank == 0) {
    MPI_Scatter(A, M * K / mpi_world_size, MPI_FLOAT, MPI_IN_PLACE, M * K / mpi_world_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Scatter(NULL, M * K / mpi_world_size, MPI_FLOAT, Abuf + K * M_node_start, M * K / mpi_world_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  // Broadcast mat B
  float *Bbuf = (float *)B;
  MPI_Bcast(Bbuf, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Async memcpy H->D on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[Mbegin[i] * K],
                               (Mend[i] - Mbegin[i]) * K * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
  }

  // Run kernels asynchronously on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(NUM_BDIM);
    dim3 gridDim(CEIL_DIV(N, BLOCK_N), CEIL_DIV(Mend[i] - Mbegin[i], BLOCK_M));
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
        A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  // Async memcpy D->H on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[i] * N], C_gpu[i],
                               (Mend[i] - Mbegin[i]) * N * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[i]));
  }

  // Wait for all async jobs to finish
  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }

  // Gather mat C
  if (mpi_rank == 0) {
    MPI_Gather(MPI_IN_PLACE, M * N / mpi_world_size, MPI_FLOAT, C, M * N / mpi_world_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Gather(C + N * M_node_start, M * N / mpi_world_size, MPI_FLOAT, NULL, M * N / mpi_world_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
  }
}

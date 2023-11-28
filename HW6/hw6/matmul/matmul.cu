/******************************************************************************
 *  OPTIMIZATION NOTE
 *  
 *  [OPT LOG]
 *  Last optimization:  allocating multiple elements per thread
 *  Baseline latency:   0.274198 sec -> 0.238121 sec (-0.036077 sec)
 *  matmul_kernel:      0.060563 sec -> 0.024486 sec (-0.036077 sec)
 * 
 *  [LATENCY BREAKDOWN]
 *  0. total latency:   0.238121 sec (100.0%)
 *  1. MPI_Scatter:     0.080922 sec ( 34.0%)
 *  2. MPI_Bcast:       0.015000 sec (  6.3%)
 *  3. cudaMemcpyAsync: 0.027967 sec ( 11.7%)
 *  4. matmul_kernel:   0.024486 sec ( 10.3%)
 *  5. cudaMemcpyAsync: 0.003006 sec (  1.3%)
 *  6. MPI_Gather:      0.062237 sec ( 26.1%)
 * 
 *  => Plan: optimize kernel more, then fine-grain MPI
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

#define BLOCK_M     64
#define BLOCK_N     64
#define BLOCK_K     8
#define THREAD_M    8
#define NUM_BDIM    (BLOCK_M * BLOCK_N / THREAD_M)

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  /* SMEM ALLOC */
  __shared__ float Asub[BLOCK_M][BLOCK_K];
  __shared__ float Bsub[BLOCK_K][BLOCK_N];

  /* REG ALLOC */
  float sum[THREAD_M] = {0.0};

  float *A_offset = A + K * blockIdx.x * BLOCK_M;
  float *B_offset = B + blockIdx.y * BLOCK_N;
  float *C_offset = C + N * blockIdx.x * BLOCK_M + blockIdx.y * BLOCK_N;

  int tx = threadIdx.x / BLOCK_N;
  int ty = threadIdx.x % BLOCK_N;

  int lnk = threadIdx.x / BLOCK_K;
  int lk = threadIdx.x % BLOCK_K;

  /* ITER THROUGH K */
  for (int k=0; k<K; k+=BLOCK_K) {

    Asub[lnk][lk] = A_offset[K * lnk + lk];
    Bsub[lk][lnk] = B_offset[N * lk + lnk];

    __syncthreads();

    A_offset += BLOCK_K;
    B_offset += BLOCK_K * N;

    for (int bk=0; bk<BLOCK_K; bk++) {
      float tempB = Bsub[bk][ty];
      for (int tm=0; tm<THREAD_M; tm++) {
        sum[tm] += Asub[tx * THREAD_M + tm][bk] * tempB;
      }
    }

    __syncthreads();
  }

  for (int tm=0; tm<THREAD_M; tm++) {
    C_offset[N * (tx * THREAD_M + tm) + ty] = sum[tm];
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

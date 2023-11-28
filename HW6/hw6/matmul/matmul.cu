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
#define BLOCK_SIZE 32

static int mpi_rank, mpi_world_size;

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (x >= M || y >= N) return;
  float sum = 0.0;
  for (int k = 0; k < K; ++k) sum += A[x * K + k] * B[k * N + y];
  C[x * N + y] = sum;
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
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(Mend[i] - Mbegin[i], BLOCK_SIZE));
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

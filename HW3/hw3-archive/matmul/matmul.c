#define _GNU_SOURCE
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Naive CPU matrix multiplication
void matmul_singlethread(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K; k++) { c += A[i * K + k] * B[k * N + j]; }
      C[i * N + j] = c;
    }
  }
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {
  // Naive single-threaded matmul implementation
  matmul_singlethread(A, B, C, M, N, K);

  // TODO: Implement multi-threaded matmul
}

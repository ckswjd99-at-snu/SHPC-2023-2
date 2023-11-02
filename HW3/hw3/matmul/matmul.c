#define _GNU_SOURCE
#include "util.h"
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <immintrin.h>
#include <omp.h>

/* HYPERPARAMS */
#define MAC_M 64
#define MAC_N 64
#define MAC_K 4096

#define MIC_M 16
#define MIC_N 16
#define MIC_K 8

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


static void sgemm_reg(int M, int N, int K, float *A, float *B, float *C, int lda, int ldb, int ldc) {
  float cbuf[MAC_M * MAC_N];

  // Regular M
  for (int mac_m = 0; mac_m < M; mac_m += MAC_M) {
    // Regular N
    for (int mac_n = 0; mac_n < N; mac_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      #if __AVX512F__
      // MACRO KERNEL - AVX512
      // Implemented with AVX512
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < MAC_M; m++) {
          __m512 avec = _mm512_set1_ps(A[(mac_m + m) * K + k]);

          for (int n = 0; n < MAC_N; n += MIC_N) {
            __m512 bvec = _mm512_loadu_ps(&B[k * ldb + (mac_n + n)]);
            __m512 cvec = _mm512_loadu_ps(&cbuf[m * MAC_N + n]);
            cvec = _mm512_fmadd_ps(avec, bvec, cvec);
            _mm512_storeu_ps(&cbuf[m * MAC_N + n], cvec);
          }
        }
      }

      #else
      // MACRO KERNEL - NAIVE
      // Implemented code is naive, but compiler will automatically optimize here.
      for (int k = 0; k < K; k++) { 
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + k] * B[k * ldb + (mac_n + n)];
          }
        }
      }
      #endif

      for (int m = 0; m < MAC_M; m++) {
        memcpy(&C[(mac_m + m) * ldc + (mac_n)], &cbuf[m * MAC_N], sizeof(float) * MAC_N);
      }

    }
  }
  
}

static void sgemm(int M, int N, int K, float *A, float *B, float *C, int lda, int ldb, int ldc) {
  // printf("%d %d %d\n", M, N, K);
  // fflush(stdout);

  float cbuf[MAC_M * MAC_N];

  if (N % MAC_N == 0 && M % MAC_M == 0) {
    sgemm_reg(M, N, K, A, B, C, lda, ldb, ldc);
    return;
  }

  // Regular M
  for (int mac_m = 0; mac_m < (M - MAC_M + 1); mac_m += MAC_M) {
    // Regular N
    for (int mac_n = 0; mac_n < (N - MAC_N + 1); mac_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      #if __AVX512F__
      // MACRO KERNEL - AVX512
      // Implemented with AVX512
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < MAC_M; m++) {
          __m512 avec = _mm512_set1_ps(A[(mac_m + m) * K + k]);

          for (int n = 0; n < MAC_N; n += MIC_N) {
            __m512 bvec = _mm512_loadu_ps(&B[k * ldb + (mac_n + n)]);
            __m512 cvec = _mm512_loadu_ps(&cbuf[m * MAC_N + n]);
            cvec = _mm512_fmadd_ps(avec, bvec, cvec);
            _mm512_storeu_ps(&cbuf[m * MAC_N + n], cvec);
          }
        }
      }

      #else
      // MACRO KERNEL - NAIVE
      // Implemented code is naive, but compiler will automatically optimize here.
      for (int k = 0; k < K; k++) { 
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + k] * B[k * ldb + (mac_n + n)];
          }
        }
      }
      #endif

      for (int m = 0; m < MAC_M; m++) {
        memcpy(&C[(mac_m + m) * ldc + (mac_n)], &cbuf[m * MAC_N], sizeof(float) * MAC_N);
      }
    }

    // Remaining N
    if (N % MAC_N != 0) {
      int mac_n = (N / MAC_N) * MAC_N;
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MACRO KERNEL - NAIVE */
      for (int k = 0; k < K; k++) {

        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < N - mac_n; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + k] * B[k * ldb + (mac_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < N - mac_n; n++) {
          C[(mac_m + m) * ldc + (mac_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }
    
  }
  
  // Remaining M
  if (M % MAC_M != 0) {
    int mac_m = (M / MAC_M) * MAC_M;
    // Regular N
    for (int mac_n = 0; mac_n < (N - MAC_N + 1); mac_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MICRO KERNEL - NAIVE */
      for (int k = 0; k < K; k++) { 
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + k] * B[k * ldb + (mac_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < MAC_N; n++) {
          C[(mac_m + m) * ldc + (mac_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }

    // Remaining N

    if (N % MAC_N != 0) {
      int mac_n = (N / MAC_N) * MAC_N;
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MICRO KERNEL - NAIVE */
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < N - mac_n; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + k] * B[k * ldb + (mac_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < N - mac_n; n++) {
          C[(mac_m + m) * ldc + (mac_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }
  }
}

static void matmul_thread(const float *A, const float *B, float *C, int M, int N, int K, int num_threads, int rank) {
  /* SPLIT WORKLOADS */
  float *thr_A, *thr_B, *thr_C;
  int thr_M, thr_N, thr_K;
  
  // Split N
  thr_M = M;
  thr_N = N * (rank + 1) / num_threads - N * rank / num_threads;
  thr_K = K;
  
  thr_A = A;
  thr_B = B + N * rank / num_threads;
  thr_C = C + N * rank / num_threads;

  /* CALL GEMM */
  sgemm(thr_M, thr_N, thr_K, thr_A, thr_B, thr_C, K, N, N);
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {
  // Naive single-threaded matmul implementation
  // matmul_singlethread(A, B, C, M, N, K);

  // TODO: Implement multi-threaded matmul
  #pragma omp parallel for num_threads(num_threads)
  for (int t = 0; t < num_threads; ++t) {
    matmul_thread(A, B, C, M, N, K, num_threads, t);
  }
}

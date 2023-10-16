#define _GNU_SOURCE
#include "util.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

#define MAX_THREADS 256

/* HYPERPARAMS */
static const int MAC_M = 64;
static const int MAC_N = 64;
static const int MAC_K = 4096;

static const int MIC_M = 16;
static const int MIC_N = 16;
static const int MIC_K = 8;

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[MAX_THREADS];
static pthread_t threads[MAX_THREADS];


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

  int mac_M = MAC_M;
  int mac_N = MAC_N;

  // Regular M
  for (int mac_m = 0; mac_m < M; mac_m += MAC_M) {
    // Regular N
    for (int mac_n = 0; mac_n < N; mac_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      #if 1
      // MACRO KERNEL - ASM
      // Macro kernel with AVX512, manually handcrafted
      for (int thr_k = 0; thr_k < K; thr_k++) {
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n += 16) {
            // Load 16 elements from A and broadcast to a vector
            __m512 avec = _mm512_set1_ps(A[(mac_m + m) * K + thr_k]);
            
            // Load 16 elements from B
            __m512 bvec = _mm512_loadu_ps(&B[thr_k * ldb + (mac_n + n)]);
            
            // Multiply A and B and add to the cbuf
            __m512 cvec = _mm512_loadu_ps(&cbuf[m * MAC_N + n]);
            cvec = _mm512_fmadd_ps(avec, bvec, cvec);
            
            // Store the result back to cbuf
            _mm512_storeu_ps(&cbuf[m * MAC_N + n], cvec);
          }
        }
      }

      #else
      // MACRO KERNEL - NAIVE
      // Implemented code is naive, but compiler will automatically parallelize here with AVX codes.
      for (int thr_k = 0; thr_k < K; thr_k++) { 
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(mac_m + m) * K + thr_k] * B[thr_k * ldb + (mac_n + n)];
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

  int mac_M = MAC_M;
  int mac_N = MAC_N;

  if (N % MAC_N == 0 && M % MAC_M == 0) {
    sgemm_reg(M, N, K, A, B, C, lda, ldb, ldc);
    return;
  }

  // Regular M
  for (int thr_m = 0; thr_m < (M - MAC_M + 1); thr_m += MAC_M) {
    // Regular N
    for (int thr_n = 0; thr_n < (N - MAC_N + 1); thr_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MACRO KERNEL - NAIVE */
      for (int thr_k = 0; thr_k < K; thr_k++) { 
        
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(thr_m + m) * K + thr_k] * B[thr_k * ldb + (thr_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < MAC_N; n++) {
          C[(thr_m + m) * ldc + (thr_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }

    // Remaining N
    if (N % MAC_N != 0) {
      int thr_n = (N / MAC_N) * MAC_N;
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MACRO KERNEL - NAIVE */
      for (int thr_k = 0; thr_k < K; thr_k++) {

        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < N - thr_n; n++) {
            cbuf[m * MAC_N + n] += A[(thr_m + m) * K + thr_k] * B[thr_k * ldb + (thr_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < N - thr_n; n++) {
          C[(thr_m + m) * ldc + (thr_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }
    
  }
  
  // Remaining M
  if (M % MAC_M != 0) {
    int thr_m = (M / MAC_M) * MAC_M;
    // Regular N
    for (int thr_n = 0; thr_n < (N - MAC_N + 1); thr_n += MAC_N) {
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MICRO KERNEL - NAIVE */
      for (int thr_k = 0; thr_k < K; thr_k++) { 
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < MAC_N; n++) {
            cbuf[m * MAC_N + n] += A[(thr_m + m) * K + thr_k] * B[thr_k * ldb + (thr_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < MAC_N; n++) {
          C[(thr_m + m) * ldc + (thr_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }

    // Remaining N

    if (N % MAC_N != 0) {
      int thr_n = (N / MAC_N) * MAC_N;
      bzero(cbuf, sizeof(float) * MAC_M * MAC_N);

      /* MICRO KERNEL - NAIVE */
      for (int thr_k = 0; thr_k < K; thr_k++) {
        for (int m = 0; m < MAC_M; m++) {
          for (int n = 0; n < N - thr_n; n++) {
            cbuf[m * MAC_N + n] += A[(thr_m + m) * K + thr_k] * B[thr_k * ldb + (thr_n + n)];
          }
        }
      }

      for (int m = 0; m < MAC_M; m++) {
        for (int n = 0; n < N - thr_n; n++) {
          C[(thr_m + m) * ldc + (thr_n + n)] = cbuf[m * MAC_N + n];
        }
      }
    }
  }
}

static void *matmul_thread(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;

  const float *A = (*input).A;
  const float *B = (*input).B;
  float *C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  int num_threads = (*input).num_threads;
  int rank = (*input).rank;

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

  return NULL;
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  // Naive single-threaded matmul implementation
  // matmul_singlethread(A, B, C, M, N, K);

  /*
   * TODO: Complete multi-threaded matrix multiplication and remove the matmul_singlethread call
   */ 

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  // Spawn num_thread CPU threads
  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_thread, (void *)&args[t]);
    if (err) {
      fprintf(stderr, "pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  // Wait for spawned threads to terminate
  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      fprintf(stderr, "pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}

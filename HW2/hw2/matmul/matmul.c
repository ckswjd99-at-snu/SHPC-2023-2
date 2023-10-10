#define _GNU_SOURCE
#include "util.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

#define MAX_THREADS 256

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

  /* HYPERPARAMS */
  int MAC_M = 64;
  int MAC_N = 64;
  int MAC_K = 4096;

  int MIC_M = 16;
  int MIC_N = 16;
  int MIC_K = 8;

  /* SPLIT WORKLOADS */
  float *thr_A, *thr_B, *thr_C;
  int thr_M, thr_N, thr_K;
  
  // Split N
  thr_M = M;
  thr_N = N / num_threads;
  thr_K = K;
  if (N % num_threads != 0 && rank == num_threads - 1) thr_N += N & num_threads;
  
  thr_A = A;
  thr_B = B + thr_N * rank;
  thr_C = C + thr_N * rank;

  for (size_t thr_m = 0; thr_m < thr_M; thr_m++) {
    for (size_t thr_n = 0; thr_n < thr_N; thr_n++) {
      float c = 0.0;
      for (size_t thr_k = 0; thr_k < thr_K; thr_k++) { c += thr_A[thr_m * thr_K + thr_k] * thr_B[thr_k * N + thr_n]; }
      thr_C[thr_m * N + thr_n] = c;
    }
  }

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

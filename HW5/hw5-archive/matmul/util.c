#include "util.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

static double start_time[8];

void timer_init() { srand(time(NULL)); }

static double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

void alloc_mat(float **m, int R, int S) {
  *m = (float *)aligned_alloc(32, sizeof(float) * R * S);
  if (*m == NULL) {
    printf("Failed to allocate memory for mat.\n");
    exit(0);
  }
}

void rand_mat(float *m, int R, int S) {
  int N = R * S;
  for (int j = 0; j < N; j++) {
    m[j] = (float)rand() / RAND_MAX - 0.5;
  }
}

void zero_mat(float *m, int R, int S) {
  int N = R * S;
  memset(m, 0, sizeof(float) * N);
}

void print_mat(float *m, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%+.3f ", m[i * N + j]);
    }
    printf("\n");
  }
}

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  printf("Validating...\n");

  float *C_ans;
  alloc_mat(&C_ans, M, N);
  zero_mat(C_ans, M, N);
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if (fabsf(c - c_ans) > eps &&
          (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                 c_ans, c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

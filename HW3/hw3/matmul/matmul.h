#pragma once

enum method {
  NAIVE,
  SIMD,
};

void matmul(const float *A, const float *B, float *C, int M, int N, int K, int how);

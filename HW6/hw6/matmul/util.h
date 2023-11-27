#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void alloc_mat(float **m, int R, int S);

void rand_mat(float *m, int R, int S);

void zero_mat(float *m, int R, int S);

void print_vec(float *m, int N);

void print_mat(float *m, int M, int N);

void check_vectordot(float *A, float *B, float candidate, int N);

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K);

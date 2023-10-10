#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul.h"
#include "util.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvh] [-t num_threads] [-n num_iterations] M N K\n",
         prog_name);
  printf("Options:\n");
  printf("     -p : print vector. (default: off)\n");
  printf("     -v : validate matrix multiplication. (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -t : number of threads (default: 1)\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      M : number of rows of matrix A and C. (default: 8)\n");
  printf("      N : number of columns of matrix B and C. (default: 8)\n");
  printf(
      "      K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_data = false;
static bool validation = false;
static int M = 8;
static int N = 8;
static int K = 8;
static int num_threads = 1;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
    case 'p':
      print_data = true;
      break;
    case 'v':
      validation = true;
      break;
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 't':
      num_threads = atoi(optarg);
      break;
    case 'h':
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
    case 0:
      M = atoi(argv[i]);
      break;
    case 1:
      N = atoi(argv[i]);
      break;
    case 2:
      K = atoi(argv[i]);
      break;
    default:
      break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
  printf("  Number of threads: %d\n", num_threads);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print matrix: %s\n", print_data ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... ");
  fflush(stdout);
  float *A, *B, *C;

  // Initialize random seed
  timer_init();

  // Allocate vectors
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);

  // Set each element to a random value
  rand_mat(A, M, K);
  rand_mat(B, K, N);

  printf("done!\n");

  // WARMUP
  zero_mat(C, M, N);
  matmul(A, B, C, M, N, K, num_threads);

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i);
    fflush(stdout);
    zero_mat(C, M, N);

    timer_start(0);
    matmul(A, B, C, M, N, K, num_threads);

    double elapsed_time = timer_stop(0);
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  if (print_data) {
    printf("MATRIX A:\n");
    print_mat(A, M, K);
    printf("MATRIX B:\n");
    print_mat(B, K, N);
    printf("MATRIX C:\n");
    print_mat(C, M, N);
  }

  if (validation) {
    check_mat_mul(A, B, C, M, N, K);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n",
         2.0 * M * N * K / elapsed_time_avg / 1e9);

  return 0;
}

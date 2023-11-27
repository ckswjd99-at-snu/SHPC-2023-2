#include <getopt.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul.h"
#include "util.h"

static bool print_matrix = false;
static bool validation = false;
static int M = 8;
static int N = 8;
static int K = 8;
static int num_iterations = 1;
static int mpi_rank, mpi_world_size;
static char processor_name[MPI_MAX_PROCESSOR_NAME];

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvh] [-n num_iterations] M N K\n", prog_name);
  printf("Options:\n");
  printf("     -p : print matrix. (default: off)\n");
  printf("     -v : validate matrix multiplication. (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      M : number of rows of matrix A and C. (default: 8)\n");
  printf("      N : number of columns of matrix B and C. (default: 8)\n");
  printf(
      "      K : number of columns of matrix A and rows of B. (default: 8)\n");
}
static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
    case 'p':
      print_matrix = true;
      break;
    case 'v':
      validation = true;
      break;
    case 'n':
      num_iterations = atoi(optarg);
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

  if (mpi_rank == 0) {
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
  }
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("(%s) Hello world, rank %d out of %d\n", processor_name, mpi_rank,
         mpi_world_size);
  MPI_Barrier(MPI_COMM_WORLD);

  parse_opt(argc, argv);

  fflush(stdout);

  timer_init();

  float *A, *B, *C;
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);
  if (mpi_rank == 0) {
    printf("[rank %d] Initializing matrices...", mpi_rank);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    printf("Done!\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  matmul_initialize(M, N, K);
  MPI_Barrier(MPI_COMM_WORLD);

  // Few warmup iterations
  zero_mat(C, M, N);
  for (int i = 0; i < 3; i++) {
    matmul(A, B, C, M, N, K);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    if (mpi_rank == 0) {
      printf("[rank %d] Calculating...(iter=%d) ", mpi_rank, i);
      fflush(stdout);
      zero_mat(C, M, N);
    } else {
      zero_mat(A, M, K);
      zero_mat(B, K, N);
      zero_mat(C, M, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer_start(0);
    matmul(A, B, C, M, N, K);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = timer_stop(0);

    if (mpi_rank == 0) {
      printf("%f sec\n", elapsed_time);
      elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0) {
    if (print_matrix) {
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
    printf("[rank %d] Avg. time: %f sec\n", mpi_rank, elapsed_time_avg);
    printf("[rank %d] Avg. throughput: %f GFLOPS\n", mpi_rank,
           2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  MPI_Finalize();
  matmul_finalize();
  return 0;
}

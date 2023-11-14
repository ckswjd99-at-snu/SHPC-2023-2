#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "timer.h"
#include "util.h"
#include "vec_add.h"

// Usage: <program name> <vector size>

static int N = 209715200;

static void parse_opt(int argc, char **argv) {
  N = atoi(argv[1]);
  if (N % 4 != 0) {
    printf("N should be multiple of 4.\n");
    exit(0);
  }
}

int main(int argc, char **argv) {
  timer_init(9);
  parse_opt(argc, argv);

  float *A, *B, *C;
  printf("Initializing vectors...\n");
  alloc_vec(&A, N);
  alloc_vec(&B, N);
  alloc_vec(&C, N);
  rand_vec(A, N);
  rand_vec(B, N);
  printf("Initializing vectors done!\n");

  printf("Initializing...\n");
  vec_add_init(N);
  printf("Initializing done!\n");

  timer_reset(0);
  timer_reset(1);
  vec_add(A, B, C, N);

  double elapsed_time_normio = timer_read(0);
  printf("Elapsed time using normal I/O: %f sec\n", elapsed_time_normio);

  double elapsed_time_vecio = timer_read(1);
  printf("Elapsed time using vector I/O: %f sec\n", elapsed_time_vecio);

  printf("Finalizing...\n");
  vec_add_finalize();
  printf("Finalizing done!\n");

  free_vec(A);
  free_vec(B);
  free_vec(C);

  timer_finalize();

  return 0;
}

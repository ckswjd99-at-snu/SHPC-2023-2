#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <cmath>

#include "classifier.h"
#include "util.h"

static int mpi_rank, mpi_size;
static const char parameter_fname[30] = "data/params.bin";
static const char input_fname[30] = "data/input.bin";
static const char answer_fname[30] = "data/answer.bin";

void validation(float *output, float *answer, int N) {
  int err_cnt = 0;
  for (int i = 0; i < N; ++i) {
    if (isnan(output[i])) {
      err_cnt++;
      printf(" Validation   : Failed!\n");
      return;
    }
    if (fabs(output[i] - answer[i]) > 1e-4) {
      if (err_cnt == 0) {
        printf(" Validation   : Fail\n");
        printf(" ---------------------------------------\n");
      }
      if (err_cnt < 10) {
        printf(" [idx=%d] output : %f <-> answer : %f\n", i, output[i],
                answer[i]);
        err_cnt++;
      }
    }
  }
  if (err_cnt == 0) {
    printf(" Validation   : Pass\n");
  }
}

int main(int argc, char **argv) {
  int N = 1;
  bool run_validation = false;
  double st = 0.0, et = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  bool iam_root = (mpi_rank == 0);

  parse_option(argc, argv, &N, &run_validation);

  float *input = nullptr, *output = nullptr, *answer = nullptr, *parameter = nullptr;

  // Only the root (rank=0) process has input, output, answer, and parameter
  if (iam_root) {
    printf(" Loading inputs ... "); fflush(stdout);
    input = (float *) read_binary(input_fname);
    output = (float *) malloc(N * sizeof(float));
    answer = (float *) read_binary(answer_fname);
    parameter = (float*) read_binary(parameter_fname);
    printf("DONE\n"); fflush(stdout);
  }

  if (iam_root) printf(" Initializing classifier ... ");
  initialize_classifier(parameter, N);
  if (iam_root) printf("DONE\n");

  if (iam_root) printf(" Classifying %d articles ... ", N);
  st = get_time();
  MPI_Barrier(MPI_COMM_WORLD);
  classifier(input, output, N);
  MPI_Barrier(MPI_COMM_WORLD);
  et = get_time();
  if (iam_root) printf("DONE\n");

  if (iam_root) {
    printf(" =======================================\n");
    printf(" Elapsed time : %lf s\n", et - st);
    printf(" Throughput   : %lf input(s)/sec\n", (double) N / (et - st));
  }

  if (run_validation && iam_root) {
    validation(output, answer, N);
  }

  finalize_classifier();
  MPI_Finalize();

  return EXIT_SUCCESS;
}

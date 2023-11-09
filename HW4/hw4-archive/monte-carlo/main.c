#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "monte_carlo.h"
#include "util.h"

static int num_points = 8;
static int threads_per_process = 8;
static int mpi_rank, mpi_world_size;

static void print_help(const char *prog_name) {
  if (mpi_rank == 0) {
    printf("Usage: %s [-h] [-t threads_per_process] N\n", prog_name);
    printf("Options:\n");
    printf("  -h : print this page.\n");
    printf("  -t : number of threads per process (default: 8).\n");
    printf("   N : number of points (default: 8)\n");
  }
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 't':
        threads_per_process = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        MPI_Finalize();
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0:
        num_points = atoi(argv[i]);
        break;
      default:
        break;
    }
  }
  if (mpi_rank == 0) {
    printf("Options:\n");
    printf("  Number of points: %d\n", num_points);
    printf("  Number of threads per process: %d\n", threads_per_process);
    printf("\n");
  }
}


int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("(%s) Hello world, rank %d out of %d\n", processor_name, mpi_rank,
      mpi_world_size);
  MPI_Barrier(MPI_COMM_WORLD);

  parse_opt(argc, argv);

  double *xs, *ys;
  alloc_mat(&xs, num_points, 1);
  alloc_mat(&ys, num_points, 1);
  if (mpi_rank == 0) {
    srand(0xdeafbeaf);
    rand_double(xs, num_points);
    rand_double(ys, num_points);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double pi_estimate = monte_carlo(xs, ys, num_points, mpi_rank, mpi_world_size, threads_per_process);
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    printf("[rank %d] Estimated PI value: %.10f\n", mpi_rank, pi_estimate);
    printf("[rank %d] Relative error: %.4e\n", mpi_rank, (pi_estimate - M_PI) / M_PI);
    printf("[rank %d] Validation: ", mpi_rank);
    check_monte_carlo(xs, ys, num_points, pi_estimate);
  }

  MPI_Finalize();

  return 0;
}

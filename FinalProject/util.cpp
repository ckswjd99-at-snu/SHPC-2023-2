#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "util.h"
#include "classifier.h"

using namespace std;

void *read_binary(const char *filename) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    printf("[ERROR] Cannot open file \'%s\'\n", filename);
    fflush(stdout);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);

  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    printf("[ERROR] Cannot read file \'%s\'\n", filename);
    fflush(stdout);
    exit(-1);
  }
  fclose(f);

  return buf;
}

double get_time() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec + tv.tv_nsec * 1e-9;
  } else
    return 0;
}

void print_help() {
  printf(" Usage: ./run.sh [-n num_inputs] [-vwh]\n");
  printf(" Options:\n");
  printf("  -n : number of inputs     (default: 1, max input size: 8192)\n");
  printf("  -v : validate classifier. (default: off)\n");
  printf("  -h : print this page.\n");
  MPI_Finalize();
  exit(0);
}

void parse_option(int argc, char **argv, int *N, bool *V) {
  int opt;
  while ((opt = getopt(argc, argv, "n:vwh")) != -1) {
    switch (opt) {
      case 'n': *N = atoi(optarg); break;
      case 'v': *V = true; break;
      case 'h': print_help(); break; 
      default: break;
    }
  }

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    printf("\n Model : Classifier\n");
    printf(" =======================================\n");
    printf(" Number of inputs : %d\n", (*N));
    printf(" Validation : %s\n", (*V) ? "ON" : "OFF");
    printf(" ---------------------------------------\n");
    fflush(stdout);
  }
}

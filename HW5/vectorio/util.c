#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

void alloc_vec(float **m, int R) {
  *m = (float *) aligned_alloc(64, sizeof(float) * R);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int R) {
  for (int i = 0; i < R; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void zero_vec(float *m, int R) {
  memset(m, 0, sizeof(float) * R);
}

void free_vec(float *m) {
  free(m);
}

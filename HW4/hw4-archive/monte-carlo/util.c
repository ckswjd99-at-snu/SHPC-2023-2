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

void check_monte_carlo(double *xs, double *ys, int num_points, double parallel_result) {
  double eps = 1e-5;

  double count = 0.0;
#pragma omp parallel for reduction(+:count)
  for (int i = 0; i < num_points; i++) {
    double x = xs[i];
    double y = ys[i];

    if (x*x + y*y <= 1)
      count++;
  }

  double pi = (double) 4 * count / num_points;

  if (fabsf(parallel_result - pi) > eps &&
      (pi == 0 || fabsf((parallel_result - pi) / pi) > eps)) {

    printf("FAIL\n");
    printf("Correct value: %f, Your value: %f\n", pi, parallel_result);
  } else {
    printf("PASS\n");
  }
}

// returns a random number in [0,1]
void rand_double(double *numbers, int n) {
  union _double_union {
    unsigned long i;
    double d;
  } u;

  for (int i = 0; i < n; ++i ) {
    unsigned long r = rand();
    r = (r << 32);
    r |= rand();
    u.i = r;

    u.i &= ((1ul << 52)-1);
    u.i |= (1022ul << 52);

    numbers[i] =  (u.d - 0.5) * 2;
  }
}

void alloc_mat(double **m, int R, int S) {
  *m = (double *)aligned_alloc(32, sizeof(double) * R * S);
  if (*m == NULL) {
    printf("Failed to allocate memory for mat.\n");
    exit(0);
  }
}

#include "timer.h"

#include <stdlib.h>
#include <time.h>

struct timer_info {
  double total_time;
  struct timespec start_time;
  int is_running;
};

static struct timer_info *ti;

void timer_init(int n) {
  ti = (struct timer_info*)malloc(n * sizeof(struct timer_info));
  for (int i = 0; i < n; ++i) {
    ti[i].total_time = 0;
    ti[i].is_running = 0;
  }
}

void timer_finalize() {
  free(ti);
}

void timer_start(int idx) {
  if (ti[idx].is_running == 0) {
    clock_gettime(CLOCK_MONOTONIC, &ti[idx].start_time);
    ti[idx].is_running = 1;
  }
}

static double diff_with_current_time(struct timespec t) {
  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  long diff_nsec = (end_time.tv_sec - t.tv_sec) * 1000000000L + (end_time.tv_nsec - t.tv_nsec);
  return diff_nsec / 1e9;
}

void timer_stop(int idx) {
  if (ti[idx].is_running == 1) {
    ti[idx].total_time += diff_with_current_time(ti[idx].start_time);
    ti[idx].is_running = 0;
  }
}

double timer_read(int idx) {
  return ti[idx].total_time + (ti[idx].is_running ? diff_with_current_time(ti[idx].start_time) : 0);
}

void timer_reset(int idx) {
  ti[idx].total_time = 0;
  ti[idx].is_running = 0;
}

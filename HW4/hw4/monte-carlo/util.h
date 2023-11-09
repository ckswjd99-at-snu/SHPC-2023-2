#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void check_monte_carlo(double *xs, double *ys, int num_points, double parallel_result);

// returns random numbers in [0,1]
void rand_double(double *numbers, int n);
void alloc_mat(double **m, int R, int S);

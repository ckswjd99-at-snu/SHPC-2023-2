#include <omp.h>

#define N   64

void simple(int n, float *a, float *b) {
    int i, j;

    #pragma omp parallel for
    for (i = 1; i < n; i++)
        /* i is private by default */
        b[i] = (a[i] + a[i-1]) / 2.0;
}

void main() {
    omp_set_num_threads(4);

    float a[N] = {0.0, };
    float b[N] = {0.0, };

    simple(N, a, b);
}
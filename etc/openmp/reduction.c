#include <stdio.h>
#include <omp.h>

#define MAX 64

void main() {
    omp_set_num_threads(4);

    double sum = 0.0, A[MAX], avg;
    int i;

    #pragma omp parallel for
    for (i = 0; i < MAX; i++)
        A[i] = i;
    
    #pragma omp parallel for reduction (+:sum)
    for (i = 0; i < MAX; i++)
        sum += A[i];

    avg = sum / MAX;

    printf("avg: %f\n", avg);
}
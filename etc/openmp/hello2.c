#include <stdio.h>
#include <omp.h>

void main() {
    int num_threads, tid;
    omp_set_num_threads(4);
    num_threads = omp_get_num_threads();

    printf("Sequential section: # of threads = %d\n", num_threads);

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Parallel section: Hello world from thread %d\n", tid);

        if (tid == 0) {
            num_threads = omp_get_num_threads();
            printf("Parallel section: # of threads = %d\n", num_threads);
        }
    }
}
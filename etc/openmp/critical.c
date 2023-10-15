#include <stdio.h>
#include <omp.h>

void thread_counter() {
    int c;

    #pragma omp parallel
    {
        #pragma omp single
        c = 0;

        #pragma omp critical (increase)
        c++;
        
        #pragma omp barrier

        #pragma omp single
        printf("# of threads: %d\n", c);
    }
}

void main() {
    omp_set_num_threads(4);

    thread_counter();
}
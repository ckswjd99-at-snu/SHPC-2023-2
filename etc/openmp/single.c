#include <stdio.h>
#include <omp.h>

void work1() {}
void work2() {}

void single_example() {
    #pragma omp parallel
    {
        #pragma omp single
        printf("Beginning work1.\n");
        // Here's barrier: the others wait for the end of single construct
        
        work1();
        
        #pragma omp single
        printf("Finishing work1.\n");
        // Here's barrier: the others wait for the end of single construct
        
        #pragma omp single nowait
        printf("Finished work1 and beginning work2.\n");
        // Here's no barrier: the others don't wait for the end of single construct
        
        work2();
    }
}

void main() {
    omp_set_num_threads(4);

    single_example();
}
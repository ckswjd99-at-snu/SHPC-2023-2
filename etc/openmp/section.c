#include <stdio.h>
#include <omp.h>

void main() {
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            printf("Section 1: Hello world!\n");
            
            #pragma omp section
            printf("Section 2: Goodbye world!\n");
            
            #pragma omp section
            printf("Section 3: Hello hell!\n");
            
            #pragma omp section
            printf("Section 4: Goodbye heaven!\n");

        }
    }
}
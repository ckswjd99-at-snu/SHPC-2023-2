#include <stdio.h>
#include <omp.h>

#define N   64

float work1(int i) {
    return 2.0 * i;
}

void atomic_example(int n) {
    int i;
    float sum_x = 0, sum_y = 0;

    #pragma omp parallel shared(n, sum_x, sum_y)
    {
        #pragma omp for
        for (i=0; i<n; i++) {
            #pragma omp atomic update
            sum_x += work1(i);
            
            sum_y += work1(i);
        }
        
        #pragma omp barrier

        #pragma omp single
        {
            printf("Sum of x: %f\nSum of y: %f\n", sum_x, sum_y);
            if (sum_x != sum_y) printf("Atomicity is required!\n");
        }
        
    }
}

void main() {
    omp_set_num_threads(4);

    atomic_example(N);
}
#include <stdio.h>
#include <omp.h>

#define N   64

float average(float a, float b, float c) {
    return (a + b + c) / 3;
}

void master_example(float* x, float* xold, int n, float tol) {
    int c, i, toobig;
    float error, y;

    c = 0;

    #pragma omp parallel
    {
        do{
            #pragma omp for private(i)
            for( i = 1; i < n-1; ++i ){
                xold[i] = x[i];
            }

            #pragma omp single
            {
                toobig = 0;
            }
            
            #pragma omp for private(i,y,error) reduction(+:toobig)
            for( i = 1; i < n-1; ++i ){
                y = x[i];
                x[i] = average( xold[i-1], x[i], xold[i+1] );
                error = y - x[i];
                if( error > tol || error < -tol ) ++toobig;
            }

            #pragma omp master
            {
                ++c;
                printf( "iteration %d, toobig=%d\n", c, toobig );
            }
        } while( toobig > 0 );
    }
}

void main() {
    omp_set_num_threads(4);

    int i;
    float x[N];
    float xold[N];
    float tol = 5e-1;

    #pragma omp prallel for
    for (i = 0; i < N; i++) {
        x[i] = i;
    }

    master_example(x, xold, N, tol);

}
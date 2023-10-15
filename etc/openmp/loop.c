#include <math.h>
#include <omp.h>

#define N   64
#define M   64

void nowait_example(int n, int m, float *a, float *b, float *y, float *z)
{
    int i;
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (i=1; i<n; i++)
            b[i] = (a[i] + a[i-1]) / 2.0;

        #pragma omp for nowait
        for (i=0; i<m; i++)
            y[i] = sqrt(z[i]);
    }
}

void main() {
    int num_threads, tid;
    omp_set_num_threads(4);
    num_threads = omp_get_num_threads();

    float a[N] = {0.0,};
    float b[N] = {0.0,};
    float y[M] = {0.0,};
    float z[M] = {0.0,};

    nowait_example(N, M, a, b, y, z);
}
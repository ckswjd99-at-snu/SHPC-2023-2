#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int produce() { 
    int data = rand();
    printf("Produced: %d\n", data);

    return data;
}

void consume(int data) {
    printf("Consumed: %d\n", data);
}

void main () {
    omp_set_num_threads(2);

    for (int i = 0; i < 16; i++) {
        int flag = 0;
        int data = 0;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                data = produce();
                
                #pragma omp flush
                
                flag = 1;
                
                #pragma omp flush(flag)
            }

            #pragma omp section
            {
                while (!flag) {
                    #pragma omp flush(flag)
                }
                
                #pragma omp flush
                
                consume(data);
            }
        }
    }
}
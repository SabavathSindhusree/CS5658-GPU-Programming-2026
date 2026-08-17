
#include <iostream>
#include <cuda_runtime.h>

__device__ int counter = 0;

__global__ void barrierDemo()
{
    int tid = threadIdx.x;

    printf("Thread %d before __syncthreads()\n", tid);

    __syncthreads();

    printf("Thread %d after __syncthreads()\n", tid);

    atomicAdd(&counter, 1);

    __syncthreads();

    if (tid == 0)
    {
        printf("\nThreads reaching the atomic barrier: %d\n",
               counter);
    }
}

int main()
{
    barrierDemo<<<1, 8>>>();

    cudaDeviceSynchronize();

    return 0;
}

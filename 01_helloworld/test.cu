#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>

__global__ void hello_world()
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello World from thread %d\n", thread_id);
}

int main()
{
    hello_world<<<2,4>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
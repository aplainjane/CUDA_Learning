#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// CUDA 核函数
__global__ void hello_world() {
    int threadId = threadIdx.x;
    printf("GPU 线程 %d: Hello World!\n", threadId);
}

// 错误检查宏
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

int main(int argc, char **argv) {
    std::cout<<"CPU: Hello World!\n";
    
    // 启动核函数，使用 1 个块，10 个线程
    hello_world<<<1, 10>>>();
    
    // 同步以确保 GPU 输出完成
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // 重置设备
    CUDA_CHECK_ERROR(cudaDeviceReset());
    
    return 0;
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // 用于 dim3
#include <stdio.h>

__global__ void nesthelloworld(int iSize, int iDepth) {
    unsigned int tid = threadIdx.x;
    printf("depth : %d blockIdx: %d, threadIdx: %d\n", iDepth, blockIdx.x, threadIdx.x);
    if (iSize == 1) return;
    int nthread = (iSize >> 1);  // iSize / 2
    if (tid == 0 && nthread > 0) {
        nesthelloworld<<<1, nthread>>>(nthread, ++iDepth);  // 动态启动内核
        printf("-----------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char* argv[]) {
    int size = 64;
    int block_x = 2;
    dim3 blocks(block_x);  // 简化：1D 块
    dim3 grid((size - 1) / block_x + 1, 1);  // 网格

    nesthelloworld<<<grid, blocks>>>(size, 0);  // 初始启动

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();  // 等待所有动态内核完成
    cudaDeviceReset();
    return 0;
}
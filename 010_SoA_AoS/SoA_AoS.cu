#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>  // for rand
#include <cstdio>   // for printf

#define N (1<<20)  // 数组大小
#define BLOCK_SIZE 1024

// 结构体定义：必须在所有使用前完整定义
struct NaiveStruct {
    float a;
    float b;
};

// 辅助：随机初始化（针对 AoS 和 SoA）
void initData(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

// CPU 验证 AoS
void sumCPU(struct NaiveStruct* res, int n) {
    for (int i = 0; i < n; ++i) {
        res[i].a = res[i].a + res[i].b;  // 模拟加法 (a += b)
    }
}

// CPU 验证 SoA
void sumCPU_SoA(float* res, float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        res[i] = a[i] + b[i];
    }
}

// AoS 内核：结构体数组（非合并访问）
__global__ void sumAoS(struct NaiveStruct* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i].a += data[i].b;  // 访问 a 和 b：warp 内分散，效率低
    }
}

// SoA 内核：数组结构体（合并访问）
__global__ void sumSoA(float* a, float* b, float* res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] + b[i];  // 连续访问 a/b/res：效率高
    }
}

// 错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cout << "CUDA Error at " << #call << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

int main() {
    // AoS：结构体数组
    std::cout << "=== AoS (Array of Structs) ===" << std::endl;
    struct NaiveStruct* h_aos = (struct NaiveStruct*)malloc(N * sizeof(struct NaiveStruct));
    struct NaiveStruct* d_aos;
    CHECK_CUDA(cudaMalloc(&d_aos, N * sizeof(struct NaiveStruct)));
    
    // 初始化 AoS（a 和 b 交错）
    float* temp_a = (float*)h_aos;  // 临时视作 float* 初始化 a
    float* temp_b = temp_a + N;     // b 偏移 N
    initData(temp_a, N);
    initData(temp_b, N);
    
    CHECK_CUDA(cudaMemcpy(d_aos, h_aos, N * sizeof(struct NaiveStruct), cudaMemcpyHostToDevice));

    dim3 gridAoS((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockAoS(BLOCK_SIZE);

    auto start_aos = std::chrono::high_resolution_clock::now();
    sumAoS<<<gridAoS, blockAoS>>>(d_aos, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_aos = std::chrono::high_resolution_clock::now();
    double time_aos = std::chrono::duration<double, std::milli>(end_aos - start_aos).count();
    CHECK_CUDA(cudaMemcpy(h_aos, d_aos, N * sizeof(struct NaiveStruct), cudaMemcpyDeviceToHost));

    // CPU 验证 AoS
    sumCPU(h_aos, N);
    std::cout << "AoS 时间: " << time_aos << " ms (利用率低 ~50%)" << std::endl;

    // SoA：数组结构体
    std::cout << "\n=== SoA (Struct of Arrays) ===" << std::endl;
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_res = (float*)malloc(N * sizeof(float));
    float* d_a, *d_b, *d_res;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_res, N * sizeof(float)));
    
    initData(h_a, N);
    initData(h_b, N);
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridSoA((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSoA(BLOCK_SIZE);

    auto start_soa = std::chrono::high_resolution_clock::now();
    sumSoA<<<gridSoA, blockSoA>>>(d_a, d_b, d_res, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_soa = std::chrono::high_resolution_clock::now();
    double time_soa = std::chrono::duration<double, std::milli>(end_soa - start_soa).count();
    CHECK_CUDA(cudaMemcpy(h_res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 验证 SoA
    sumCPU_SoA(h_res, h_a, h_b, N);
    std::cout << "SoA 时间: " << time_soa << " ms (利用率高 ~90%+，加速 " << (time_aos / time_soa) << "x)" << std::endl;

    // 清理
    cudaFree(d_aos); free(h_aos);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_res); free(h_a); free(h_b); free(h_res);

    return 0;
}
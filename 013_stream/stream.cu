#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// 修正 sumKernel：块内共享归约 + 原子加到全局结果
__global__ void sumKernel(float* c, float* result, int n) {
    extern __shared__ float sdata[];  // 动态共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 加载到共享（边界处理）
    float sum = 0.0f;
    if (idx < n) sum = c[idx];
    sdata[tid] = sum;
    __syncthreads();

    // 块内归约（交错配对）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 线程0 原子加块和到全局结果
    if (tid == 0) atomicAdd(result, sdata[0]);
}

void callback(cudaStream_t stream, cudaError_t status, void* userData) {
    std::cout << "流1 完成！用户数据: " << *(int*)userData << std::endl;
}

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(1); } }

int main() {
    const int N = 1024 * 1024;  // 1M 元素
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);

    float *h_a = new float[N], *h_b = new float[N], *h_result = new float[1];
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    *h_result = 0.0f;

    float *d_a, *d_b, *d_c, *d_result;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    cudaEvent_t start_event, end_event1, end_event2;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&end_event1));
    CHECK_CUDA(cudaEventCreate(&end_event2));

    CHECK_CUDA(cudaEventRecord(start_event, stream1));

    // 流1: 异步 H2D + addKernel
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream1));
    addKernel<<<(N + 255) / 256, 256, 0, stream1>>>(d_a, d_b, d_c, N);

    CHECK_CUDA(cudaEventRecord(end_event1, stream1));

    int user_data = 42;
    CHECK_CUDA(cudaStreamAddCallback(stream1, callback, &user_data, 0));

    // 流2: 等待后 sumKernel（多块归约）
    CHECK_CUDA(cudaStreamWaitEvent(stream2, end_event1, 0));
    sumKernel<<<numBlocks, blockSize, blockSize * sizeof(float), stream2>>>(d_c, d_result, N);

    CHECK_CUDA(cudaEventRecord(end_event2, stream2));

    CHECK_CUDA(cudaEventSynchronize(end_event2));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, end_event1));
    std::cout << "流1 执行时间: " << ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "结果: " << *h_result << " (预期 ~3M)" << std::endl;

    delete[] h_a; delete[] h_b; delete[] h_result;
    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c)); CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(end_event1));
    CHECK_CUDA(cudaEventDestroy(end_event2));

    return 0;
}
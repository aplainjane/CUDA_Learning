/*
 * CUDA 示例：数组求和（Array Sum）
 * 
 * 这个示例演示了 CUDA 的基本使用：
 * 1. 在主机（CPU）上分配和初始化两个浮点数组 a 和 b。
 * 2. 将数组复制到设备（GPU）内存。
 * 3. 启动一个 CUDA 内核（sumArraysGPU），每个线程负责计算一个元素的和（a[i] + b[i]）。
 *    - 使用一个线程块（block size = 32），一个网格（grid size = 1），因为数组大小为 32。
 * 4. 将结果从设备复制回主机。
 * 5. 在主机上使用 CPU 版本的函数（sumArrays）计算相同的和。
 * 6. 比较 CPU 和 GPU 结果的正确性（使用 checkResult 函数）。
 * 
 * 这个示例是 CUDA 入门级（Freshman）教程的一部分，目的是验证 GPU 并行计算的基本正确性。
 * 注意：数组大小固定为 32，以匹配一个线程块的大小。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>  // 用于 memset
#include <cmath>    // 用于 fabs

// 定义 CHECK 宏：检查 CUDA API 调用是否成功，如果失败则打印错误并退出
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// initialData 函数：初始化数组 data 的前 n 个元素为 0.0f, 1.0f, 2.0f, ..., (n-1).0f
// 这个函数用于生成测试数据，便于验证结果（预期和为 a[i] + b[i] = 2*i）
void initialData(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(i);
    }
}

// sumArrays 函数：CPU 版本的数组求和，使用循环（每 4 个元素展开以模拟向量化）
void sumArrays(float *a, float *b, float *res, const int size) {
    for (int i = 0; i < size; i += 4) {
        res[i] = a[i] + b[i];
        if (i + 1 < size) res[i + 1] = a[i + 1] + b[i + 1];
        if (i + 2 < size) res[i + 2] = a[i + 2] + b[i + 2];
        if (i + 3 < size) res[i + 3] = a[i + 3] + b[i + 3];
    }
}

// checkResult 函数：比较两个数组 ref（CPU 结果）和 gpu（GPU 结果）的总和差异
// 如果相对误差小于 1e-5，则测试通过并打印 "PASSED"，否则 "FAILED"
void checkResult(float *ref, float *gpu, int n) {
    double sum_ref = 0.0, sum_gpu = 0.0;
    for (int i = 0; i < n; i++) {
        sum_ref += ref[i];
        sum_gpu += gpu[i];
    }
    double eps = 1e-5;
    double diff = fabs(sum_ref - sum_gpu) / sum_ref;
    std::cout << "Test " << (diff < eps ? "PASSED" : "FAILED") << std::endl;
}

// CUDA 内核函数：GPU 版本的数组求和
// 每个线程（threadIdx.x）计算 res[i] = a[i] + b[i]
__global__ void sumArraysGPU(float *a, float *b, float *res) {
    int i = threadIdx.x;  // 获取当前线程的索引（假设块大小等于数组大小）
    res[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int dev = 0;  // 指定设备 ID（默认 GPU 0）
    CHECK(cudaSetDevice(dev));

    int nElem = 32;  // 数组大小：32（适合一个线程块）
    std::cout << "Vector size: " << nElem << std::endl;
    int nByte = sizeof(float) * nElem;  // 字节数

    // 主机内存分配
    float *a_h = new float[nElem];
    float *b_h = new float[nElem];
    float *res_h = new float[nElem];  // CPU 结果
    float *res_from_gpu_h = new float[nElem];  // GPU 结果
    memset(res_h, 0, nByte);  // 清零
    memset(res_from_gpu_h, 0, nByte);

    // 设备内存分配
    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((void**)&a_d, nByte));
    CHECK(cudaMalloc((void**)&b_d, nByte));
    CHECK(cudaMalloc((void**)&res_d, nByte));

    // 初始化主机数据
    initialData(a_h, nElem);
    initialData(b_h, nElem);

    // 主机到设备复制
    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    // 执行配置：一个块（32 线程），一个网格（32/32=1）
    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    std::cout << "Execution configuration <<<" << grid.x << ", " << block.x << ">>>" << std::endl;

    // 设备到主机复制
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));

    // CPU 计算
    sumArrays(a_h, b_h, res_h, nElem);

    // 验证结果
    checkResult(res_h, res_from_gpu_h, nElem);

    // 清理内存
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
    delete[] a_h;
    delete[] b_h;
    delete[] res_h;
    delete[] res_from_gpu_h;

    return 0;
}
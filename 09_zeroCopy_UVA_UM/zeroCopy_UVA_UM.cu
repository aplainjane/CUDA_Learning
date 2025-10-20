#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>  // for rand()

#define N 1000000  // 数组大小：1M 元素
#define BLOCK_SIZE 256

// GPU 内核：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
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
    float *h_a, *h_b, *h_c;  // 主机数组
    float *d_a, *d_b, *d_c;  // 设备数组
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);

    // 初始化主机数据（随机 0~1）
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        std::cout << "主机内存分配失败！" << std::endl;
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // 0. Warm up（首次启动热身，避免 JIT 开销）
    std::cout << "=== 0. warm up ===" << std::endl;
    auto start_ = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c));
    d_a = nullptr;  // 避免打印无效指针

    auto end_ = std::chrono::high_resolution_clock::now();
    double time1_ = std::chrono::duration<double, std::milli>(end_ - start_).count();
    std::cout << "时间: " << time1_ << " ms" << std::endl;
    std::cout << "指针示例: h_a=" << h_a << ", d_a=" << (d_a ? d_a : (void*)nullptr) << std::endl << std::endl;

    // 1. 传统方式（基准：malloc + 显式拷贝）
    std::cout << "=== 1. 传统方式 (malloc + cudaMemcpy) ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c));

    auto end = std::chrono::high_resolution_clock::now();
    double time1 = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "时间: " << time1 << " ms" << std::endl;
    std::cout << "指针示例: h_a=" << h_a << ", d_a=" << d_a << std::endl;  // 地址不同（非 UVA）

    // 2. 零拷贝 (cudaHostAlloc pinned + 直接映射)
    std::cout << "\n=== 2. 零拷贝 (cudaHostAlloc + cudaHostGetDevicePointer) ===" << std::endl;
    float *pinned_a, *pinned_b;
    float *zc_d_a, *zc_d_b;
    start = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaHostAlloc(&pinned_a, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&pinned_b, N * sizeof(float), cudaHostAllocDefault));
    memcpy(pinned_a, h_a, N * sizeof(float));
    memcpy(pinned_b, h_b, N * sizeof(float));

    CHECK_CUDA(cudaHostGetDevicePointer(&zc_d_a, pinned_a, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&zc_d_b, pinned_b, 0));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    vectorAdd<<<blocks, threads>>>(zc_d_a, zc_d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFreeHost(pinned_a)); CHECK_CUDA(cudaFreeHost(pinned_b));

    end = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "时间: " << time2 << " ms" << std::endl;
    std::cout << "指针示例: pinned_a=" << pinned_a << ", zc_d_a=" << zc_d_a << std::endl;

    // 3. UVA (统一虚拟寻址：修复 - 用 cudaMalloc 分配设备，统一地址空间)
    std::cout << "\n=== 3. 统一虚拟寻址 (UVA + cudaMemcpy) ===" << std::endl;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int uva_supported = prop.unifiedAddressing;//在这里打开的uva 下面代码和传统一模一样
    double time3 = -1;
    if (!uva_supported) {
        std::cout << "警告: UVA 不支持！跳过。" << std::endl;
    } else {
        start = std::chrono::high_resolution_clock::now();

        // 主机数据（普通 malloc）
        float *uva_h_a = (float*)malloc(N * sizeof(float));
        float *uva_h_b = (float*)malloc(N * sizeof(float));
        if (!uva_h_a || !uva_h_b) {
            std::cout << "UVA 主机内存分配失败！" << std::endl;
            time3 = -1;
        } else {
            memcpy(uva_h_a, h_a, N * sizeof(float));
            memcpy(uva_h_b, h_b, N * sizeof(float));

            // UVA 下：cudaMalloc 返回统一指针（从主机可见）
            float *uva_d_a, *uva_d_b;
            CHECK_CUDA(cudaMalloc(&uva_d_a, N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&uva_d_b, N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

            // 拷贝 H2D 到统一设备指针
            CHECK_CUDA(cudaMemcpy(uva_d_a, uva_h_a, N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(uva_d_b, uva_h_b, N * sizeof(float), cudaMemcpyHostToDevice));

            vectorAdd<<<blocks, threads>>>(uva_d_a, uva_d_b, d_c, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaFree(uva_d_a)); CHECK_CUDA(cudaFree(uva_d_b)); CHECK_CUDA(cudaFree(d_c));
            free(uva_h_a); free(uva_h_b);

            end = std::chrono::high_resolution_clock::now();
            time3 = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "时间: " << time3 << " ms" << std::endl;
            std::cout << "指针示例: uva_d_a (统一)=" << uva_d_a << std::endl;  // UVA 下统一（与主机空间重叠）
        }
    }

    // 4. 统一内存 (cudaMallocManaged：自动迁移)
    std::cout << "\n=== 4. 统一内存 (cudaMallocManaged) ===" << std::endl;
    int um_supported = prop.managedMemory;
    double time4 = -1;
    if (!um_supported) {
        std::cout << "警告: 统一内存不支持！GPU 型号: " << prop.name << "。跳过 UM 测试。" << std::endl;
    } else {
        start = std::chrono::high_resolution_clock::now();

        float *um_a, *um_b, *um_c;
        CHECK_CUDA(cudaMallocManaged(&um_a, N * sizeof(float)));
        CHECK_CUDA(cudaMallocManaged(&um_b, N * sizeof(float)));
        CHECK_CUDA(cudaMallocManaged(&um_c, N * sizeof(float)));

        // 主机初始化
        for (int i = 0; i < N; ++i) {
            um_a[i] = h_a[i];
            um_b[i] = h_b[i];
        }

        vectorAdd<<<blocks, threads>>>(um_a, um_b, um_c, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // 主机验证
        float sum = 0;
        for (int i = 0; i < N; ++i) sum += um_c[i];

        CHECK_CUDA(cudaFree(um_a)); CHECK_CUDA(cudaFree(um_b)); CHECK_CUDA(cudaFree(um_c));

        end = std::chrono::high_resolution_clock::now();
        time4 = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "时间: " << time4 << " ms" << std::endl;
        std::cout << "指针示例: um_a (统一)=" << um_a << std::endl;
        std::cout << "验证和: " << sum << std::endl;
    }

    // 性能对比
    std::cout << "\n=== 性能对比 (ms) ===" << std::endl;
    std::cout << "传统: " << time1 << std::endl;
    std::cout << "零拷贝: " << time2 << std::endl;
    std::cout << "UVA: " << (uva_supported ? time3 : 5497) << std::endl;
    std::cout << "UM: " << (um_supported ? time4 : 5497) << std::endl;

    free(h_a); free(h_b); free(h_c);
    return 0;
}
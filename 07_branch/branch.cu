#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib> // 用于rand()函数

// CPU递归归约函数（交错配对风格，处理奇数大小）
int recursiveReduce(int* data, int size) {
    if (size == 1) return data[0];  // 终止条件：只有一个元素时返回它
    int stride = size / 2;  // 计算步长
    if (size % 2 == 1) {  // 如果大小为奇数，处理最后一个无人配对的元素
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];  // 对配对元素求和
        }
        data[0] += data[size - 1];  // 将最后一个元素加到第一个
    } else {  // 偶数大小，直接配对
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];
        }
    }
    return recursiveReduce(data, stride);  // 递归调用，缩小问题规模
}

// GPU内核1：reduceNeighbored（原始相邻配对版本，存在分支分化问题）
__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;  // 获取线程ID
    if (tid >= n) return;  // 边界检查
    int* idata = g_idata + blockIdx.x * blockDim.x;  // 指向当前块的局部数据指针

    // 原地归约在全局内存中
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {  // 分支条件：导致线程束分化
            idata[tid] += idata[tid + stride];  // 对相邻配对元素求和
        }
        __syncthreads();  // 块内同步
    }

    if (tid == 0) {  // 线程0写入块结果
        g_odata[blockIdx.x] = idata[0];
    }
}

// GPU内核2：reduceNeighboredLess（优化相邻配对，避免分支分化）
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;  // 获取线程ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 获取全局索引
    int* idata = g_idata + blockIdx.x * blockDim.x;  // 指向当前块的局部数据指针
    if (idx >= n) return;  // 边界检查

    for (int stride = 1; stride < blockDim.x; stride *= 2) {  // 从小步长开始迭代
        int index = 2 * stride * tid;  // 计算调整后的索引，确保线程利用率高
        if (index < blockDim.x) {  // 检查是否在块内
            idata[index] += idata[index + stride];  // 对配对元素求和
        }
        __syncthreads();  // 块内同步，确保所有线程完成当前迭代
    }

    if (tid == 0) {  // 只有线程0将块结果写入输出
        g_odata[blockIdx.x] = idata[0];
    }
}

// GPU内核3：reduceInterleaved（交错配对）
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;  // 获取线程ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 获取全局索引
    int* idata = g_idata + blockIdx.x * blockDim.x;  // 指向当前块的局部数据指针
    if (idx >= n) return;  // 边界检查

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {  // 从大步长开始迭代，向小步长缩小
        if (tid < stride) {  // 只有前stride个线程参与当前迭代
            idata[tid] += idata[tid + stride];  // 对交错配对元素求和
        }
        __syncthreads();  // 块内同步
    }

    if (tid == 0) {  // 线程0写入块结果
        g_odata[blockIdx.x] = idata[0];
    }
}

// GPU内核4：reduceUnrolled（基于交错配对的循环展开优化版本）
__global__ void reduceUnrolled(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;  // 获取线程ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 获取全局索引
    int* idata = g_idata + blockIdx.x * blockDim.x;  // 指向当前块的局部数据指针
    if (idx >= n) return;  // 边界检查

    // 先用循环处理大部分迭代，直到stride=16
    for (int stride = blockDim.x / 2; stride > 16; stride >>= 1) {
        if (tid < stride) {  // 只有前stride个线程参与当前迭代
            idata[tid] += idata[tid + stride];  // 对交错配对元素求和
        }
        __syncthreads();  // 块内同步
    }

    // 手动展开最后几轮（stride=16,8,4,2,1），减少循环开销和分支
    if (tid < 16) {
        // stride=16
        idata[tid] += idata[tid + 16];
        __syncthreads();

        // stride=8
        idata[tid] += idata[tid + 8];
        __syncthreads();

        // stride=4
        idata[tid] += idata[tid + 4];
        __syncthreads();

        // stride=2
        idata[tid] += idata[tid + 2];
        __syncthreads();

        // stride=1
        idata[tid] += idata[tid + 1];
    }

    if (tid == 0) {  // 线程0写入块结果
        g_odata[blockIdx.x] = idata[0];
    }
}

// GPU内核5：reduceShared（使用共享内存的交错配对版本）
__global__ void reduceShared(int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];  // 声明动态共享内存
    unsigned int tid = threadIdx.x;  // 获取线程ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 获取全局索引
    unsigned int bid = blockIdx.x;  // 获取块ID

    // 将全局内存数据加载到共享内存（处理边界情况）
    if (idx < n) {
        sdata[tid] = g_idata[idx];
    } else {
        sdata[tid] = 0;  // 填充边界外的值为0
    }
    __syncthreads();  // 同步，确保所有线程加载完成

    // 在共享内存中进行交错配对归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];  // 对配对元素求和
        }
        __syncthreads();  // 块内同步
    }

    // 线程0将块结果写入全局输出
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

// 工具函数：对GPU块的部分结果求和
int sumPartialResults(int* partials, int numBlocks) {
    int sum = 0;  // 初始化总和
    for (int i = 0; i < numBlocks; ++i) {
        sum += partials[i];  // 累加每个块的结果
    }
    return sum;
}

int main() {
    const int size = 1 << 26;  // 数组大小：16M元素（2的幂）
    const int blockSize = 512;  // 每个块的线程数
    const int numBlocks = (size + blockSize - 1) / blockSize;  // 计算块数(向上取整)

    // 主机内存分配
    int* h_idata = new int[size];  // 输入数据
    int* h_odata = new int[numBlocks];  // 输出部分结果
    int* tmp = new int[size];  // CPU使用的临时数组（用于就地修改）

    // 用随机值（1到10）初始化数据，便于验证求和
    for (int i = 0; i < size; ++i) {
        h_idata[i] = rand() % 10 + 1;
        tmp[i] = h_idata[i];  // 复制到临时数组
    }

    // CPU归约计算
    auto cpu_start = std::chrono::high_resolution_clock::now();  // 开始计时
    int cpu_sum = recursiveReduce(tmp, size);  // 执行CPU递归归约
    auto cpu_end = std::chrono::high_resolution_clock::now();  // 结束计时
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();  // 计算毫秒时间

    // 设备内存分配
    int* d_idata;
    int* d_odata;
    cudaMalloc(&d_idata, size * sizeof(int));  // 输入设备内存
    cudaMalloc(&d_odata, numBlocks * sizeof(int));  // 输出设备内存

    dim3 block(blockSize);  // 块维度
    dim3 grid(numBlocks);  // 网格维度

    // GPU内核1：reduceNeighbored（分支分化版本）
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);  // 主机到设备复制（修正了可能的打字错误）
    auto gpu1_start = std::chrono::high_resolution_clock::now();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);  // 启动内核
    cudaDeviceSynchronize();  // 等待内核完成
    auto gpu1_end = std::chrono::high_resolution_clock::now();
    double gpu1_time = std::chrono::duration<double, std::milli>(gpu1_end - gpu1_start).count();
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);  // 设备到主机复制
    int gpu1_sum = sumPartialResults(h_odata, numBlocks);  // 求部分结果总和

    // GPU内核2：reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);  // 重新复制输入
    auto gpu2_start = std::chrono::high_resolution_clock::now();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);  // 启动内核
    cudaDeviceSynchronize();
    auto gpu2_end = std::chrono::high_resolution_clock::now();
    double gpu2_time = std::chrono::duration<double, std::milli>(gpu2_end - gpu2_start).count();
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu2_sum = sumPartialResults(h_odata, numBlocks);

    // GPU内核3：reduceInterleaved
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);  // 重新复制输入
    auto gpu3_start = std::chrono::high_resolution_clock::now();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);  // 启动内核
    cudaDeviceSynchronize();
    auto gpu3_end = std::chrono::high_resolution_clock::now();
    double gpu3_time = std::chrono::duration<double, std::milli>(gpu3_end - gpu3_start).count();
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu3_sum = sumPartialResults(h_odata, numBlocks);

    // GPU内核4：reduceUnrolled（循环展开版本）
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);  // 重新复制输入
    auto gpu4_start = std::chrono::high_resolution_clock::now();
    reduceUnrolled<<<grid, block>>>(d_idata, d_odata, size);  // 启动内核
    cudaDeviceSynchronize();
    auto gpu4_end = std::chrono::high_resolution_clock::now();
    double gpu4_time = std::chrono::duration<double, std::milli>(gpu4_end - gpu4_start).count();
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu4_sum = sumPartialResults(h_odata, numBlocks);

   // GPU内核5：reduceShared（共享内存版本）
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);  // 重新复制输入
    auto gpu5_start = std::chrono::high_resolution_clock::now();
    reduceShared<<<grid, block, blockSize * sizeof(int)>>>(d_idata, d_odata, size);  // 启动内核，指定共享内存大小
    cudaDeviceSynchronize();
    auto gpu5_end = std::chrono::high_resolution_clock::now();
    double gpu5_time = std::chrono::duration<double, std::milli>(gpu5_end - gpu5_start).count();
    cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu5_sum = sumPartialResults(h_odata, numBlocks);

    // 输出结果
    std::cout << "数组大小: " << size << std::endl;
    std::cout << "CPU 求和: " << cpu_sum << ", 时间: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU reduceNeighbored（分支分化） 求和: " << gpu1_sum << ", 时间: " << gpu1_time << " ms" << std::endl;
    std::cout << "GPU reduceNeighboredLess 求和: " << gpu2_sum << ", 时间: " << gpu2_time << " ms" << std::endl;
    std::cout << "GPU reduceInterleaved 求和: " << gpu3_sum << ", 时间: " << gpu3_time << " ms" << std::endl;
    std::cout << "GPU reduceUnrolled（循环展开） 求和: " << gpu4_sum << ", 时间: " << gpu4_time << " ms" << std::endl;
    std::cout << "GPU reduceShared（共享内存） 求和: " << gpu5_sum << ", 时间: " << gpu5_time << " ms" << std::endl;
    
    // 清理内存
    delete[] h_idata;
    delete[] h_odata;
    delete[] tmp;
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
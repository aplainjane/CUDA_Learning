#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

// 预热内核，用于消除开销
__global__ void warmup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算线程ID
    float a = 0.0f;
    float b = 0.0f;

    if ((tid / warpSize) % 2 == 0)  // 根据warp大小的奇偶性设置a
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    // printf("%d %d %f \n", tid, warpSize, a + b);  // 注释掉的打印语句
    c[tid] = a + b;  // 计算并存储结果
}

// 数学内核1：基于线程ID的奇偶性进行分支
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算线程ID

    float a = 0.0f;
    float b = 0.0f;
    if (tid % 2 == 0)  // 根据线程ID的奇偶性设置a或b
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;  // 计算并存储结果
}

// 数学内核2：基于warp的奇偶性进行分支
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算线程ID
    float a = 0.0f;
    float b = 0.0f;
    if ((tid / warpSize) % 2 == 0)  // 根据warp ID的奇偶性设置a或b
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;  // 计算并存储结果
}

// 数学内核3：使用bool变量进行分支预测
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 计算线程ID
    float a = 0.0f;
    float b = 0.0f;
    bool ipred = (tid % 2 == 0);  // 使用bool变量存储预测条件
    if (ipred)  // 根据bool变量设置a或b
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;  // 计算并存储结果
}

// CPU时间函数，使用chrono库获取当前时间
double cpuSecond() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

int main(int argc, char* argv[])
{
    int dev = 0;  // 设备ID
    cudaDeviceProp deviceProp;  // 设备属性
    cudaGetDeviceProperties(&deviceProp, dev);  // 获取设备属性
    std::cout << argv[0] << " using Device " << dev << ": " << deviceProp.name << std::endl;  // 打印设备信息

    // 设置数据大小
    int size = 64;  // 默认数据大小
    int blocksize = 64;  // 默认块大小
    if (argc > 1) blocksize = std::atoi(argv[1]);  // 从命令行参数读取块大小
    if (argc > 2) size = std::atoi(argv[2]);  // 从命令行参数读取数据大小
    std::cout << "Data size " << size << std::endl;  // 打印数据大小

    // 设置执行配置
    dim3 block(blocksize, 1);  // 块维度
    dim3 grid((size - 1) / block.x + 1, 1);  // 网格维度
    std::cout << "Execution Configure (block " << block.x << " grid " << grid.x << ")" << std::endl;  // 打印执行配置

    // 分配GPU内存
    float *C_dev;  // 设备端数组指针
    size_t nBytes = size * sizeof(float);  // 字节数
    std::vector<float> C_host(size);  // 主机端数组
    cudaMalloc((void**)&C_dev, nBytes);  // 分配设备内存

    // 运行预热内核以消除开销
    double iStart, iElaps;  // 时间变量
    cudaDeviceSynchronize();  // 同步设备
    iStart = cpuSecond();  // 记录开始时间
    warmup<<<grid, block>>>(C_dev);  // 启动预热内核
    cudaDeviceSynchronize();  // 同步设备
    iElaps = cpuSecond() - iStart;  // 计算耗时

    std::cout << std::fixed << std::setprecision(6);  // 设置输出精度
    std::cout << "warmup     <<<" << std::setw(4) << grid.x << "," << std::setw(4) << block.x << ">>> elapsed " << iElaps << " sec" << std::endl;  // 打印预热时间

    // 运行内核1
    iStart = cpuSecond();  // 记录开始时间
    mathKernel1<<<grid, block>>>(C_dev);  // 启动内核1
    cudaDeviceSynchronize();  // 同步设备
    iElaps = cpuSecond() - iStart;  // 计算耗时
    std::cout << "mathKernel1<<<" << std::setw(4) << grid.x << "," << std::setw(4) << block.x << ">>> elapsed " << iElaps << " sec" << std::endl;  // 打印内核1时间
    cudaMemcpy(C_host.data(), C_dev, nBytes, cudaMemcpyDeviceToHost);  // 拷贝结果到主机
    // for (int i = 0; i < size; ++i) {  // 注释掉的打印循环
    //     std::cout << C_host[i] << " ";
    // }
    // std::cout << std::endl;

    // 运行内核2
    iStart = cpuSecond();  // 记录开始时间
    mathKernel2<<<grid, block>>>(C_dev);  // 启动内核2
    cudaDeviceSynchronize();  // 同步设备
    iElaps = cpuSecond() - iStart;  // 计算耗时
    std::cout << "mathKernel2<<<" << std::setw(4) << grid.x << "," << std::setw(4) << block.x << ">>> elapsed " << iElaps << " sec" << std::endl;  // 打印内核2时间

    // 运行内核3
    iStart = cpuSecond();  // 记录开始时间
    mathKernel3<<<grid, block>>>(C_dev);  // 启动内核3
    cudaDeviceSynchronize();  // 同步设备
    iElaps = cpuSecond() - iStart;  // 计算耗时
    std::cout << "mathKernel3<<<" << std::setw(4) << grid.x << "," << std::setw(4) << block.x << ">>> elapsed " << iElaps << " sec" << std::endl;  // 打印内核3时间

    cudaFree(C_dev);  // 释放设备内存
    cudaDeviceReset();  // 重置设备
    return 0;  // 程序正常结束
}
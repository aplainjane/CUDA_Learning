/**
 * CUDA 设备查询示例
 * 
 * 此程序查询并打印第一个 CUDA 兼容设备的属性。
 * 基于 NVIDIA 的 deviceQuery 示例，使用 C++ 编写，并更新为最新的 CUDA 12.x API。
 * 
 * 使用的关键 API（来自 CUDA 运行时 API，CUDA 12.x 中的最新版本）：
 * - cudaGetDeviceCount()：检索 CUDA 兼容设备的数量。
 * - cudaSetDevice()：设置当前 CUDA 设备（此处为设备 0）。
 * - cudaGetDeviceProperties()：用设备属性填充 cudaDeviceProp 结构体。
 *   - cudaDeviceProp：存储设备属性的结构体（例如，主要/次要计算能力、
 *     内存大小、线程限制）。详见 NVIDIA 文档：https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
 * - cudaDriverGetVersion() / cudaRuntimeGetVersion()：获取驱动程序和运行时版本。
 * 
 * 编译（需要 CUDA Toolkit >= 12.x）：
 *   nvcc -std=c++17 -o deviceQuery deviceQuery.cu
 * 
 * 使用方法：
 *   ./deviceQuery [可选参数如果扩展]
 * 
 * 输出：将设备信息打印到标准输出。优雅处理错误。
 * 
 * 注意：
 * - 专注于原始 C 代码中的核心属性；可扩展到所有 cudaDeviceProp 成员。
 * - 错误处理：检查 cudaError_t 返回值。
 * - 精度：使用 <iomanip> 进行格式化的浮点数输出。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>  // 用于 EXIT_SUCCESS/FAILURE

int main(int argc, char** argv) {
    std::cout << argv[0] << " Starting ..." << std::endl;

    // 步骤 1: 查询 CUDA 设备数量
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);  // 最新 API：在 12.x 中无变化
    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << std::endl
                  << " -> " << cudaGetErrorString(error_id) << std::endl;
        std::cout << "Result = FAIL" << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
        return EXIT_SUCCESS;  // 如果没有设备，不是失败
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    // 步骤 2: 选择第一个设备（dev=0）并查询属性
    int dev = 0;
    cudaSetDevice(dev);  // 设置活动设备；如果设备无效则出错
    cudaDeviceProp deviceProp;  // 属性结构体（在 CUDA 12.x 中扩展，例如 memoryPoolsSupported）
    error_id = cudaGetDeviceProperties(&deviceProp, dev);
    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties returned " << static_cast<int>(error_id) << std::endl
                  << " -> " << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }

    // 步骤 3: 打印基本设备信息
    std::cout << "Device " << dev << ": \"" << deviceProp.name << "\"" << std::endl;  // ASCII 设备名称（最大 256 字符）

    // 驱动程序和运行时版本（最新 API：支持高达 CUDA 12.8+）
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "  CUDA Driver Version / Runtime Version         "
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << "  /  "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;

    // 计算能力（major/minor；例如，Ampere 的 8.6）
    std::cout << "  CUDA Capability Major/Minor version number:   "
              << deviceProp.major << "." << deviceProp.minor << std::endl;

    // 全局内存（totalGlobalMem：size_t，以字节为单位）
    std::cout << "  Total amount of global memory:                "
              << std::fixed << std::setprecision(2)
              << static_cast<float>(deviceProp.totalGlobalMem) / std::pow(1024.0, 3) << " MBytes ("
              << deviceProp.totalGlobalMem << " bytes)" << std::endl;

    // 时钟速率（clockRate：int，以 kHz 为单位）
    std::cout << "  GPU Clock rate:                               "
              << std::fixed << std::setprecision(0) << deviceProp.clockRate * 1e-3f << " MHz ("
              << std::setprecision(2) << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;

    // 内存总线宽度（memoryBusWidth：int，位）
    std::cout << "  Memory Bus width:                             " << deviceProp.memoryBusWidth << "-bits" << std::endl;

    // L2 缓存（l2CacheSize：int，字节；如果不支持则为 0）
    if (deviceProp.l2CacheSize) {
        std::cout << "  L2 Cache Size:                                " << deviceProp.l2CacheSize << " bytes" << std::endl;
    }

    // 纹理维度（各种 maxTexture* 成员：int 数组）
    std::cout << "  Max Texture Dimension Size (x,y,z)            1D=(" << deviceProp.maxTexture1D
              << "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1]
              << "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1]
              << "," << deviceProp.maxTexture3D[2] << ")" << std::endl;
    std::cout << "  Max Layered Texture Size (dim) x layers       1D=(" << deviceProp.maxTexture1DLayered[0]
              << ") x " << deviceProp.maxTexture1DLayered[1]
              << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << "," << deviceProp.maxTexture2DLayered[1]
              << ") x " << deviceProp.maxTexture2DLayered[2] << std::endl;

    // 常量内存和共享内存（totalConstMem/sharedMemPerBlock：size_t，字节）
    std::cout << "  Total amount of constant memory               " << deviceProp.totalConstMem << " bytes" << std::endl;
    std::cout << "  Total amount of shared memory per block:      " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;

    // 寄存器（regsPerBlock：int）
    std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;

    // 线程束大小（warpSize：int，通常为 32）
    std::cout << "  Warp size:                                    " << deviceProp.warpSize << std::endl;

    // 线程限制（maxThreadsPerMultiProcessor/maxThreadsPerBlock：int）
    std::cout << "  Maximum number of threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Maximum number of threads per block:          " << deviceProp.maxThreadsPerBlock << std::endl;

    // 块维度（maxThreadsDim[3]：int 数组，x/y/z）
    std::cout << "  Maximum size of each dimension of a block:    " << deviceProp.maxThreadsDim[0] << " x "
              << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;

    // 网格维度（maxGridSize[3]：size_t 数组，x/y/z；在 12.x 中为 size_t）
    std::cout << "  Maximum size of each dimension of a grid:     " << deviceProp.maxGridSize[0] << " x "
              << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;

    // 内存间距（memPitch：size_t，最大字节）
    std::cout << "  Maximum memory pitch                          " << deviceProp.memPitch << " bytes" << std::endl;

    std::cout << "Result = PASS" << std::endl;  // 成功指示
    
    // 打印设备属性详细信息
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Number of multiprocessors:                      " << deviceProp.multiProcessorCount << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Total amount of constant memory:                " << (deviceProp.totalConstMem / 1024.0) << " KB" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Total amount of shared memory per block:        " << (deviceProp.sharedMemPerBlock / 1024.0) << " KB" << std::endl;
    std::cout << "Total number of registers available per block:  " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size                                       " << deviceProp.warpSize << std::endl;
    std::cout << "Maximum number of threads per block:            " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum number of warps per multiprocessor:     " << (deviceProp.maxThreadsPerMultiProcessor / 32) << std::endl;

    return EXIT_SUCCESS;
}
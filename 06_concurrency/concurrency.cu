#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdlib>

// CUDA错误检查宏
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// 初始化数据函数（假设填充随机数据或固定值，根据需要调整）
void initialData(std::vector<float>& data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;  // 随机值 [0,1]
    }
}

// CPU时间函数，使用chrono库获取当前时间
double cpuSecond() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

// 矩阵求和内核：元素-wise相加
__global__ void sumMatrix(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny;
    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char* argv[]) {
    // printf("strating...\n");  // 注释掉的打印语句
    // initDevice(0);  // 假设的初始化设备函数，注释掉

    int nx = 1 << 13;  // 8192
    int ny = 1 << 13;  // 8192
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);

    // 主机端内存分配，使用vector
    std::vector<float> A_host(nxy);
    std::vector<float> B_host(nxy);
    std::vector<float> C_host(nxy);  // 未使用，但保留
    std::vector<float> C_from_gpu(nxy);
    initialData(A_host, nxy);
    initialData(B_host, nxy);

    // 设备端内存分配
    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;
    CHECK(cudaMalloc((void**)&A_dev, nBytes));
    CHECK(cudaMalloc((void**)&B_dev, nBytes));
    CHECK(cudaMalloc((void**)&C_dev, nBytes));

    CHECK(cudaMemcpy(A_dev, A_host.data(), nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host.data(), nBytes, cudaMemcpyHostToDevice));

    // 块和网格维度（注意原代码中argc>2的条件可能有误，按原样保留）
    int dimx = (argc > 1) ? std::atoi(argv[1]) : 32;
    int dimy = (argc > 2) ? std::atoi(argv[2]) : 32;

    double iStart, iElaps;

    // 2D块和2D网格
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    iStart = cpuSecond();
    sumMatrix<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "GPU Execution configuration<<<(" << grid.x << "," << grid.y << "),(" << block.x << "," << block.y << ")|" << iElaps << " sec" << std::endl;

    CHECK(cudaMemcpy(C_from_gpu.data(), C_dev, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    // 无需手动free vector

    cudaDeviceReset();
    return 0;
}
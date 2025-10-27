#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>  // rand

#define NX 1024
#define NY 1024
#define NXY (NX * NY)
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_DIM 16  // For shared memory tile size (square)

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

void initMatrix(float* mat, int nxy) {
    for (int i = 0; i < nxy; ++i) mat[i] = rand() / (float)RAND_MAX;
}

void transposeCPU(float* A, float* B, int nx, int ny) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            B[i * ny + j] = A[j * nx + i];
        }
    }
}

// 上限：行读行写复制（合并满分）
__global__ void copyRow(float* A, float* B, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) {
        int idx = ix + iy * nx;  // 行索引（连续读写）
        B[idx] = A[idx];
    }
}

// 下限：列读列写复制（交叉最乱）
__global__ void copyCol(float* A, float* B, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) {
        int idx = ix * ny + iy;  // 列索引（交叉读写）
        B[idx] = A[idx];
    }
}

// Naive 转置（行读，列写） - 修正索引bug
__global__ void transposeNaive(float* A, float* B, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) {
        int a_idx = iy * nx + ix;  // A[row * nx + col]
        int b_idx = ix * ny + iy;  // B[col * ny + row]
        B[b_idx] = A[a_idx];
    }
}

// 展开转置（行读，列写 + 4x unroll） - 修正索引bug，并调整为正确偏移
__global__ void transposeUnrolled(float* A, float* B, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny) {
        int a_idx = iy * nx + ix;
        int b_idx = ix * ny + iy;
        B[b_idx] = A[a_idx];
        if (ix + blockDim.x < nx) {
            int a_idx_k = iy * nx + (ix + blockDim.x);
            int b_idx_k = (ix + blockDim.x) * ny + iy;
            B[b_idx_k] = A[a_idx_k];
        }
        if (ix + 2 * blockDim.x < nx) {
            int a_idx_k = iy * nx + (ix + 2 * blockDim.x);
            int b_idx_k = (ix + 2 * blockDim.x) * ny + iy;
            B[b_idx_k] = A[a_idx_k];
        }
        if (ix + 3 * blockDim.x < nx) {
            int a_idx_k = iy * nx + (ix + 3 * blockDim.x);
            int b_idx_k = (ix + 3 * blockDim.x) * ny + iy;
            B[b_idx_k] = A[a_idx_k];
        }
    }
}

// 共享内存优化转置（tiled transpose，避免银行冲突基本版）
__global__ void transposeShared(float* A, float* B, int nx, int ny) {
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM];  // 共享内存瓦片（可加padding优化银行冲突，如[BLOCK_DIM][BLOCK_DIM + 4]）

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 加载阶段：每个线程加载A的元素到共享内存的转置位置
    int row = by * BLOCK_Y + ty;  // row in A
    int col = bx * BLOCK_X + tx;  // col in A
    if (row < ny && col < nx) {
        tile[tx][ty] = A[row * nx + col];  // tile[col_local][row_local] = A[row_local][col_local]
    }
    __syncthreads();  // 同步，确保加载完成

    // 写出阶段：从共享内存写到B的转置位置
    int out_row = bx * BLOCK_X + tx;  // row in B = col in A
    int out_col = by * BLOCK_Y + ty;  // col in B = row in A
    if (out_row < nx && out_col < ny) {
        B[out_row * ny + out_col] = tile[tx][ty];
    }
}

int main() {
    int nBytes = NXY * sizeof(float);
    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *h_B_cpu = (float*)malloc(nBytes);

    initMatrix(h_A, NXY);

    float *d_A, *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, nBytes));
    CHECK_CUDA(cudaMalloc(&d_B, nBytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((NX + BLOCK_X - 1) / BLOCK_X, (NY + BLOCK_Y - 1) / BLOCK_Y);
    dim3 grid_unroll((NX + 4 * BLOCK_X - 1) / (4 * BLOCK_X), (NY + BLOCK_Y - 1) / BLOCK_Y);

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(h_A, h_B_cpu, NX, NY);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU 时间: " << time_cpu << " ms" << std::endl;

    // 上限：copyRow
    CHECK_CUDA(cudaMemset(d_B, 0, nBytes));
    auto start_upper = std::chrono::high_resolution_clock::now();
    copyRow<<<grid, block>>>(d_A, d_B, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_upper = std::chrono::high_resolution_clock::now();
    double time_upper = std::chrono::duration<double, std::milli>(end_upper - start_upper).count();
    double bandwidth_upper = (2.0 * NXY * sizeof(float) * 1e-9) / (time_upper * 1e-3);
    std::cout << "上限 (行复制) 时间: " << time_upper << " ms, 带宽: " << bandwidth_upper << " GB/s" << std::endl;

    // 下限：copyCol
    CHECK_CUDA(cudaMemset(d_B, 0, nBytes));
    auto start_lower = std::chrono::high_resolution_clock::now();
    copyCol<<<grid, block>>>(d_A, d_B, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_lower = std::chrono::high_resolution_clock::now();
    double time_lower = std::chrono::duration<double, std::milli>(end_lower - start_lower).count();
    double bandwidth_lower = (2.0 * NXY * sizeof(float) * 1e-9) / (time_lower * 1e-3);
    std::cout << "下限 (列复制) 时间: " << time_lower << " ms, 带宽: " << bandwidth_lower << " GB/s" << std::endl;

    // Naive
    CHECK_CUDA(cudaMemset(d_B, 0, nBytes));
    auto start_naive = std::chrono::high_resolution_clock::now();
    transposeNaive<<<grid, block>>>(d_A, d_B, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_naive = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();
    double bandwidth_naive = (2.0 * NXY * sizeof(float) * 1e-9) / (time_naive * 1e-3);
    CHECK_CUDA(cudaMemcpy(h_B, d_B, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Naive 时间: " << time_naive << " ms, 带宽: " << bandwidth_naive << " GB/s" << std::endl;

    // Unrolled
    CHECK_CUDA(cudaMemset(d_B, 0, nBytes));
    auto start_unroll = std::chrono::high_resolution_clock::now();
    transposeUnrolled<<<grid_unroll, block>>>(d_A, d_B, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_unroll = std::chrono::high_resolution_clock::now();
    double time_unroll = std::chrono::duration<double, std::milli>(end_unroll - start_unroll).count();
    double bandwidth_unroll = (2.0 * NXY * sizeof(float) * 1e-9) / (time_unroll * 1e-3);
    CHECK_CUDA(cudaMemcpy(h_B, d_B, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Unrolled 时间: " << time_unroll << " ms, 带宽: " << bandwidth_unroll << " GB/s (加速 " << (time_naive / time_unroll) << "x)" << std::endl;

    // Shared Memory 优化
    CHECK_CUDA(cudaMemset(d_B, 0, nBytes));
    auto start_shared = std::chrono::high_resolution_clock::now();
    transposeShared<<<grid, block>>>(d_A, d_B, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_shared = std::chrono::high_resolution_clock::now();
    double time_shared = std::chrono::duration<double, std::milli>(end_shared - start_shared).count();
    double bandwidth_shared = (2.0 * NXY * sizeof(float) * 1e-9) / (time_shared * 1e-3);
    CHECK_CUDA(cudaMemcpy(h_B, d_B, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Shared 时间: " << time_shared << " ms, 带宽: " << bandwidth_shared << " GB/s (vs Naive 加速 " << (time_naive / time_shared) << "x)" << std::endl;

    // 简单验证（使用Shared结果）
    bool correct = true;
    for (int i = 0; i < NXY; ++i) {
        if (fabs(h_B_cpu[i] - h_B[i]) > 1e-5) { correct = false; break; }
    }
    std::cout << "验证: " << (correct ? "正确" : "错误") << std::endl;

    // 清理
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B); free(h_B_cpu);

    return 0;
}
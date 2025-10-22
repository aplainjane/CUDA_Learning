#include <stdio.h>
#include <cuda_runtime.h>

// 定义块尺寸
#define BDIMX 32
#define BDIMY 32
#define IPAD 1  // 填充列数（测试1列填充）

// 核函数1: 行主序写+读 (静态，无填充) - 最优，无冲突
__global__ void setRowReadRow(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;  // 行主序写
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];  // 行主序读
}

// 核函数2: 列主序写+读 (静态，无填充) - 最差，32冲突
__global__ void setColReadCol(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;  // 列主序写
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];  // 列主序读
}

// 核函数3: 行主序写 + 列主序读 (静态，无填充) - 读时32冲突
__global__ void setRowReadCol(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;  // 行主序写
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];  // 列主序读
}

// 核函数4: 动态共享内存，行写+列读
__global__ void setRowReadColDyn(int *out) {
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx] = row_idx;  // 行主序写 (线性)
    __syncthreads();
    out[row_idx] = tile[col_idx];  // 列主序读 (线性)
}

// 核函数5: 静态填充，行写+列读 - 填充避免冲突
__global__ void setRowReadColIpad(int *out) {
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;  // 行主序写
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];  // 列主序读 (填充后无冲突)
}

// 核函数6: 动态填充，行写+列读
__global__ void setRowReadColDynIpad(int *out) {
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.y + IPAD) + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

// 热身核函数 (简单全局写，避免缓存影响)
__global__ void warmup(int *out) {
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    out[idx] = idx;
}

int main() {
    const int size = BDIMX * BDIMY;
    int *d_out, *h_out;
    
    // 主机/设备内存分配
    cudaMallocHost((void**)&h_out, size * sizeof(int));
    cudaMalloc((void**)&d_out, size * sizeof(int));
    
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);  // 单块测试
    
    // 运行每个核函数 (实际用nvprof/ncu包裹main)
    warmup<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();
    
    setRowReadRow<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    setColReadCol<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();
    
    setRowReadCol<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();
    
    // 动态: 指定共享内存大小
    setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_out);
    cudaDeviceSynchronize();
    
    setRowReadColIpad<<<grid, block>>>(d_out);
    cudaDeviceSynchronize();
    
    // 动态填充
    setRowReadColDynIpad<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_out);
    cudaDeviceSynchronize();
    
    // 清理
    cudaFree(d_out);
    cudaFreeHost(h_out);
    
    printf("Demo运行完成。使用nvprof或ncu测量性能。\n");
    printf("示例输出验证: h_out[0]=%d, h_out[1]=%d\n", h_out[0], h_out[1]);  // 应为0,1 (行主序)
    return 0;
}
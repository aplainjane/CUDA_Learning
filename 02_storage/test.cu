#include<cuda_runtime.h>
#include<iostream>
#include<cstring>
#include<cmath>

#define CHECK(call){ \
    cudaError_t error = call; \
    if(error != cudaSuccess){ \
        std::cout << "Error: " << cudaGetErrorString(error) << std::endl; \
        exit(-1); \
    } \
} \

void initialalize(float *a, int size){
    for(int i = 0; i < size; i++){
        a[i] = static_cast<float>(rand() % 10);
    }
}

__global__ void sumArraysGPU(float *a, float *b, float *res) {
    int i = threadIdx.x;  // 获取当前线程的索引（假设块大小等于数组大小）
    res[i] = a[i] + b[i];
}

int main(){
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    int size = 32;
    size_t bytes = size * sizeof(float);

    float *h_a, *h_b, *h_res;
    h_a = new float[size];
    h_b = new float[size];
    h_res = new float[size];

    initialalize(h_a, size);
    initialalize(h_b, size);

    float *d_a, *d_b, *d_res;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_res, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    sumArraysGPU<<<1, size>>>(d_a, d_b, d_res);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_res, d_res, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < size; i++){
        std::cout << h_res[i] << " ";
        std::cout << h_a[i] + h_b[i] << std::endl;
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    delete[] h_a;
    delete[] h_b;
    delete[] h_res;

    return 0;
}
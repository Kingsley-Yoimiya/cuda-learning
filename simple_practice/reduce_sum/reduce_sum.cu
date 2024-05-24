// %%writefile reduce_sum.cu

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

using namespace std;

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(const float* input_vecs, size_t n, size_t dim, float* output_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n * dim) {
        atomicAdd(&output_vec[idx % dim], input_vecs[idx]);
    }
}

void reduce_sum(const float* input_vecs, size_t n, size_t dim, float* output_vec) {
    float* cu_input_vecs;
    float* cu_output_vecs;
    size_t tot_size = n * dim * sizeof(float);
    cudaMalloc((void**) &cu_input_vecs, tot_size);
    cudaMalloc((void**) &cu_output_vecs, tot_size);
    cudaMemcpy(cu_input_vecs, input_vecs, tot_size, cudaMemcpyHostToDevice);
    cudaMemset(cu_output_vecs, 0, tot_size);
    int grid_size = (n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel <<< grid_size, BLOCK_SIZE >>>(cu_input_vecs, n, dim, cu_output_vecs);
    cudaMemcpy(output_vec, cu_output_vecs, tot_size, cudaMemcpyDeviceToHost);
    cudaFree(cu_input_vecs);
    cudaFree(cu_output_vecs);
}

const long long N = 2e9;

uniform_real_distribution<float> u(0, 1);
mt19937 rnd(chrono::system_clock::now().time_since_epoch().count());

int main() {
    float* input_vecs = new float[N];
    float* output_vec = new float[N];
    for(int i = 0; i < N; i++) {
        input_vecs[i] = i;
    }

    cerr << "GENERATE OK!" << endl;
    double st = clock();
    reduce_sum(input_vecs, N / 100, 100, output_vec);
    double ed = clock();
    std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

    std::cout << output_vec[0] << std::endl;
    delete[] input_vecs;
    delete[] output_vec;

    return 0;
}
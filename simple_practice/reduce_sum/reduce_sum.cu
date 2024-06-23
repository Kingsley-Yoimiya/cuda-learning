#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

using namespace std;

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(const float *input_vecs, size_t n, size_t dim,
                                  float *output_vec) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * dim) {
    atomicAdd(&output_vec[idx % dim], input_vecs[idx]);
  }
}

void reduce_sum(const float *input_vecs, size_t n, size_t dim,
                float *output_vec) {
  float *cu_input_vecs;
  float *cu_output_vecs;
  size_t input_size = n * dim * sizeof(float),
         output_size = dim * sizeof(float);
  cudaMalloc((void **)&cu_input_vecs, input_size);
  cudaMalloc((void **)&cu_output_vecs, output_size);
  cudaMemcpy(cu_input_vecs, input_vecs, input_size, cudaMemcpyHostToDevice);
  cudaMemset(cu_output_vecs, 0, output_size);
  size_t grid_size = (n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
  reduce_sum_kernel<<<grid_size, BLOCK_SIZE>>>(cu_input_vecs, n, dim,
                                               cu_output_vecs);
  cudaDeviceSynchronize();
  cudaMemcpy(output_vec, cu_output_vecs, output_size, cudaMemcpyDeviceToHost);
  cudaFree(cu_input_vecs);
  cudaFree(cu_output_vecs);
}

void reduce_sum_in(const float *input_vecs, size_t n, size_t dim, float *output_vec) {
  // 分配设备内存
  float *d_in = nullptr;
  float *d_out = nullptr;
  cudaMalloc(&d_in, n * dim * sizeof(float));
  cudaMalloc(&d_out, dim * sizeof(float));

  // 将数据从主机复制到设备
  cudaMemcpy(d_in, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice);

  // 为每个维度计算总和
  for (size_t i = 0; i < dim; ++i) {
      // 确定临时存储需求
      void *d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      
      // 定义指向第 i 个维度的指针
      float* d_in_dim = d_in + i;

      // 第一次调用用于计算临时存储大小
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in_dim, d_out + i, n);

      // 分配临时存储
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      
      // 第二次调用用于实际归约操作
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in_dim, d_out + i, n);
      
      // 释放临时存储
      cudaFree(d_temp_storage);
  }

  // 将结果从设备复制到主机
  cudaMemcpy(output_vec, d_out, dim * sizeof(float), cudaMemcpyDeviceToHost);

  // 释放设备内存
  cudaFree(d_in);
  cudaFree(d_out);
}

const long long N = 1e8;
const int T = 100;

uniform_real_distribution<float> u(0, 1);
mt19937 rnd(chrono::system_clock::now().time_since_epoch().count());

int main() {
  float *input_vecs = new float[N];
  float *output_vec = new float[T];
  float *correct_vec = new float[T];
  for (int i = 0; i < N; i++) {
    input_vecs[i] = 1. * i / N;
  }

  cerr << "GENERATE OK!" << endl;
  double st = clock();
  reduce_sum(input_vecs, N / T, T, output_vec);
  double ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  st = clock();

  // for (int i = 0; i < T; i++) {
  //   correct_vec[i] = 0;
  // }
  // for (int i = 0; i < N; i++) {
  //   correct_vec[i % T] += input_vecs[i];
  // }
  reduce_sum_in(input_vecs, N / T, T, correct_vec);

  ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  for (int i = 0; i < T; i++) {
    if (abs(correct_vec[i] - output_vec[i]) > 1) {
      std::cout << correct_vec[i] << " " << output_vec[i] << " ERROR!"
                << std::endl;
      break;
    }
  }

  std::cout << output_vec[0] << std::endl;
  delete[] input_vecs;
  delete[] output_vec;

  return 0;
}
/*
GENERATE OK!
0.536832
0.10144
5000 500004 ERROR!
500004
*/
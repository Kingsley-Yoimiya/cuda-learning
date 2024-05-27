#include <cuda.h>
#include <cuda_runtime.h>

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

  for (int i = 0; i < T; i++) {
    correct_vec[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    correct_vec[i % T] += input_vecs[i];
  }

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
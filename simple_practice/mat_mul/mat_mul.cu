#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

using namespace std;

#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define cudaCheckErrorSync()                                                   \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define cudaCheckErrorSync()                                                   \
  {}
#define cudaCheckError()                                                       \
  {}

__global__ void matmul_kernel(const float *A, const float *B, size_t n,
                              size_t m, size_t k, float *output) {
  size_t idi = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idj = blockIdx.y * blockDim.y + threadIdx.y;
  if (idi < n && idj < k) {
    float res = 0;
    for (int idk = 0; idk < m; idk++) {
      res += A[idi * m + idk] * B[idk * k + idj];
    }
    output[idi * k + idj] = res;
  }
}

void matmul(const float *A, const float *B, size_t n, size_t m, size_t k,
            float *output) {
  float *cu_A;
  float *cu_B;
  float *cu_output;
  size_t A_size = n * m * sizeof(float), B_size = m * k * sizeof(float),
         out_size = n * k * sizeof(float);
  cudaMalloc((void **)&cu_A, A_size);
  cudaCheckError();
  cudaMalloc((void **)&cu_B, B_size);
  cudaMalloc((void **)&cu_output, out_size);
  cudaCheckError();
  cudaMemcpy(cu_A, A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_B, B, B_size, cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemset(cu_output, 0, out_size);
  dim3 grid(GRID_SIZE, GRID_SIZE);
  dim3 block((n + GRID_SIZE - 1) / GRID_SIZE, (k + GRID_SIZE - 1) / GRID_SIZE);
  cerr << (n + GRID_SIZE - 1) / GRID_SIZE << " "
       << (k + GRID_SIZE - 1) / GRID_SIZE << endl;
  matmul_kernel<<<grid, block>>>(cu_A, cu_B, n, m, k, cu_output);
  cudaCheckErrorSync();

  cudaDeviceSynchronize();
  cudaMemcpy(output, cu_output, out_size, cudaMemcpyDeviceToHost);
  cudaCheckError();
  cudaFree(cu_A);
  cudaFree(cu_B);
  cudaFree(cu_output);
}

void matmul_sample(const float *A, const float *B, size_t n, size_t m, size_t k,
    float *output) {
  float *cu_A;
  float *cu_B;
  float *cu_output;
  size_t A_size = n * m * sizeof(float), B_size = m * k * sizeof(float),
         out_size = n * k * sizeof(float);
  
  cudaMalloc((void **)&cu_A, A_size);
  cudaMalloc((void **)&cu_B, B_size);
  cudaMalloc((void **)&cu_output, out_size);
  
  cudaMemcpy(cu_A, A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_B, B, B_size, cudaMemcpyHostToDevice);
  cudaMemset(cu_output, 0, out_size);
  
  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0, beta = 0.0; 
  cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, n, m, &alpha, cu_B, k, cu_A, m, &beta, cu_output, k);

  cudaMemcpy(output, cu_output, out_size, cudaMemcpyDeviceToHost);

  cudaFree(cu_A);
  cudaFree(cu_B);
  cudaFree(cu_output);
  cublasDestroy(handle);
}

const int N = 2048, M = 2048, K = 2048;
const int T = 100;

uniform_real_distribution<float> u(0, 100);
mt19937 rnd(chrono::system_clock::now().time_since_epoch().count());

int main() {
  float *A = new float[N * M];
  float *B = new float[M * K];
  float *C = new float[N * K];
  float *D = new float[N * K];
  for (int i = 0; i < N * M; i++) {
    A[i] = u(rnd);
  }
  for (int i = 0; i < M * K; i++) {
    B[i] = u(rnd);
  }
  for (int i = 0; i < N * K; i++) {
    C[i] = 0;
  }
  cerr << "GENERATE OK!" << endl;

  double st = clock();
  matmul_sample(A, B, N, M, K, C);
  double ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  st = clock();
  for (int i = 0; i < N * K; i++) {
    D[i] = 0;
  }
  matmul_sample(A, B, N, M, K, D);
/*
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        D[i * K + k] += A[i * M + j] * B[j * K + k];
      }
    }
  }
*/
  ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  float maxx = 0;
  for (int i = 0; i < N * K; i++) {
      maxx = max(maxx, fabs(C[i] - D[i]) / max(fabs(C[i]), fabs(D[i])));      
  }
  
  std::cout << "RESULT: " << C[0] << " " << maxx << std::endl;
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  return 0;
}
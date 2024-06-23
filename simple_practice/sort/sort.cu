#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

// using namespace std;

#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define BATCH 131072
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

// #define cudaCheckErrorSync() {}
// #define cudaCheckError() {}

int nextPow2(int x) {
  x--;
  for (int i = 1; i < 32; i <<= 1)
    x |= x >> i;
  return x + 1;
}

__global__ void sort_kernel(const int *input, int *output, int batch, int n,
                            int offset) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int stb = id * batch, edb = stb + batch;
  for (int id = stb; id < edb; id++) {
    int st = id * offset * 2, mid = st + offset, ed = mid + offset;
    st = min(st, n), mid = min(mid, n), ed = min(ed, n);
    int pos = st, l = st, r = mid;
    while (l < mid && r < ed) {
      if (input[l] < input[r])
        output[pos++] = input[l++];
      else
        output[pos++] = input[r++];
    }
    while (l < mid)
      output[pos++] = input[l++];
    while (r < ed)
      output[pos++] = input[r++];
  }
}
// 需要优化，直接访问 output 全局数组理论上不快。

void sort(std ::vector<int> &nums) {
  int *cu_in;
  int *cu_out;
  int n = nums.size();
  cudaMalloc((void **)&cu_in, n * sizeof(int));
  cudaMalloc((void **)&cu_out, n * sizeof(int));
  cudaMemcpy(cu_in, nums.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(cu_out, 0, n * sizeof(int));
  for (int offset = 1, tot = n + 1 >> 1; tot > 0;
       offset <<= 1, tot = tot + 1 >> 1) {
    int totcore = min(tot, GRID_SIZE * BLOCK_SIZE);
    int blocknum = min(totcore, BLOCK_SIZE);
    int gridnum = (totcore + blocknum - 1) / blocknum;
    int batch = (tot + blocknum * gridnum - 1) / (blocknum * gridnum);
    // std :: cout << totcore << " " << blocknum << " " << gridnum << " " <<
    // batch << std :: endl;
    sort_kernel<<<gridnum, blocknum>>>(cu_in, cu_out, batch, n, offset);
    cudaDeviceSynchronize();
    std ::swap(cu_in, cu_out);
    if (tot == 1)
      break;
  }
  cudaMemcpy(nums.data(), cu_in, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(cu_in);
  cudaFree(cu_out);
}

void sort_in(std::vector<int>& nums) {
  int num_items = nums.size();

  // 分配设备内存
  int *d_keys_in = nullptr;
  int *d_keys_out = nullptr;
  cudaMalloc(&d_keys_in, num_items * sizeof(int));
  cudaMalloc(&d_keys_out, num_items * sizeof(int));

  // 将数据从主机复制到设备
  cudaMemcpy(d_keys_in, nums.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);

  // 确定临时存储需求
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // 执行排序操作
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  // 将结果从设备复制到主机
  cudaMemcpy(nums.data(), d_keys_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

  // 释放设备内存
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
}

const int N = 5e6, M = 2000, K = 2000;
const int T = 100;

std ::uniform_real_distribution<float> u(0, 1);
std ::mt19937 rnd(std ::chrono::system_clock::now().time_since_epoch().count());

int main() {
  std ::vector<int> a, b, c;
  for (int i = 0; i < N; i++) {
    a.emplace_back(rnd() % 1024);
  }
  b = a, c = a;
  std ::cerr << "GENERATE OK!" << a.size() << std ::endl;

  double st = clock();
  sort(a);
  double ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  // for(int i = 0; i < 10; i++) cout << a[i] << endl;

  st = clock();
  
  //std ::sort(c.begin(), c.end());
  sort_in(c);

  ed = clock();
  std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

  for (int i = 0; i < N; i++)
    if (a[i] != c[i]) {
      std ::cout << i << " " << a[i] << " " << c[i] << std ::endl;
      break;
    }

  std ::cout << a.size() << " " << c.size() << std ::endl;
  assert(a == c);
  return 0;
}
/*
GENERATE OK!500000000
139.75
280.152(brute)
500000000 500000000
*/
/*
GENERATE OK!5000000
2.78544
0.01314(cub)
5000000 5000000
*/
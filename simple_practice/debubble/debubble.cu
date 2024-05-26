%%cuda

#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <assert.h>

using namespace std;

#define GRID_SIZE 128
#define BLOCK_SIZE 256
#define BATCH 65536
#define cudaCheckError() {                                                      \
    cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}
#define cudaCheckErrorSync() {                                                  \
    cudaDeviceSynchronize();                                                    \
    cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

// #define cudaCheckErrorSync() {}
// #define cudaCheckError() {}

int nextPow2(int x) {
    x--;
    for(int i = 1; i < 32; i <<= 1) x |= x >> i;
    return x + 1;
}

__global__ void debubble_kernel_1(const int *a, int n, int *blockpre) {
    __shared__ int s[BLOCK_SIZE];
    int st = (blockIdx.x * blockDim.x + threadIdx.x) * BATCH, ed = st + BATCH;
    int blockid = blockIdx.x;
    int cnt = 0;
    for(int i = min(st, n); i < min(ed, n); i++) cnt += a[i] != 0;
    s[threadIdx.x] = cnt;
    __syncthreads();
    for(int i = 1; i < blockDim.x; i <<= 1) {
        if(threadIdx.x + i < blockDim.x) s[threadIdx.x + i] += s[threadIdx.x];
        __syncthreads();
    }
    if(threadIdx.x == 0) blockpre[blockid] = s[blockDim.x - 1];
}

__global__ void debubble_kernel_2(const int *a, int n, const int *blockpre, int *output) {
    __shared__ int s[BLOCK_SIZE];
    int st = (blockIdx.x * blockDim.x + threadIdx.x) * BATCH, ed = st + BATCH;
    int cnt = 0;
    for(int i = min(st, n); i < min(ed, n); i++) cnt += a[i] != 0;
    s[threadIdx.x] = cnt;
    __syncthreads();
    for(int i = 1, totnum = blockDim.x >> 1; i < blockDim.x; i <<= 1, totnum >>= 1) {
        if(threadIdx.x < totnum) {
            int id1 = threadIdx.x * (i << 1) + i - 1, id2 = id1 + i;
            s[id2] += s[id1];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) s[blockDim.x - 1] = 0;
    for(int i = blockDim.x >> 1, totnum = 1; i > 0; i >>= 1, totnum <<= 1) {
        if(threadIdx.x < totnum) {
            int id1 = threadIdx.x * (i << 1) + i - 1, id2 = id1 + i;
            int tmp = s[id1];
            s[id1] = s[id2];
            s[id2] += tmp;
        }
        __syncthreads();
    }

    cnt = s[threadIdx.x] + blockpre[blockIdx.x];
    for(int i = min(st, n); i < min(ed, n); i++) {
        if(a[i] != 0) {
            output[cnt] = a[i];
            cnt++;
        }
    }
}

__global__ void preSum(int *a, int tot, size_t *ans) {
    __shared__ int s[BLOCK_SIZE];
    s[threadIdx.x] = a[threadIdx.x];
    __syncthreads();
    for(int i = 1, totnum = blockDim.x >> 1; i < blockDim.x; i <<= 1, totnum >>= 1) {
        if(threadIdx.x < totnum) {
            int id1 = threadIdx.x * (i << 1) + i - 1, id2 = id1 + i;
            s[id2] += s[id1];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) ans[0] = s[blockDim.x - 1], s[blockDim.x - 1] = 0;
    for(int i = blockDim.x >> 1, totnum = 1; i > 0; i >>= 1, totnum <<= 1) {
        if(threadIdx.x < totnum) {
            int id1 = threadIdx.x * (i << 1) + i - 1, id2 = id1 + i;
            int tmp = s[id1];
            s[id1] = s[id2];
            s[id2] += tmp;
        }
        __syncthreads();
    }
    a[threadIdx.x] = s[threadIdx.x];
}

size_t debubble(vector < int >&a) {
    int* cu_a;
    int* blockpre;
    int* output;
    int n = a.size(), totcore = (n + BATCH - 1) / BATCH, blocknum = nextPow2((totcore + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cout << totcore << " " << blocknum << endl;
    cudaMalloc((void**) &cu_a, n * sizeof(int));
    cudaCheckError();
    cudaMalloc((void**) &blockpre, (blocknum) * sizeof(int));
    cudaMalloc((void**) &output, n * sizeof(int));
    cudaMemcpy(cu_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemset(blockpre, 0, (blocknum)  * sizeof(int));
    cudaMemset(output, 0, n * sizeof(int));
    cudaCheckError();
    // cout << "!" << endl;
    debubble_kernel_1 <<< blocknum, BLOCK_SIZE >>>(cu_a, n, blockpre);
    cudaDeviceSynchronize();
    cudaCheckError();
    // cout << "!1" << " " << blocknum << endl;

    // illigal: 
    // for(int i = 1; i < blocknum; i++) cout << i << endl, blockpre[i] += blockpre[i - 1];
    size_t * cu_ans, ans;
    cudaMalloc((void**) &cu_ans, sizeof(size_t));
    preSum <<< 1, blocknum >>> (blockpre, blocknum, cu_ans);
    cudaMemcpy(&ans, cu_ans, sizeof(size_t), cudaMemcpyDeviceToHost);
    // cout << "!2" << " " << ans << endl;
    cudaCheckError();
    debubble_kernel_2 <<< blocknum, BLOCK_SIZE >>>(cu_a, n, blockpre, output);
    cudaCheckError();
    cudaDeviceSynchronize();
    // cout << "!3" << endl;
    cudaCheckError();
    // cout << "!" << endl;
    cudaMemcpy(a.data(), output, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    // cout << "!" << endl;
    cudaFree(cu_a);
    cudaFree(blockpre);
    cudaFree(output);
    cudaCheckError();
    return ans;
}

const int N = 536870912, M = 2000, K = 2000;
const int T = 100;

uniform_real_distribution<float> u(0, 1);
mt19937 rnd(chrono::system_clock::now().time_since_epoch().count());

int main() {
    vector < int > a, b, c;
    for(int i = 0; i < N; i++) {
        a.emplace_back(rnd() % 1024);
    }
    b = a;
    cerr << "GENERATE OK!" << a.size() << endl;

    double st = clock();
    cout << debubble(a) << endl;
    double ed = clock();
    std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;

    // for(int i = 0; i < 10; i++) cout << a[i] << endl;

    st = clock();
    for(int x : b) if(x != 0) c.emplace_back(x);
    while(c.size() < b.size()) c.emplace_back(0);
    
    ed = clock();
    std::cout << (ed - st) / CLOCKS_PER_SEC << std::endl;
    for(int i = 0; i < N; i++) if(a[i] != c[i]) {
        cout << i << " " << a[i] << " " << c[i] << endl;
        break;
    }

    cout << a.size() << " " << c.size() << endl;
    assert(a == c);
    return 0;
}
/*GENERATE OK!536870912
65536 256
536346554
2.01072
28.8447
536870912 536870912
*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>

// Kernel for batched matrix multiplication using cuBLAS
template <typename scalar_t>
void batched_matmul_cuda_impl(const scalar_t* A, const scalar_t* B, scalar_t* C, int batch_size, int m, int n, int k, cublasHandle_t handle);

// Specialization for float
template <>
void batched_matmul_cuda_impl<float>(const float* A, const float* B, float* C, int batch_size, int m, int n, int k, cublasHandle_t handle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int lda = m, ldb = k, ldc = m;
    int strideA = m * k, strideB = k * n, strideC = m * n;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batch_size
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SgemmStridedBatched failed");
    }
}

// Specialization for double
template <>
void batched_matmul_cuda_impl<double>(const double* A, const double* B, double* C, int batch_size, int m, int n, int k, cublasHandle_t handle) {
    const double alpha = 1.0;
    const double beta = 0.0;
    int lda = m, ldb = k, ldc = m;
    int strideA = m * k, strideB = k * n, strideC = m * n;

    cublasStatus_t status = cublasDgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batch_size
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS DgemmStridedBatched failed");
    }
}

// Function wrapper to be called from Pytorch
at::Tensor batched_matmul_cuda(at::Tensor A, at::Tensor B) {
    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = at::empty({batch_size, m, n}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "batched_matmul_cuda", ([&] {
        batched_matmul_cuda_impl<scalar_t>(
            A_.data_ptr<scalar_t>(),
            B_.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batch_size, m, n, k,
            handle
        );
    }));

    cublasDestroy(handle);
    return C;
}

// Wrapper for autograd
class BatchedMatMulFunction : public at::autograd::Function<BatchedMatMulFunction> {
public:
    static at::Tensor forward(at::autograd::AutogradContext *ctx, at::Tensor A, at::Tensor B) {
        ctx->save_for_backward({A, B});
        return batched_matmul_cuda(A, B);
    }

    static std::vector<at::Tensor> backward(at::autograd::AutogradContext *ctx, std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        auto A = saved[0];
        auto B = saved[1];
        auto grad_A = at::bmm(grad_output[0], B.transpose(1, 2));
        auto grad_B = at::bmm(A.transpose(1, 2), grad_output[0]);
        return {grad_A, grad_B};
    }
};

at::Tensor batched_matmul(at::Tensor A, at::Tensor B) {
    return BatchedMatMulFunction::apply(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Batched Matrix Multiplication with Autograd");
}

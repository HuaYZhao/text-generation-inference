#include <ATen/ATen.h>
#include <cassert>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")

void solve_fp16_gemv(__half *mat_ptr,
                     __half *vec_ptr,
                     __half *out_ptr,
                     unsigned int m,
                     unsigned int k,
                     unsigned int n,
                     unsigned int block_dim_x,
                     unsigned int block_dim_y);

torch::Tensor torch_matmul(torch::Tensor mat1, torch::Tensor mat2)
{
    // torch matmul， 调用的是cublas(Lt)/cutlass的矩阵乘

    torch::Tensor out = torch::matmul(mat1, mat2);
    return out;
}

torch::Tensor fast_fp16_gemv(torch::Tensor mat1,
                             torch::Tensor mat2,
                             unsigned int block_dim_x,
                             unsigned int block_dim_y)
{
    // fast gemv
    unsigned int m = mat1.size(0);
    unsigned int k = mat1.size(1);
    unsigned int n = mat2.size(1);
    assert(n == 1);

    torch::Tensor out = torch::empty({m, n}, mat1.options());
    solve_fp16_gemv(reinterpret_cast<__half *>(mat1.data_ptr()),
                    reinterpret_cast<__half *>(mat2.data_ptr()),
                    reinterpret_cast<__half *>(out.data_ptr()),
                    m, k, n,
                    block_dim_x, block_dim_y);

    return out;
}

torch::Tensor moe_gemm(torch::Tensor mat1,
                       torch::Tensor mat2,
                       unsigned int block_dim_x,
                       unsigned int block_dim_y)
{
    CHECK_CUDA(mat1);
    CHECK_CUDA(mat2);

    unsigned int m = mat1.size(0);
    unsigned int k = mat1.size(1);
    unsigned int n = mat2.size(1);

    // printf("m: %d, k: %d, n: %d\n", m, k, n);

    torch::Tensor out;
    if (n == 1)
    {
        CHECK_CONTIGUOUS(mat1);
        CHECK_CONTIGUOUS(mat2);
        out = fast_fp16_gemv(mat1, mat2, block_dim_x, block_dim_y);
    }
    else
    {
        out = torch_matmul(mat1, mat2);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cublas_gemm", &torch_matmul, "torch matmul with cublas/cutlass gemm");
    m.def("moe_gemm", &moe_gemm, "mixture of experts gemm", py::arg("mat1"), py::arg("mat2"), py::arg("block_dim_x") = 256, py::arg("block_dim_y") = 4);
}
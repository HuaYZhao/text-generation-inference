#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <torch/extension.h>

#include <cassert>
#include <chrono>

#include "fast_gemv.cuh"
#include "utility.cuh"

void solve_fp16_gemv(half *mat_ptr,
                     half *vec_ptr,
                     half *out_ptr,
                     unsigned int m,
                     unsigned int k,
                     unsigned int n,
                     unsigned int block_dim_x,
                     unsigned int block_dim_y)
{
    assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
    assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    unsigned int num_per_thread = k / block_dim_x;
    assert(num_per_thread >= 8);
    dim3 grid_dim(1, m / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_fp16<<<grid_dim, block_dim>>>(mat_ptr, vec_ptr, out_ptr,
                                       k, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
}
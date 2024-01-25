#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <chrono>

namespace py = pybind11;

class MyMatmulKernel
{
public:
    // 获取单例实例的静态成员函数
    static MyMatmulKernel &getInstance()
    {
        static MyMatmulKernel instance;
        return instance;
    }

    // 禁用拷贝构造函数和赋值运算符
    MyMatmulKernel(const MyMatmulKernel &) = delete;
    MyMatmulKernel &operator=(const MyMatmulKernel &) = delete;

    // 公用析构函数，因为要暴露到python，私有会报错
    ~MyMatmulKernel()
    {
        cudaStreamDestroy(stream_);
        cublasDestroy(handle_);
        cublasLtDestroy(cublaslt_handle_);
    }

    torch::Tensor Matmul(torch::Tensor A, torch::Tensor B)
    {
        // 获取输入张量的维度信息
        int64_t m = A.size(0);
        int64_t k = A.size(1);
        int64_t n = B.size(1);

        // 数据检查
        TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input tensors must be 2-dimensional");
        TORCH_CHECK(A.size(1) == B.size(0), "Incompatible tensor dimensions");
        TORCH_CHECK(A.scalar_type() == B.scalar_type(), "Incompatible tensor dtypes");

        // 创建输出张量，根据张量类型推断算子入参
        torch::Tensor C = torch::empty({m, n}, A.options());
        torch::ScalarType torch_type = C.scalar_type();

        if (torch_type == torch::kFloat16)
        {
            __half alpha = __float2half(1.0f);
            __half beta = __float2half(0.0f);

            cublasLtMatmulDesc_t operationDesc = NULL;
            cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
            cudaDataType_t scaleType = CUDA_R_16F;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
            cublasOperation_t transa = CUBLAS_OP_N;
            cublasOperation_t transb = CUBLAS_OP_N;

            // 创建描述符对象和计算配置对象
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, k, m, k);
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, n, k, n);
            cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n);
            // 设置矩阵乘法描述符参数
            cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));
            // 创建偏好对象，设置偏好参数, 获取最佳算法
            // cublasLtMatmulPreference_t preference;
            // cublasLtMatmulHeuristicResult_t heuristic_result;
            // cublasLtMatmulAlgo_t algo;
            // size_t workspace_size;
            // int return_count = 0;
            // cublasLtMatmulPreferenceCreate(&preference);
            // // cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
            // cublasLtMatmulAlgoGetHeuristic(cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristic_result, &return_count);
            // algo = heuristic_result.algo;

            cublasStatus_t status = cublasLtMatmul(cublaslt_handle_,
                                                   operationDesc, &alpha,
                                                   reinterpret_cast<const __half *>(B.data_ptr()), Bdesc,
                                                   reinterpret_cast<const __half *>(A.data_ptr()), Adesc,
                                                   &beta,
                                                   reinterpret_cast<__half *>(C.data_ptr()), Cdesc,
                                                   reinterpret_cast<__half *>(C.data_ptr()), Cdesc,
                                                   NULL,
                                                   NULL,
                                                   0,
                                                   stream_);
            cublasLtMatmulDescDestroy(operationDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            // cudaDeviceSynchronize();
            // cublasStatus_t status = cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            //                                      n, m, k, &alpha,
            //                                      reinterpret_cast<const __half *>(B.data_ptr()), CUDA_R_16F, n,
            //                                      reinterpret_cast<const __half *>(A.data_ptr()), CUDA_R_16F, k,
            //                                      &beta,
            //                                      reinterpret_cast<__half *>(C.data_ptr()), CUDA_R_16F, n,
            //                                      CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            // printf("status: %d\n", status);
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS FP16 matrix multiplication failed");
        }
        else if (torch_type == torch::kFloat32)
        {
            float alpha = 1.0f;
            float beta = 0.0f;
            cublasStatus_t status = cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k, &alpha,
                                                 reinterpret_cast<const float *>(B.data_ptr()), CUDA_R_32F, n,
                                                 reinterpret_cast<const float *>(A.data_ptr()), CUDA_R_32F, k,
                                                 &beta,
                                                 reinterpret_cast<float *>(C.data_ptr()), CUDA_R_32F, n,
                                                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS FP32 matrix multiplication failed");
        }
        else
        {
            TORCH_CHECK(false, "Unsupported tensor data type");
        }

        return C;
    }

    std::vector<torch::Tensor> BatchedMatmul(std::vector<torch::Tensor> AList, std::vector<torch::Tensor> BList)
    {
        // Check input sizes
        TORCH_CHECK(AList.size() == BList.size(), "Number of input tensors must be the same");
        TORCH_CHECK(!AList.empty() && !BList.empty(), "Input tensors must not be empty");

        // Get tensor dimensions
        int batch_size = AList.size();
        int m = AList[0].size(0);
        int n = BList[0].size(1);
        int k = AList[0].size(1);

        // 创建输出张量列表，获取张量类型
        std::vector<torch::Tensor> CList;
        CList.reserve(batch_size);
        torch::TensorOptions options = AList[0].options();
        for (int i = 0; i < batch_size; ++i)
        {
            CList.push_back(torch::empty({m, n}, options));
        }

        torch::ScalarType torch_type = AList[0].scalar_type();
        if (torch_type == torch::kFloat16)
        {
            const __half alpha = __float2half(1.0f);
            const __half beta = __float2half(0.0f);

            // 创建并赋值多个矩阵指针列表
            __half **h_A_array = new __half *[batch_size];
            __half **h_B_array = new __half *[batch_size];
            __half **h_C_array = new __half *[batch_size];
            for (int i = 0; i < batch_size; ++i)
            {
                h_A_array[i] = reinterpret_cast<__half *>(AList[i].data_ptr());
                h_B_array[i] = reinterpret_cast<__half *>(BList[i].data_ptr());

                // 在 C++ 中分配用于存储结果的内存空间
                h_C_array[i] = reinterpret_cast<__half *>(CList[i].data_ptr());
            }

            // 将指向指针的指针数组的内存从主机移到设备
            const __half **d_A_array;
            const __half **d_B_array;
            __half **d_C_array;
            cudaMalloc((const void **)&d_A_array, batch_size * sizeof(__half *));
            cudaMalloc((const void **)&d_B_array, batch_size * sizeof(__half *));
            cudaMalloc((void **)&d_C_array, batch_size * sizeof(__half *));
            cudaMemcpy(d_A_array, h_A_array, batch_size * sizeof(__half *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B_array, h_B_array, batch_size * sizeof(__half *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_C_array, h_C_array, batch_size * sizeof(__half *), cudaMemcpyHostToDevice);

            cublasStatus_t status;
            // status = cublasGemmBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            //                                             n, m, k, &alpha,
            //                                             (const void **)d_B_array, CUDA_R_16F, n,
            //                                             (const void **)d_A_array, CUDA_R_16F, k,
            //                                             &beta,
            //                                             (void **)d_C_array, CUDA_R_16F, n,
            //                                             batch_size,
            //                                             CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            for (int i = 0; i < batch_size; i++)
            {
                status = cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n, m, k, &alpha,
                                      reinterpret_cast<const __half *>(BList[i].data_ptr()), CUDA_R_16F, n,
                                      reinterpret_cast<const __half *>(AList[i].data_ptr()), CUDA_R_16F, k,
                                      &beta,
                                      reinterpret_cast<__half *>(CList[i].data_ptr()), CUDA_R_16F, n,
                                      CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            }

            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS FP16 matrix multiplication failed");
            delete[] h_A_array;
            delete[] h_B_array;
            delete[] h_C_array;
            cudaFree(d_A_array);
            cudaFree(d_B_array);
            cudaFree(d_C_array);
        }
        else if (torch_type == torch::kFloat32)
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            // 创建并赋值多个矩阵指针列表
            float **h_A_array = new float *[batch_size];
            float **h_B_array = new float *[batch_size];
            float **h_C_array = new float *[batch_size];
            for (int i = 0; i < batch_size; ++i)
            {
                h_A_array[i] = reinterpret_cast<float *>(AList[i].data_ptr());
                h_B_array[i] = reinterpret_cast<float *>(BList[i].data_ptr());

                // 在 C++ 中分配用于存储结果的内存空间
                h_C_array[i] = reinterpret_cast<float *>(CList[i].data_ptr());
            }

            // 将指向指针的指针数组的内存从主机移到设备
            const float **d_A_array;
            const float **d_B_array;
            float **d_C_array;
            cudaMalloc((const void **)&d_A_array, batch_size * sizeof(float *));
            cudaMalloc((const void **)&d_B_array, batch_size * sizeof(float *));
            cudaMalloc((void **)&d_C_array, batch_size * sizeof(float *));
            cudaMemcpy(d_A_array, h_A_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B_array, h_B_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_C_array, h_C_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

            cublasStatus_t status = cublasGemmBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                                        n, m, k, &alpha,
                                                        (const void **)d_B_array, CUDA_R_32F, n,
                                                        (const void **)d_A_array, CUDA_R_32F, k,
                                                        &beta,
                                                        (void **)d_C_array, CUDA_R_32F, n,
                                                        batch_size,
                                                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS FP32 matrix multiplication failed");
            delete[] h_A_array;
            delete[] h_B_array;
            delete[] h_C_array;
            cudaFree(d_A_array);
            cudaFree(d_B_array);
            cudaFree(d_C_array);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported tensor data type");
        }

        // 计算持续时间
        // float duration;
        // cudaEventElapsedTime(&duration, start, end);

        // // 输出执行时间（以毫秒为单位）
        // std::cout << "Cost time: " << duration << " milliseconds" << std::endl;

        return CList;
    }

    void test(std::vector<torch::Tensor> AList, std::vector<torch::Tensor> BList)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasStatus_t status;

        int m = 2; // A 的行数和 C 的行数
        int n = 4; // B 的列数和 C 的列数
        int k = 3; // A 的列数和 B 的行数
        float alpha = 1.0f;
        float beta = 0.0f;
        // 一、c++开辟内存，实现cublasGemmEx
        // float *A = new float[m * k];
        // float *B = new float[k * n];
        // for (int i = 0; i < m * k; i++)
        // {
        //     A[i] = float(i + 1);
        // }
        // for (int i = 0; i < k * n; i++)
        // {
        //     B[i] = float(i + 1);
        // }
        // float *d_A;
        // float *d_B;
        // float *d_C;
        // cudaMalloc((void **)&d_A, m * k * sizeof(float));
        // cudaMalloc((void **)&d_B, k * n * sizeof(float));
        // cudaMalloc((void **)&d_C, m * n * sizeof(float));
        // cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

        // status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //                       n, m, k, &alpha,
        //                       d_B, CUDA_R_32F, n,
        //                       d_A, CUDA_R_32F, k,
        //                       &beta,
        //                       d_C, CUDA_R_32F, n,
        //                       CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        // // 将结果从设备拷贝回主机
        // float *C = new float[m * n];
        // cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        // std::cout << "Matrix C:" << std::endl;
        // for (int i = 0; i < m; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << C[i * n + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // 二、利用从python传入的张量列表中的第一个张量进行计算
        // torch::Tensor t_C = torch::empty({m, n}, AList[0].options());
        // status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //                       n, m, k, &alpha,
        //                       reinterpret_cast<const float *>(BList[0].data_ptr()), CUDA_R_32F, n,
        //                       reinterpret_cast<const float *>(AList[0].data_ptr()), CUDA_R_32F, k,
        //                       &beta,
        //                       reinterpret_cast<float *>(t_C.data_ptr()), CUDA_R_32F, n,
        //                       CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // t_C = t_C.to(torch::kCPU);
        // float *data_ptr = t_C.data_ptr<float>();

        // std::cout << "Tensor data:" << std::endl;
        // for (int i = 0; i < t_C.numel(); ++i)
        // {
        //     std::cout << data_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        // 三、在c++开辟内存进行矩阵批计算
        // int batch_size = 1; // 批量大小

        // // 开辟主机上的 指针数组
        // float **h_A_array = new float *[batch_size];
        // float **h_B_array = new float *[batch_size];
        // float **h_C_array = new float *[batch_size];
        // // 开辟 主机 内存，并赋值
        // float *h_A = new float[m * k];
        // float *h_B = new float[k * n];
        // float *h_C = new float[m * n];
        // for (int j = 0; j < m * k; j++)
        // {
        //     h_A[j] = float(j + 1);
        // }
        // for (int j = 0; j < k * n; j++)
        // {
        //     h_B[j] = float(j + 1);
        // }

        // // 转移到设备
        // float *d_A;
        // float *d_B;
        // float *d_C;
        // cudaMalloc((void **)&d_A, m * k * sizeof(float));
        // cudaMalloc((void **)&d_B, k * n * sizeof(float));
        // cudaMalloc((void **)&d_C, m * n * sizeof(float));
        // cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);
        // // 设备指针赋值给上层指针
        // h_A_array[0] = d_A;
        // h_B_array[0] = d_B;
        // h_C_array[0] = d_C;

        // // 在设备上申请`指向设备指针`的指针数组，大小为batch_size
        // // 关键，要把指向设备指针的指针数组内存也放到设备上！
        // const float **d_A_array;
        // const float **d_B_array;
        // float **d_C_array;
        // cudaMalloc((void **)&d_A_array, batch_size * sizeof(float *));
        // cudaMalloc((void **)&d_B_array, batch_size * sizeof(float *));
        // cudaMalloc((void **)&d_C_array, batch_size * sizeof(float *));
        // cudaMemcpy(d_A_array, h_A_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_B_array, h_B_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_C_array, h_C_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

        // status = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //                              n, m, k, &alpha,
        //                              (void **)d_B_array, CUDA_R_32F, n,
        //                              (void **)d_A_array, CUDA_R_32F, k,
        //                              &beta,
        //                              (void **)d_C_array, CUDA_R_32F, n,
        //                              batch_size,
        //                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        // printf("sgemm status  %d, line(%d)\n", status, __LINE__);
        // // 将结果从设备拷贝回主机
        // cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "Matrix C:" << std::endl;
        // for (int i = 0; i < m; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << h_C[i * n + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // 四、从torch入参中读取张量列表进行批矩阵乘法
        int batch_size = 1; // 批量大小
        std::vector<torch::Tensor> CList;
        CList.reserve(batch_size);
        torch::TensorOptions options = AList[0].options();
        for (int i = 0; i < batch_size; ++i)
        {
            CList.push_back(torch::empty({m, n}, options));
        }

        // 开辟主机上的 指针数组
        float **h_A_array = new float *[batch_size];
        float **h_B_array = new float *[batch_size];
        float **h_C_array = new float *[batch_size];
        for (int i = 0; i < batch_size; i++)
        {
            h_A_array[i] = AList[i].data_ptr<float>();
            h_B_array[i] = BList[i].data_ptr<float>();
            h_C_array[i] = CList[i].data_ptr<float>();
        }
        // 在设备上申请`指向设备指针`的指针数组，大小为batch_size
        const float **d_A_array;
        const float **d_B_array;
        float **d_C_array;
        cudaMalloc((void **)&d_A_array, batch_size * sizeof(float *));
        cudaMalloc((void **)&d_B_array, batch_size * sizeof(float *));
        cudaMalloc((void **)&d_C_array, batch_size * sizeof(float *));
        cudaMemcpy(d_A_array, h_A_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_array, h_B_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_array, h_C_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

        status = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k, &alpha,
                                     (void **)d_B_array, CUDA_R_32F, n,
                                     (void **)d_A_array, CUDA_R_32F, k,
                                     &beta,
                                     (void **)d_C_array, CUDA_R_32F, n,
                                     batch_size,
                                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        CList[0] = CList[0].to(torch::kCPU);
        float *data_ptr = CList[0].data_ptr<float>();

        std::cout << "Tensor data:" << std::endl;
        for (int i = 0; i < CList[0].numel(); ++i)
        {
            std::cout << data_ptr[i] << " ";
        }
        std::cout << std::endl;

        printf("sgemm status  %d, line(%d)\n", status, __LINE__);
        cublasDestroy(handle);
    }

private:
    cudaStream_t stream_;
    cublasHandle_t handle_;
    cublasLtHandle_t cublaslt_handle_;
    // 私有构造函数
    MyMatmulKernel()
    {
        cudaStreamCreate(&stream_);
        cublasCreate(&handle_);
        cublasLtCreate(&cublaslt_handle_);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<MyMatmulKernel>(m, "MyMatmulKernel")
        .def_static("get_instance", &MyMatmulKernel::getInstance, py::return_value_policy::reference)
        .def("matmul", &MyMatmulKernel::Matmul)
        .def("batched_matmul", &MyMatmulKernel::BatchedMatmul)
        .def("test", &MyMatmulKernel::test);
}
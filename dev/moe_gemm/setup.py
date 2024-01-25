from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="MoeMatmul",
    packages=find_packages(),
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "moe_matmul",  # operator name
            sources=[
                "./src/moe_gemm.cpp",
                "./src/fast_gemv/fast_gemv.cu",
                "./src/fast_gemv/fast_gemv_main.cu",
                "./src/fast_gemv/utility.cu",
            ],
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        ),
        CUDAExtension(
            "base_matmul",
            [
                "./src/gemm/base_matmul.cc",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

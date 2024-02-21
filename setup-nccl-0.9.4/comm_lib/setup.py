from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="comm_lib",
    include_dirs=["/usr/local/cuda/targets/x86_64-linux/include/"],
    ext_modules=[
        CUDAExtension("comm_lib", ["comm.cc"], libraries=["mpi"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)

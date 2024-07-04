from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_home = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"

setup(
    name='cu_transformer_extension',
    ext_modules=[
        CUDAExtension(
            name='cu_transformer_extension',
            sources=[
                'cu_transformer.cu',
            ],
            include_dirs=[os.path.join(cuda_home, 'include')],
            library_dirs=[os.path.join(cuda_home, 'lib\\x64')],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': [
                    '-allow-unsupported-compiler',
                    '-lcublas',
                ],
            },
            extra_link_flags=['-lcuda', '-lcublas'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

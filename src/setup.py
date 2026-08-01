from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='t4_kernels',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='t4_kernels',
            sources=[
                os.path.join(src_dir, 'bindings.cpp'),
                os.path.join(src_dir, 'kernels/lop3_dequant.cu'),
                os.path.join(src_dir, 'kernels/fused_w4a16_gemm.cu'),
                os.path.join(src_dir, 'kernels/h17_mega_kernel.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_75,code=sm_75',
                    '--use_fast_math',
                    '-Xptxas=-v',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

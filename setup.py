import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [os.path.join('backend', f'pybind_cuda.cpp'), 
        os.path.join('backend', f'spconv.cu')]

setup(
    name='FGMS',
    version='1.3',
    ext_modules=[
        CUDAExtension('FGMS.backend',
            sources=sources,
            extra_compile_args = {
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
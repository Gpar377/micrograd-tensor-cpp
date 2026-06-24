from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class Pybind11Extension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class BuildExt(build_ext):
    def build_extensions(self):
        # Resolve pybind11 include headers dynamically
        import pybind11
        pybind_include = pybind11.get_include()
        
        for ext in self.extensions:
            ext.include_dirs.append(pybind_include)
            
            # Enforce C++17 compilation standard flags
            if sys.platform == 'win32':
                ext.extra_compile_args = ['/std:c++17', '/openmp', '/O2']
            else:
                ext.extra_compile_args = ['-std=c++17', '-fopenmp', '-O3']
                ext.extra_link_args = ['-fopenmp']
                
        super().build_extensions()

setup(
    name='autograd',
    version='0.1',
    packages=['autograd'],
    ext_modules=[
        Pybind11Extension(
            'autograd_backend',
            sources=['src/tensor_ops.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExt
    },
    zip_safe=False,
    install_requires=[
        'numpy',
        'pybind11'
    ]
)

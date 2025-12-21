# File: setup.py (Đã sửa)
from setuptools import setup, Extension
import pybind11
import os

# Định nghĩa các cờ biên dịch tùy thuộc vào hệ điều hành
if os.name == 'nt': # Windows
    # MSVC không dùng -O3, -std=c++17, -fopenmp
    compile_args = ['/EHsc', '/std:c++17'] 
    link_args = []
else: # Linux/macOS
    compile_args = ['-O3', '-std=c++17', '-fopenmp']
    link_args = ['-fopenmp', '-pthread']

# Định nghĩa module mở rộng
ext_modules = [
    Extension(
        "tree_core",           
        ["tree_core.cpp"],     
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=compile_args, 
        extra_link_args=link_args 
    ),
]

setup(
    name="tree_core",
    version="1.0",
    ext_modules=ext_modules,
)
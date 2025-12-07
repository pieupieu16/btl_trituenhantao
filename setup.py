# File: setup.py
from setuptools import setup, Extension
import pybind11

# Định nghĩa module mở rộng
ext_modules = [
    Extension(
        "tree_core",           # Tên module sẽ import trong Python
        ["tree_core.cpp"],     # File nguồn C++
        include_dirs=[pybind11.get_include()],
        language="c++",
        # Trên Windows thường dùng cờ /O2, trên Linux dùng -O3 để tối ưu tốc độ
        extra_compile_args=["/O2"] 
    ),
]

setup(
    name="tree_core",
    version="1.0",
    ext_modules=ext_modules,
)
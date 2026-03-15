from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check for CUDA availability
if not os.environ.get("FORCE_CUDA", "0") == "1":
    # Fallback to C++ only for CPU building (mostly for CI)
    ext_modules = []
else:
    ext_modules = [
        CUDAExtension(
            "quantized_mesh._quantized_mesh_cpp",
            [
                "src/binding.cpp",
                "src/kernels/attention.cu",
                "src/kernels/quantization.cu",
            ],
            include_dirs=[os.path.join(os.getcwd(), "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-arch=sm_80"],
            },
        )
    ]

setup(
    name="quantized_mesh",
    version="0.1.0",
    author="Omer Antverg",
    description="High-performance quantized attention kernels for edge transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)

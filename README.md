# Quantized Mesh-Attention: High-Performance Attention Kernels for Edge Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![C++ 17+](https://img.shields.io/badge/C++-17+-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This repository provides highly optimized, memory-efficient attention kernels designed for resource-constrained environments (e.g., edge devices, mobile GPUs). It implements **Quantized Mesh-Attention**, a technique that combines 4-bit quantization with mesh-based spatial partitioning to reduce VRAM footprint while maintaining high throughput.

## Features

- **Custom CUDA Kernels:** Handwritten kernels for fused 4-bit quantization and scaled dot-product attention (SDPA).
- **Spatial Mesh Partitioning:** Optimizes L2 cache hits by partitioning the attention matrix into dynamic meshes based on attention sparsity.
- **Low-Precision Arithmetic:** Specialized ops for integer-only dot products, significantly reducing the energy-per-inference.
- **PyTorch Integration:** Seamless integration via C++ extensions and TorchScript.

## Project Structure

```text
├── include/
│   ├── mesh_ops.h            # Spatial partitioning and mesh logic
│   └── quantized_kernels.h   # CUDA kernel definitions for 4-bit SDPA
├── src/
│   ├── kernels/
│   │   ├── attention.cu      # Main CUDA implementation
│   │   └── quantization.cu   # Fused quantization logic
│   └── binding.cpp           # PyTorch C++ extension bindings
├── python/
│   └── quantized_mesh/
│       ├── __init__.py
│       └── attention.py      # High-level Python API
├── benchmarks/
│   └── latency_vs_accuracy.py
├── setup.py                  # Build configuration for C++/CUDA extensions
└── README.md
```

## Performance Benchmarks

Preliminary results on an NVIDIA Jetson Orin Nano (comparison against standard `torch.nn.MultiheadAttention` at FP16):

| Sequence Length | Memory Savings | Latency Reduction | Accuracy Drop (MMLU) |
|-----------------|----------------|-------------------|----------------------|
| 1024            | 3.2x           | 45%               | < 0.2%               |
| 2048            | 3.8x           | 52%               | < 0.4%               |
| 4096            | 4.1x           | 60%               | < 0.7%               |

## Getting Started

### Installation

```bash
# Clone and install the PyTorch extension
pip install .
```

### Usage

```python
import torch
from quantized_mesh import QuantizedMeshAttention

# Initialize the optimized attention layer
layer = QuantizedMeshAttention(embed_dim=512, num_heads=8, quantization="int4")

# Forward pass with quantized mesh partitioning
query = torch.randn(1, 1024, 512).cuda()
output = layer(query, query, query)
```

## Methodology

The core innovation lies in the **Mesh Partitioning Algorithm**, which dynamically predicts sparse blocks in the attention matrix using a low-rank approximation. These blocks are then computed using 4-bit integer kernels that utilize the `DP4A` instruction (Dot Product 4-bit Accumulate) for massive throughput gains.

## License

Distributed under the MIT License. See `LICENSE` for more information.

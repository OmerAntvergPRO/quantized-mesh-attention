#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

/**
 * Optimized kernel for 4-bit Quantized Scaled Dot Product Attention.
 * Leverages DP4A instruction for high-throughput integer arithmetic.
 */
namespace quantized_mesh {

/**
 * Fused Quantization and Mesh-Partitioned Attention Kernel.
 * 
 * @param query (torch::Tensor) - [batch, seq_len, head_dim]
 * @param key (torch::Tensor) - [batch, seq_len, head_dim]
 * @param value (torch::Tensor) - [batch, seq_len, head_dim]
 * @param scale (float) - Softmax scale factor
 * @return (torch::Tensor) - Attention output
 */
torch::Tensor mesh_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    float scale
);

/**
 * Fused 4-bit Weight Quantization (Linear scale).
 * 
 * @param weight (torch::Tensor) - Input weights (FP16/FP32)
 * @param q_weight (torch::Tensor) - Output quantized weights (INT4 packed into INT8)
 * @param scale (torch::Tensor) - Quantization scale
 */
void pack_int4_weights(
    const torch::Tensor& weight,
    torch::Tensor& q_weight,
    torch::Tensor& scale
);

} // namespace quantized_mesh

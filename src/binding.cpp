#include <torch/extension.h>
#include "quantized_kernels.h"

namespace quantized_mesh {

// Forward declaration of the CUDA kernel wrapper
torch::Tensor mesh_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    float scale
);

// PyBind11 definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantized Mesh-Attention C++ Extension";
    
    m.def("mesh_attention_forward", &mesh_attention_forward, 
          "Quantized Mesh-Attention forward pass (CUDA)");
}

} // namespace quantized_mesh

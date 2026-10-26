#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tinyinfer {

struct Config {
    int n_layer = 0;
    int n_head = 0;
    int n_embd = 0;      // C
    int n_ctx = 0;       // max sequence length
    int vocab_size = 0;  // V
};

// Linear weights are row-major [out, in]: output j = dot(row j, input) + bias[j].
// All pointers point into Model::data and stay valid for the Model's lifetime.
struct LayerWeights {
    const float* ln_1_w;       // [C]
    const float* ln_1_b;       // [C]
    const float* attn_qkv_w;   // [3C, C]  rows 0..C-1 = Q, C..2C-1 = K, 2C..3C-1 = V
    const float* attn_qkv_b;   // [3C]
    const float* attn_proj_w;  // [C, C]
    const float* attn_proj_b;  // [C]
    const float* ln_2_w;       // [C]
    const float* ln_2_b;       // [C]
    const float* mlp_fc_w;     // [4C, C]
    const float* mlp_fc_b;     // [4C]
    const float* mlp_proj_w;   // [C, 4C]
    const float* mlp_proj_b;   // [C]
};

struct Weights {
    const float* wte;  // [V, C]  token embeddings, also the (tied) LM head
    const float* wpe;  // [n_ctx, C]  position embeddings
    std::vector<LayerWeights> layers;
    const float* ln_f_w;  // [C]
    const float* ln_f_b;  // [C]
};

struct Model {
    Config config;
    Weights weights;
    std::vector<float> data;  // owns every weight

    size_t num_params() const { return data.size(); }
};

// Loads a file written by scripts/export_gpt2.py. Throws std::runtime_error
// on a bad magic number, version mismatch, or truncated file.
Model load_model(const std::string& path);

}  // namespace tinyinfer

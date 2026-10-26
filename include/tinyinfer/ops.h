#pragma once

#include <stdexcept>
#include <vector>

#include "tinyinfer/model.h"

namespace tinyinfer {

// Thrown by stubs that haven't been written yet. Tests treat it as "skip".
struct NotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

// ---- Building blocks (single vector, no batching) -------------------------

// out = (x - mean) / sqrt(var + 1e-5) * w + b, over C elements.
void layernorm(float* out, const float* x, const float* w, const float* b, int C);

// out[j] = dot(W[j, :], x) + bias[j]   for j in [0, out_dim).
// W is row-major [out_dim, in_dim]. bias may be nullptr.
void linear(float* out, const float* x, const float* W, const float* bias, int in_dim, int out_dim);

// In-place GELU, tanh approximation (what GPT-2 was trained with).
void gelu(float* x, int n);

// In-place softmax over n elements. Subtract the max first for stability.
void softmax(float* x, int n);

// ---- Full model ------------------------------------------------------------

// Runs GPT-2 on `tokens` and returns logits for every position,
// row-major [tokens.size(), vocab_size].
std::vector<float> forward(const Model& model, const std::vector<int>& tokens);

}  // namespace tinyinfer

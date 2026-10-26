#include "tinyinfer/ops.h"

#include <algorithm>
#include <cmath>

namespace tinyinfer {

// Week 1: these are yours to write. Each test in tests/ is skipped until the
// matching stub stops throwing NotImplemented.

void layernorm(float* out, const float* x, const float* w, const float* b, int C) {
    (void)out, (void)x, (void)w, (void)b, (void)C;
    throw NotImplemented("layernorm");
}

void linear(float* out, const float* x, const float* W, const float* bias, int in_dim, int out_dim) {
    (void)out, (void)x, (void)W, (void)bias, (void)in_dim, (void)out_dim;
    throw NotImplemented("linear");
}

void gelu(float* x, int n) {
    (void)x, (void)n;
    throw NotImplemented("gelu");
}

void softmax(float* x, int n) {
    // Loop 1: find the largest value
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        max_val = std::max(max_val, x[i]);
    }

    // Loop 2: exponentiate (shifted by max so nothing overflows) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Loop 3: normalize so everything adds to 1
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

// The forward pass, per token position t:
//   x = wte[token] + wpe[t]
//   for each layer:
//     x = x + attn_proj(causal_self_attention(layernorm_1(x)))
//     x = x + mlp_proj(gelu(mlp_fc(layernorm_2(x))))
//   logits = wte @ layernorm_f(x)      (tied LM head)
// Attention needs every earlier position's K and V, so process all positions
// layer by layer rather than one token through the whole stack.
std::vector<float> forward(const Model& model, const std::vector<int>& tokens) {
    (void)model, (void)tokens;
    throw NotImplemented("forward");
}

}  // namespace tinyinfer

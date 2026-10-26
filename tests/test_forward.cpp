#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include "tinyinfer/model.h"
#include "tinyinfer/ops.h"

using namespace tinyinfer;

// Unwritten ops show up as skipped, not failed.
#define SKIP_IF_TODO(stmt)                                     \
    try {                                                      \
        stmt;                                                  \
    } catch (const NotImplemented& e) {                        \
        GTEST_SKIP() << e.what() << " not implemented yet";    \
    }

// ---- Ops, checked against hand-computed values ------------------------------

TEST(Ops, Layernorm) {
    float x[4] = {1, 2, 3, 4}, w[4] = {1, 1, 1, 1}, b[4] = {0, 0, 0, 0}, out[4];
    SKIP_IF_TODO(layernorm(out, x, w, b, 4));
    const float want[4] = {-1.341635f, -0.447212f, 0.447212f, 1.341635f};
    for (int i = 0; i < 4; i++) EXPECT_NEAR(out[i], want[i], 1e-5);
}

TEST(Ops, Linear) {
    // W is [out=3, in=2]
    float W[6] = {1, 2, 3, 4, 5, 6}, x[2] = {1, 1}, bias[3] = {0.5f, 0, -1}, out[3];
    SKIP_IF_TODO(linear(out, x, W, bias, 2, 3));
    EXPECT_FLOAT_EQ(out[0], 3.5f);
    EXPECT_FLOAT_EQ(out[1], 7.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
}

TEST(Ops, Gelu) {
    float x[3] = {-1, 0, 1};
    SKIP_IF_TODO(gelu(x, 3));
    EXPECT_NEAR(x[0], -0.158808f, 1e-5);
    EXPECT_NEAR(x[1], 0.0f, 1e-7);
    EXPECT_NEAR(x[2], 0.841192f, 1e-5);
}

TEST(Ops, SoftmaxIsStableForLargeInputs) {
    float x[3] = {1001, 1002, 1003};  // naive exp() overflows here
    SKIP_IF_TODO(softmax(x, 3));
    EXPECT_NEAR(x[0], 0.0900306f, 1e-6);
    EXPECT_NEAR(x[1], 0.2447285f, 1e-6);
    EXPECT_NEAR(x[2], 0.6652410f, 1e-6);
}

// ---- Full forward pass vs. PyTorch -------------------------------------------

namespace {

struct Reference {
    std::vector<int> tokens;
    int vocab_size = 0;
    std::vector<float> logits;  // [tokens.size(), vocab_size]
};

Reference load_reference(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint32_t magic;
    int32_t n, v;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&v), 4);
    if (!f || magic != 0x46455254) throw std::runtime_error("bad reference file " + path);
    Reference r;
    r.vocab_size = v;
    std::vector<int32_t> toks(n);
    f.read(reinterpret_cast<char*>(toks.data()), n * 4);
    r.tokens.assign(toks.begin(), toks.end());
    r.logits.resize(size_t(n) * v);
    f.read(reinterpret_cast<char*>(r.logits.data()), r.logits.size() * 4);
    return r;
}

}  // namespace

TEST(Forward, MatchesPyTorchLogits) {
    const std::string dir = TINYINFER_MODELS_DIR;
    if (!std::filesystem::exists(dir + "/gpt2.bin")) GTEST_SKIP() << "run scripts/export_gpt2.py first";

    Model m = load_model(dir + "/gpt2.bin");
    Reference ref = load_reference(dir + "/gpt2_ref.bin");

    std::vector<float> logits;
    SKIP_IF_TODO(logits = forward(m, ref.tokens));
    ASSERT_EQ(logits.size(), ref.logits.size());

    float max_err = 0;
    for (size_t i = 0; i < logits.size(); i++)
        max_err = std::max(max_err, std::fabs(logits[i] - ref.logits[i]));
    // GPT-2 logits are in the -100s, so 1e-3 absolute is ~1e-5 relative.
    EXPECT_LT(max_err, 1e-3) << "max abs logit error vs PyTorch";
}

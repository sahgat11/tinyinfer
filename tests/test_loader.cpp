#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "tinyinfer/model.h"

using namespace tinyinfer;

namespace {

const std::string kModelPath = std::string(TINYINFER_MODELS_DIR) + "/gpt2.bin";

bool model_exported() { return std::filesystem::exists(kModelPath); }

}  // namespace

TEST(Loader, Gpt2SmallConfig) {
    if (!model_exported()) GTEST_SKIP() << "run scripts/export_gpt2.py first";
    Model m = load_model(kModelPath);
    EXPECT_EQ(m.config.n_layer, 12);
    EXPECT_EQ(m.config.n_head, 12);
    EXPECT_EQ(m.config.n_embd, 768);
    EXPECT_EQ(m.config.n_ctx, 1024);
    EXPECT_EQ(m.config.vocab_size, 50257);
    EXPECT_EQ(m.num_params(), 124439808u);
}

TEST(Loader, PointersCoverWholeBuffer) {
    if (!model_exported()) GTEST_SKIP() << "run scripts/export_gpt2.py first";
    Model m = load_model(kModelPath);
    const float* begin = m.data.data();
    const float* end = begin + m.data.size();
    // ln_f is the last tensor; if the walk is off by anything, this fails.
    EXPECT_EQ(m.weights.ln_f_b + m.config.n_embd, end);
    EXPECT_EQ(m.weights.wte, begin);
}

TEST(Loader, LayerNormGainsLookSane) {
    // LayerNorm gains are learned around 1, unlike other weights. A wrong
    // offset would land in a matmul tensor and break this.
    if (!model_exported()) GTEST_SKIP() << "run scripts/export_gpt2.py first";
    Model m = load_model(kModelPath);
    for (const LayerWeights& l : m.weights.layers) {
        double mean = 0;
        for (int i = 0; i < m.config.n_embd; i++) mean += l.ln_1_w[i];
        mean /= m.config.n_embd;
        EXPECT_GT(mean, 0.0);
        EXPECT_LT(mean, 2.0);
    }
}

TEST(Loader, RejectsBadMagic) {
    const std::string path = testing::TempDir() + "/bad.bin";
    {
        std::ofstream f(path, std::ios::binary);
        std::string junk(512, 'x');
        f.write(junk.data(), junk.size());
    }
    EXPECT_THROW(load_model(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Loader, RejectsMissingFile) {
    EXPECT_THROW(load_model("/nonexistent/model.bin"), std::runtime_error);
}

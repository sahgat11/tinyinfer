#include "tinyinfer/model.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tinyinfer {

namespace {

constexpr uint32_t kMagic = 0x464E4954;  // "TINF"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 256;

// Total float count for a config; must match the order in export_gpt2.py.
size_t expected_floats(const Config& c) {
    const size_t C = c.n_embd;
    const size_t per_layer = 2 * C            // ln_1
                           + 3 * C * C + 3 * C  // attn qkv
                           + C * C + C          // attn proj
                           + 2 * C              // ln_2
                           + 4 * C * C + 4 * C  // mlp fc
                           + 4 * C * C + C;     // mlp proj
    return size_t(c.vocab_size) * C + size_t(c.n_ctx) * C + c.n_layer * per_layer + 2 * C;
}

}  // namespace

Model load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t file_bytes = f.tellg();
    f.seekg(0);

    char header[kHeaderBytes];
    if (file_bytes < kHeaderBytes || !f.read(header, kHeaderBytes))
        throw std::runtime_error(path + ": file too small for header");

    uint32_t magic, version;
    int32_t dims[5];
    std::memcpy(&magic, header, 4);
    std::memcpy(&version, header + 4, 4);
    std::memcpy(dims, header + 8, sizeof(dims));
    if (magic != kMagic) throw std::runtime_error(path + ": bad magic (not a tinyinfer file)");
    if (version != kVersion)
        throw std::runtime_error(path + ": unsupported version " + std::to_string(version));

    Model m;
    m.config = {dims[0], dims[1], dims[2], dims[3], dims[4]};
    const Config& c = m.config;
    if (c.n_embd % c.n_head != 0) throw std::runtime_error(path + ": n_embd not divisible by n_head");

    const size_t n = expected_floats(c);
    if (file_bytes != kHeaderBytes + n * sizeof(float))
        throw std::runtime_error(path + ": size mismatch, expected " +
                                 std::to_string(kHeaderBytes + n * sizeof(float)) + " bytes, got " +
                                 std::to_string(file_bytes));

    m.data.resize(n);
    if (!f.read(reinterpret_cast<char*>(m.data.data()), n * sizeof(float)))
        throw std::runtime_error(path + ": read failed");

    // Walk the buffer in file order, handing out pointers.
    const float* p = m.data.data();
    auto take = [&p](size_t count) {
        const float* t = p;
        p += count;
        return t;
    };
    const size_t C = c.n_embd;
    Weights& w = m.weights;
    w.wte = take(size_t(c.vocab_size) * C);
    w.wpe = take(size_t(c.n_ctx) * C);
    w.layers.resize(c.n_layer);
    for (LayerWeights& l : w.layers) {
        l.ln_1_w = take(C);
        l.ln_1_b = take(C);
        l.attn_qkv_w = take(3 * C * C);
        l.attn_qkv_b = take(3 * C);
        l.attn_proj_w = take(C * C);
        l.attn_proj_b = take(C);
        l.ln_2_w = take(C);
        l.ln_2_b = take(C);
        l.mlp_fc_w = take(4 * C * C);
        l.mlp_fc_b = take(4 * C);
        l.mlp_proj_w = take(4 * C * C);
        l.mlp_proj_b = take(C);
    }
    w.ln_f_w = take(C);
    w.ln_f_b = take(C);

    return m;
}

}  // namespace tinyinfer

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>

#include "tinyinfer/model.h"

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "models/gpt2.bin";

    try {
        auto t0 = std::chrono::steady_clock::now();
        tinyinfer::Model m = tinyinfer::load_model(path);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const auto& c = m.config;
        std::printf("loaded %s in %.0f ms\n", path.c_str(), ms);
        std::printf("  layers=%d heads=%d embd=%d ctx=%d vocab=%d\n", c.n_layer, c.n_head, c.n_embd,
                    c.n_ctx, c.vocab_size);
        std::printf("  params=%zu (%.1f MB fp32)\n", m.num_params(), m.num_params() * 4 / 1e6);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

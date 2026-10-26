"""Export Hugging Face GPT-2 weights into tinyinfer's flat binary format.

Also writes a reference file (prompt tokens + PyTorch logits) that the C++
tests compare against, so every optimization can be checked for correctness.

Usage:
    .venv/bin/python scripts/export_gpt2.py            # gpt2 (124M)
    .venv/bin/python scripts/export_gpt2.py --model gpt2-medium

File format (all little-endian):

  models/<name>.bin
    header, 256 bytes:
      uint32 magic    = 0x464E4954  ("TINF")
      uint32 version  = 1
      int32  n_layer, n_head, n_embd, n_ctx, vocab_size
      zero padding to 256 bytes
    float32 tensors, in this order:
      wte   [vocab_size, n_embd]
      wpe   [n_ctx, n_embd]
      for each layer:
        ln_1.weight [C]       ln_1.bias [C]
        attn_qkv.weight [3C, C]   attn_qkv.bias [3C]
        attn_proj.weight [C, C]   attn_proj.bias [C]
        ln_2.weight [C]       ln_2.bias [C]
        mlp_fc.weight [4C, C]     mlp_fc.bias [4C]
        mlp_proj.weight [C, 4C]   mlp_proj.bias [C]
      ln_f.weight [C]  ln_f.bias [C]
    The LM head is tied to wte, so it is not stored separately.

    Linear weights are stored [out, in] (row-major): output j is the dot
    product of the input with row j. HF's Conv1D stores [in, out], so those
    are transposed here.

  models/<name>_ref.bin
    uint32 magic = 0x46455254  ("TREF")
    int32  n_tokens, vocab_size
    int32  tokens[n_tokens]
    float32 logits[n_tokens, vocab_size]   (every position, not just the last)
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import tiktoken
import torch
from transformers import GPT2LMHeadModel

MAGIC = 0x464E4954
REF_MAGIC = 0x46455254
VERSION = 1
HEADER_BYTES = 256

REF_PROMPT = "The quick brown fox jumps over the lazy dog. Once upon a time"


def write_tensor(f, t: torch.Tensor) -> int:
    arr = t.detach().to(torch.float32).contiguous().numpy()
    f.write(arr.tobytes())
    return arr.size


def export_weights(model: GPT2LMHeadModel, path: Path) -> None:
    cfg = model.config
    sd = model.state_dict()
    C = cfg.n_embd

    header = struct.pack(
        "<II5i", MAGIC, VERSION, cfg.n_layer, cfg.n_head, C, cfg.n_positions, cfg.vocab_size
    )
    header += b"\0" * (HEADER_BYTES - len(header))

    n_floats = 0
    with open(path, "wb") as f:
        f.write(header)
        n_floats += write_tensor(f, sd["transformer.wte.weight"])
        n_floats += write_tensor(f, sd["transformer.wpe.weight"])
        for i in range(cfg.n_layer):
            p = f"transformer.h.{i}."
            # Conv1D weights are [in, out]; .T gives [out, in].
            tensors = [
                sd[p + "ln_1.weight"], sd[p + "ln_1.bias"],
                sd[p + "attn.c_attn.weight"].T, sd[p + "attn.c_attn.bias"],
                sd[p + "attn.c_proj.weight"].T, sd[p + "attn.c_proj.bias"],
                sd[p + "ln_2.weight"], sd[p + "ln_2.bias"],
                sd[p + "mlp.c_fc.weight"].T, sd[p + "mlp.c_fc.bias"],
                sd[p + "mlp.c_proj.weight"].T, sd[p + "mlp.c_proj.bias"],
            ]
            for t in tensors:
                n_floats += write_tensor(f, t)
        n_floats += write_tensor(f, sd["transformer.ln_f.weight"])
        n_floats += write_tensor(f, sd["transformer.ln_f.bias"])

    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({n_floats:,} params, {size_mb:.1f} MB)")


def export_reference(model: GPT2LMHeadModel, path: Path) -> None:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(REF_PROMPT)
    with torch.no_grad():
        logits = model(torch.tensor([tokens])).logits[0]  # [T, V]

    with open(path, "wb") as f:
        f.write(struct.pack("<I2i", REF_MAGIC, len(tokens), logits.shape[1]))
        f.write(np.asarray(tokens, dtype=np.int32).tobytes())
        f.write(logits.to(torch.float32).numpy().tobytes())

    next_tok = int(logits[-1].argmax())
    print(f"wrote {path} ({len(tokens)} tokens, greedy next token: {enc.decode([next_tok])!r})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id (gpt2, gpt2-medium, ...)")
    ap.add_argument("--out-dir", default=Path(__file__).resolve().parent.parent / "models", type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    name = args.model.split("/")[-1]
    export_weights(model, args.out_dir / f"{name}.bin")
    export_reference(model, args.out_dir / f"{name}_ref.bin")


if __name__ == "__main__":
    main()

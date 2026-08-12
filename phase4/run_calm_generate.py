#!/usr/bin/env python3
"""
Phase 4 — text generation from a real CALM language model.

Loads the autoencoder + a CALM checkpoint (default: CALM-M, 371M) and generates
text using the model's custom continuous autoregressive `generate()` and the
likelihood-free `temperature_sampling` (T must be 1/integer).

Downloads from Hugging Face. CPU-friendly for CALM-M; slow for L/XL.

Usage:
    python phase4/run_calm_generate.py --prompt "Once upon a time" --max_new 48 --temperature 0.5
"""
import os
import sys
import subprocess
import argparse

import torch

CALM_SRC = os.environ.get("CALM_SRC", os.path.join(os.path.dirname(__file__), "..", ".calm_src"))
CALM_SRC = os.path.abspath(CALM_SRC)
if not os.path.isdir(os.path.join(CALM_SRC, "models")):
    print(f"[setup] cloning upstream CALM into {CALM_SRC}")
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/shaochenze/calm.git", CALM_SRC])
sys.path.insert(0, CALM_SRC)

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from models.configuration_calm import CALMConfig
from models.modeling_energy import EnergyTransformer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm_repo", default="cccczshao/CALM-M")
    ap.add_argument("--ae_repo", default="cccczshao/CALM-Autoencoder")
    ap.add_argument("--prompt", default="The key idea behind continuous autoregressive models is")
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="Must be the reciprocal of an integer (1, 1/2, 1/3, ...).")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[dl] autoencoder: {args.ae_repo}")
    ae_dir = snapshot_download(repo_id=args.ae_repo)
    print(f"[dl] calm: {args.calm_repo}")
    calm_dir = snapshot_download(repo_id=args.calm_repo)

    tok = AutoTokenizer.from_pretrained(os.path.join(CALM_SRC, "llama3_tokenizer"))

    cfg = CALMConfig.from_pretrained(calm_dir)
    cfg.ae_path = ae_dir  # point the frozen AE at the local snapshot
    model = EnergyTransformer.from_pretrained(calm_dir, config=cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] K={cfg.patch_size}  hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}  "
          f"params(incl. AE)={n_params:.0f}M")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    max_length = prompt_len + args.max_new
    print(f"\n[gen] prompt={args.prompt!r}  T={args.temperature}  max_new={args.max_new}")

    out = model.generate(ids, max_length=max_length, temperature=args.temperature)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n===== GENERATED =====")
    print(text)
    print("=====================")


if __name__ == "__main__":
    main()

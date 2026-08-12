#!/usr/bin/env python3
"""
Phase 4 — autoencoder round-trip + latent-robustness probe on the REAL CALM weights.

Verifies the paper's load-bearing claim: a chunk of K tokens compresses to one
continuous vector and reconstructs at >99.9% token accuracy, *and* still decodes
correctly when the latent is perturbed (which is what happens at inference, since
the LM only predicts an approximate vector).

Downloads `cccczshao/CALM-Autoencoder` from Hugging Face. CPU-friendly.

Usage:
    python phase4/run_autoencoder.py
"""
import os
import sys
import subprocess
import argparse

import torch

# --- bootstrap the upstream `models/` package (MIT, github.com/shaochenze/calm) ---
CALM_SRC = os.environ.get("CALM_SRC", os.path.join(os.path.dirname(__file__), "..", ".calm_src"))
CALM_SRC = os.path.abspath(CALM_SRC)
if not os.path.isdir(os.path.join(CALM_SRC, "models")):
    print(f"[setup] cloning upstream CALM into {CALM_SRC}")
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/shaochenze/calm.git", CALM_SRC])
sys.path.insert(0, CALM_SRC)

from transformers import AutoTokenizer
from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder
from huggingface_hub import snapshot_download

SAMPLE_TEXT = (
    "The history of language modeling is a history of prediction. For decades, the "
    "dominant paradigm has been to predict the next discrete token given the tokens "
    "that came before it. Continuous Autoregressive Language Models challenge that "
    "assumption by predicting a single continuous vector for an entire chunk of tokens."
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="cccczshao/CALM-Autoencoder")
    ap.add_argument("--text", default=SAMPLE_TEXT)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[dl] snapshot_download({args.repo}) ...")
    ae_dir = snapshot_download(repo_id=args.repo)
    tok = AutoTokenizer.from_pretrained(os.path.join(CALM_SRC, "llama3_tokenizer"))

    cfg = AutoencoderConfig.from_pretrained(ae_dir)
    K = cfg.patch_size
    model = Autoencoder.from_pretrained(ae_dir, config=cfg).to(device).eval()
    print(f"[model] K(patch_size)={K}  latent_size={cfg.latent_size}  "
          f"vocab={cfg.vocab_size}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ids = tok(args.text, return_tensors="pt").input_ids.to(device)
    # trim to a multiple of K
    ids = ids[:, : (ids.shape[1] // K) * K]
    n_tok = ids.shape[1]
    print(f"[input] {n_tok} tokens = {n_tok // K} chunks of K={K}")

    # --- clean round-trip (use posterior mean = deterministic encode) ---
    latent = model.encoder(input_ids=ids)                 # (1, n_chunks, 2*latent)
    mean, log_std = torch.chunk(latent, 2, dim=-1)
    logits = model.decoder(latent_states=mean)            # (1, n_tok, vocab)
    preds = logits.argmax(-1)
    clean_acc = (preds == ids).float().mean().item()
    print(f"\n[clean round-trip] token accuracy = {clean_acc*100:.4f}%  "
          f"({int((preds==ids).sum())}/{n_tok})")

    # --- latent robustness sweep: z = mean + sigma * N(0, I) ---
    print("\n[robustness] decode accuracy vs latent Gaussian perturbation:")
    print(f"    {'sigma':>8} | {'token acc %':>11}")
    print("    " + "-" * 24)
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        eps = torch.randn_like(mean)
        z = mean + sigma * eps
        acc = (model.decoder(latent_states=z).argmax(-1) == ids).float().mean().item()
        print(f"    {sigma:>8.2f} | {acc*100:>11.4f}")

    print("\n[note] The paper's premise is that this curve stays near 100% for the range "
          "of sigma the LM's prediction error actually occupies. That is the exact\n"
          "       fragility Phase 3 should map across K, latent_size, and the AE dropout rates.")


if __name__ == "__main__":
    main()

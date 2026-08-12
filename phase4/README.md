# Phase 4 — run the real CALM checkpoints

Runs the authors' released weights (Hugging Face `cccczshao/CALM-*`) to verify two
things first-hand: the autoencoder's near-perfect, perturbation-robust reconstruction,
and end-to-end generation with the likelihood-free sampler.

## Requirements

- **Network egress to Hugging Face.** This environment blocks it by default; set the
  cloud environment's Network access to **Custom** and allow `huggingface.co`,
  `*.huggingface.co`, `*.hf.co` (keep the default package-manager list on). See the
  repo root chat/notes for the click-path.
- **Python deps** (CPU is fine; no GPU / no flash-attn needed — the model falls back to
  SDPA attention):
  ```bash
  pip install "torch==2.2.2" "transformers==4.43.0" "tokenizers==0.19.1" \
              safetensors "accelerate==0.30.1" huggingface-hub "numpy<2"
  ```
  `download.pytorch.org` is also blocked here, so torch comes from PyPI (a CUDA-bundled
  wheel that still runs on CPU).

The scripts self-bootstrap the upstream `models/` package by shallow-cloning
`github.com/shaochenze/calm` into `./.calm_src` (override with `CALM_SRC=/path`).

## Scripts

### `run_autoencoder.py` — the load-bearing claim
Downloads `CALM-Autoencoder` (75M), does a clean encode→decode round-trip on sample
text (expect ~100% token accuracy), then sweeps Gaussian noise `σ` injected into the
latent and prints decode accuracy vs `σ`. That robustness curve is the crux of the
whole method and the entry point for Phase 3.
```bash
python phase4/run_autoencoder.py
```

### `run_calm_generate.py` — end-to-end generation
Loads the autoencoder + a CALM LM (default `CALM-M`, 371M) and generates text via the
model's custom continuous-AR `generate()` and `temperature_sampling` (T must be `1/n`).
```bash
python phase4/run_calm_generate.py --prompt "Once upon a time" --max_new 48 --temperature 0.5
# heavier: --calm_repo cccczshao/CALM-L  (735M)  /  cccczshao/CALM-XL (1.82B, slow on CPU)
```

## Notes
- Checkpoints total ~5–6 GB; ensure disk headroom.
- CALM-M generation on CPU is fine for short samples; L/XL get slow.
- Results (accuracy numbers, sample text) are recorded in `RESULTS.md` after a run.

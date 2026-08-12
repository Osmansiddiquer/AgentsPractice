# CALM — paper digest

**Continuous Autoregressive Language Models** — Chenze Shao, Darren Li, Fandong Meng, Jie Zhou
(WeChat AI / Tencent · Tsinghua). arXiv:2510.27688, Oct 2025. Code: github.com/shaochenze/calm (MIT).

## The idea in one line
Stop predicting one discrete token per step. Compress every **K** tokens into a single
**continuous vector**, and have the LM autoregressively predict the *next vector*. That cuts the
number of generative steps by K → the main lever on inference cost.

## The two pieces
1. **Autoencoder (the "tokenizer" for vectors).** Encodes a chunk of K tokens → one latent
   vector `z`; decoder reconstructs the K tokens from `z` at **>99.9%** accuracy.
   - Robustness so the LM's imperfect predictions still decode: latent **dropout (0.15)** +
     VAE-style regularization → a redundant, error-tolerant latent space.
   - Released autoencoder: ~75M params.

2. **Continuous-domain LM.** A Transformer whose output feeds a small **generative head**
   (residual MLP + SwiGLU) that fuses the hidden state with a **noise vector** to *sample* the
   next latent — not regress a point estimate.

## Why it's not just "regression on embeddings"
There is no softmax over a vocab, so **no maximum-likelihood / cross-entropy**. Training is
**likelihood-free**:
- **Energy loss** (Monte Carlo energy score) is the headline objective; the repo also ships
  **diffusion** and **flow-matching** heads as alternatives.
- Sampling supports **temperature** in the likelihood-free setting (batch/rejection approximation).

## Evaluation: BrierLM
New **likelihood-free, calibrated** metric (Brier-score based) since perplexity isn't available.
Correlates strongly with cross-entropy (Pearson ρ ≈ **−0.966**), so it's a usable stand-in.

## Results (released checkpoints)
| Model | Params | BrierLM |
|-------|--------|---------|
| Autoencoder | 75M | — |
| CALM-M | 371M | 5.72 |
| CALM-L | 735M | 6.58 |
| CALM-XL | 1.82B | 8.53 |

- Reported **~40%+ fewer FLOPs** vs standard Transformer baselines at matched performance;
  better performance–compute frontier overall.
- Data: **Pile-uncopyrighted** (~2.5TB). AE trained on ~15B tokens; LM on the remainder.
  Validation: WikiText document-level.

## Honest caveats
- Full replication is **not laptop-scale** (multi-B-token training, 2.5TB data).
- The autoencoder is the crux: if `z` isn't robust, LM errors cascade into garbled decodes.
- Likelihood-free eval means the whole community's perplexity intuition doesn't transfer;
  BrierLM is new and unproven outside this paper.

## Sources
- https://arxiv.org/abs/2510.27688
- https://github.com/shaochenze/calm

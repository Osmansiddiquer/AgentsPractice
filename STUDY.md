# CALM — a code-grounded study

*Continuous Autoregressive Language Models*, Shao, Li, Meng, Zhou (WeChat AI · Tsinghua),
[arXiv:2510.27688](https://arxiv.org/abs/2510.27688), Oct 2025. Official code (MIT):
[github.com/shaochenze/calm](https://github.com/shaochenze/calm).

This is a study written against the *actual source*, not the abstract. Line references are to
the upstream repo as of the clone in this project's Phase 1.

---

## 1. The thesis

Autoregressive LLMs are bottlenecked by **one discrete token per forward pass**. CALM raises the
**semantic bandwidth** of a step: compress every **K** tokens into one continuous vector, and make
the LM predict *the next vector*. That is K× fewer autoregressive steps — the dominant cost at
inference. The claim is a better **performance-per-FLOP frontier**, not a new SOTA on absolute
quality.

The price of leaving the discrete world: you lose the softmax, and with it **maximum-likelihood
training** and **perplexity evaluation**. Most of CALM's machinery exists to buy those two things
back in a continuous, likelihood-free form. That is the real intellectual content.

---

## 2. Stage 1 — the autoencoder (`models/modeling_autoencoder.py`)

A **VAE** over token chunks. It is the load-bearing component: if the latent isn't robust, the LM's
imperfect vector predictions decode to garbage.

**Encoder.** K token embeddings → 2 Llama-MLP residual layers → a `squeeze_layer` that maps
`K·hidden → hidden` (this is where K tokens collapse to one position) → 2 more layers →
`hidden_to_latent` producing `2·latent_size` = a **mean and log-std** (`latent_size=128`).

**Decoder.** latent → hidden → 2 layers → `expand_layer` maps `hidden → K·hidden` (un-squeeze back
to K positions) → 2 layers → linear to vocab logits (weight-tied to the encoder embedding).

**Why reconstruction hits >99.9% *and* survives LM error** — three deliberate regularizers
(`AutoencoderConfig`, forward pass):
- **Input token dropout** (`ae_dropout=0.15`): randomly zeroes input token ids before encoding, so
  the latent can't rely on any single token being present.
- **Latent dropout** (`F.dropout(..., 0.15)` on the sampled latent): the decoder must tolerate a
  perturbed vector — exactly the situation at inference, where the LM's predicted vector is an
  approximation.
- **KL regularization with free bits** (`kl_weight=1e-3`, `kl_clamp=0.5`): a light VAE prior keeps
  the latent space smooth/continuous (so nearby vectors decode to similar text) without over-
  compressing. `kl_clamp` is a per-dim floor (free-bits) that stops posterior collapse.
- Reconstruction loss is scaled by `patch_size` so it stays commensurate with the KL term.

> **This is the whole ballgame.** The headline ">99.9% reconstruction under perturbation" is what
> makes next-vector prediction viable at all. Phase 3 of this project should attack exactly this:
> *how does decode accuracy fall as you inject noise into `z`, and how does that curve change with
> K, latent_size, and the dropout rates?*

---

## 3. Stage 2 — the continuous LM (`models/modeling_energy.py`)

A standard **LlamaModel** backbone, but both ends are continuous:
- **Input:** K token embeddings per step are concatenated and pushed through `embed_proj`
  (`K·hidden → 2·hidden → hidden → LayerNorm`) to form one input vector per patch.
- **Output head — `MLPGenerator`:** *not* a regressor. It takes the transformer hidden state **plus
  a fresh uniform noise vector** (`noise_size=64`, drawn in `[-0.5, 0.5]`) and pushes the noise
  through `num_mlp_layers=4` gated residual `MLPBlock`s conditioned on the hidden state, emitting a
  latent-dim vector. Different noise → different sample. **The head defines an implicit conditional
  distribution you can sample from but whose density you cannot evaluate.** That is what "black-box /
  likelihood-free generator" means here.

### The energy loss (the MLE replacement)

The AE encoder gives the *true* next chunk's posterior `N(mean, std)`. Training matches the head's
sample distribution to it using the **energy score**, a strictly proper scoring rule
(`energy_score`, `distance = ‖·‖₂^β`, `β=1`):

```
loss = 2·E‖x − y‖  −  E‖x − x'‖
```

- `x, x'` = independent model samples (`num_samples=8`), `y` = samples from the target posterior
  (they draw `n_y=100`).
- First term pulls samples **toward** the target; second term is a **repulsion** that preserves
  spread and prevents mode collapse to the mean. Being *proper* means the loss is minimized exactly
  when the model's distribution equals the target's — the continuous analogue of what cross-entropy
  does for a categorical.

The repo also ships **diffusion** (`modeling_diffusion.py`) and **flow-matching**
(`modeling_flow.py`) heads as drop-in alternatives; the README reports both slightly below the
energy head. That three-way choice is a clean, cheap ablation target.

---

## 4. Evaluation — BrierLM (`CALM.eval_brier`, `train/train_calm.py`)

No likelihood ⇒ no perplexity. They use the **Brier score**, which *is* estimable from samples
alone via the identity (two independent model samples `x1, x2`, target `y`):

```
Brier ≈ E[ 1{x1=y} + 1{x2=y} − 1{x1=x2} ]
```

Every term is an equality indicator on **decoded token ids** — no density needed. They compute it at
n-gram orders 1–4 (`brier1..4`) using cumulative-product prefix matches, then aggregate like BLEU:

```
BrierLM = ( brier1 · brier2 · brier3 · brier4 )^(1/4)     # geometric mean; reported ×100
```

The `1{x1=x2}` collision term needs autoregressive rollout when `K<4` (the messy loop in
`eval_brier`). The paper reports BrierLM correlates with cross-entropy at **ρ ≈ −0.966**, which is
the evidence that this sample-only metric is a faithful stand-in. **That correlation claim is
independently checkable at small scale** and is a good Phase-3 probe.

---

## 5. Likelihood-free temperature sampling (`CALM.temperature_sampling`)

A genuinely elegant trick. You can't compute `p^{1/T}`, but you can *sample* from it with only a
black-box sampler, for `T = 1/n` (n integer):

1. Draw `N` patches from the head (default `num_samples=200`), decode each to token ids.
2. Count duplicates. Select a patch that appeared **≥ n** times, weighted by `C(count, n)`.
3. Cascade `n → n−1 → …` until something is selectable.

Drawing n samples and keeping them only if they *collide* is an unbiased sampler for `p^n`; the
combinatorial weighting is the batched version. `T=1` is just one draw. It's the sampling-time
counterpart to BrierLM: both extract calibrated behavior from equality tests on samples.

---

## 6. Numbers that matter (defaults + released checkpoints)

| Thing | Value | Source |
|---|---|---|
| Chunk size K (`patch_size`) | **4** | configs / scripts |
| Latent dim | **128** | scripts |
| Energy: model samples / target samples | **8 / 100** | `energy_score` |
| Noise dim | 64 | `CALMConfig` |
| AE dropout (input & latent) | 0.15 | `AutoencoderConfig` |
| Autoencoder | 75M | README / HF |
| CALM-M / L / XL | 371M / 735M / 1.82B | README |
| BrierLM M / L / XL | 5.72 / 6.58 / 8.53 | README |
| **Token AR baseline** | ~124M-config | BrierLM **6.05** |
| Data | Pile-uncopyrighted (~2.5TB) | `data/get_data.sh` |
| Stack | HF Transformers **4.43**, flash-attn 2.1.1, deepspeed | `requirements.txt` |

> **Read the baseline carefully.** The small token AR baseline scores **6.05**, *above* CALM-M's
> **5.72**. Higher BrierLM = better, so at these particular sizes the discrete baseline is not
> strictly beaten on quality — CALM's win is on the **compute–performance curve** (4× fewer steps),
> which is precisely the paper's claim. Anyone "replicating for SOTA" will be disappointed; the
> result is an efficiency frontier, and should be reported as one.

---

## 7. Critique / open questions (what Phase 3 should test)

1. **Latent robustness is the crux, so measure it directly.** Perturbation-vs-decode-accuracy
   curves as a function of noise magnitude, K, latent_size, and the two dropout knobs. If the curve
   is steep, the method is fragile in a way the 99.9% headline hides.
2. **Is BrierLM trustworthy?** Re-derive the estimator; re-check the −0.966 correlation on a toy
   setup where cross-entropy *is* available. Probe its variance (it's a sample estimate with only 2
   samples for the collision term).
3. **Head bake-off.** Energy vs diffusion vs flow, controlled, small. The repo makes this a
   one-flag change.
4. **Error accumulation.** Continuous AR has no discrete "snap-to-token" per step during the vector
   rollout — do small per-vector errors compound over long generations? Decoded-text quality vs
   sequence length.
5. **K sensitivity.** Everything ships at K=4. What does the reconstruction/quality/efficiency
   trade look like at K=2 and K=8? The `squeeze_layer`/`expand_layer` scale with K·hidden.
6. **Semantic bandwidth as a scaling axis** is the paper's boldest framing. Is the frontier smooth
   in K, or does it break once a chunk exceeds what one vector can faithfully carry?

---

## 8. Feasibility notes for later phases

- **Full replication (Pile, multi-B tokens, 8×GPU scripts): out of scope here.** The training
  scripts assume `nproc_per_node 8` and 2.5TB of data.
- **Phase 4 (run their checkpoints):** checkpoints live on Hugging Face. This sandbox's egress
  proxy currently **blocks huggingface.co**, so pulling weights may need a whitelist or a manual
  upload. Flag before committing to it.
- **Phase 3 (build on it):** the *tractable* science is the **autoencoder in isolation** — it's
  75M, trainable on a small corpus, and hosts the paper's most important and most checkable claim.
  That is the recommended entry point.

---

*Sources:* [paper](https://arxiv.org/abs/2510.27688) ·
[code](https://github.com/shaochenze/calm) ·
[blog](https://shaochenze.github.io/blog/2025/CALM) · HF collection `cccczshao/calm`.

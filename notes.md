# Architecture Test Bench

We built a test harness for sequence architectures: same data, same tokenizer, same optimizer, same eval. Swap the model, compare bpb.

## Current model (FORA/Rechat)

Diagonal linear recurrence with input-dependent decay (decoupled minGRU-style):

```
decay = sigmoid(proj_decay(x))   # input-dependent, per-token
gate  = sigmoid(proj_gate(x))
inp   = proj_in(x)
h[t]  = decay[t] * h[t-1] + inp * gate
out   = proj_out(h)
```

Parallel training via chunk-based scan with cumulative product (log-sum-exp trick).

## Results (TinyStories, 5000 steps, seq_len=512, batch=4096)

### Quality (val_bpb, lower is better)

| Model | Params | val_bpb | Notes |
|---|---|---|---|
| GPT d4 (attention) | 129M | 0.627 | Baseline with value embeddings |
| **Rechat d11 (input-dep decay)** | **128M** | **0.646** | Nearly matches GPT at equal params |
| Rechat d4 (input-dep decay) | ~79M | 0.761 | No value embeddings → fewer params |
| Rechat d4 (fixed decay) | ~79M | 0.981 | Original fixed decay baseline |
| Hydra v1 (all grouped) | varied | 1.322 | Failed — too few effective params |
| Hydra v2 (grouped input, full MLP) | varied | 1.317 | Failed — grouped input bottleneck |
| Hydra v3 (conv extend-collapse) | varied | 1.332 | Failed — can't shortcut linear projections |

Key finding: input-dependent decay was the single biggest improvement (+0.22 bpb over fixed decay).

### Inference speed (tok/s, generating 50 tokens)

| Prompt len | GPT d4 | Rechat d4 | Rechat d11 | R4/GPT | R11/GPT |
|---|---|---|---|---|---|
| 16 | 229 | 357 | 205 | 1.56x | 0.89x |
| 64 | 230 | 353 | 201 | 1.54x | 0.87x |
| 128 | 184 | 358 | 196 | 1.95x | 1.07x |
| 256 | 173 | 362 | 202 | 2.10x | 1.17x |
| 512 | 86 | 342 | 198 | 3.96x | 2.29x |

Key finding: Rechat inference is **constant time** regardless of prompt length (O(1) per token via state carry). GPT slows linearly. At 512 tokens, Rechat d11 is 2.3x faster at equal param count.

### Training speed

| Model | tok/sec | Time (5000 steps) |
|---|---|---|
| GPT d4 | ~185K | ~2m |
| Rechat d4 | ~185K | ~2m |
| Rechat d11 | ~22K | ~15m |

Rechat d11 is slower to train due to 11 sequential scan layers vs GPT's 4 attention layers. At equal depth (d4) training speed is identical.

## Param count breakdown (depth=4, n_embd=768)

GPT has 50M extra params from **value embeddings** (2x full embedding tables). Rechat has no equivalent:

| Component | GPT | Rechat |
|---|---|---|
| wte | 25.2M | 25.2M |
| lm_head | 25.2M | 25.2M |
| transformer blocks | 28.3M | 28.3M |
| value_embeds | 50.3M | 0 |
| **Total** | **129M** | **79M** |

To match params fairly: Rechat d11 (128M) ≈ GPT d4 (129M).

## Recurrence variants to test

1. ~~**Fixed decay, separate gate**~~ — baseline, 0.981 bpb (superseded)
2. ~~**Input-dependent decay**~~ — done, 0.761 bpb (d4), 0.646 bpb (d11)
3. **Decoupled minGRU** — two input-dependent gates (forget + input). Most expressive diagonal recurrence.
4. **Multi-head recurrence** — split dim into 4-8 heads, each with its own decay dynamics.
5. **Conv + recurrence** — short 1D conv before recurrence (alongside full linears, not replacing them).

## Wilder ideas

6. **Hybrid** — first N layers recurrence, last M layers attention. O(1) compression early, precise retrieval late.
7. **Linear attention** — reformulate attention as `h[t] = h[t-1] + k[t] * v[t]^T`, query at read time.
8. **Bidirectional recurrence in MLP** — forward scan + backward scan, concat, then project. Training only.
9. **Stacked recurrence** — two recurrence steps per block instead of one.
10. **Cross-channel recurrence** — small matrix A instead of diagonal. More expressive but heavier.

## How to test

Each variant is a new model file (e.g. `nanochat/rechat_v2.py`) and a training script copy. Compare val_bpb in a table.

Train on TinyStories first, then scale the winners to FineWeb on the A100.

## Comparison to existing architectures

| | FORA/Rechat (current) | minGRU | Mamba | RWKV-7 |
|---|---|---|---|---|
| Decay | Input-dependent | Input-dependent (1-z) | Input-dependent | Input-dependent |
| Input gate | sigmoid(proj_gate(x)) | Same gate as decay (z) | Input-dependent | Input-dependent |
| Gate coupling | Independent | Tied (sum to 1) | Independent | Independent |
| Candidate | proj_in(x) | Linear(x) | Linear(x) | Complex gating |
| State dims | Diagonal (per-dim) | Diagonal (per-dim) | Multi-dim per head | Matrix-valued |
| Parallel training | Chunk-based scan | Parallel scan | Selective scan | Custom CUDA |

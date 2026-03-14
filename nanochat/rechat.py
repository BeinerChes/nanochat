"""
Rechat model — diagonal linear recurrence replacing attention.

Drop-in replacement for GPT with the same interface:
- Same forward(idx, targets) signature
- Same init_weights() convention
- Same setup_optimizer() returning MuonAdamW
- Same num_scaling_params(), estimate_flops()
- MLP, norm, Linear, logit softcap all identical to nanochat GPT

The only difference: CausalSelfAttention → DiagRecurrence
No positional embeddings needed (recurrence is inherently sequential).
No sliding window, no rotary, no KV cache, no value embeddings.

Decay is INPUT-DEPENDENT: each token decides how much to remember/forget.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class RechatConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    chunk_size: int = 128  # parallel scan chunk size


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# ---------------------------------------------------------------------------
# Diagonal Linear Recurrence (replaces CausalSelfAttention)
# ---------------------------------------------------------------------------

def parallel_scan_chunked(decay, x, chunk_size=128):
    """
    Diagonal linear recurrence: h[t] = decay[t] * h[t-1] + x[t]

    Chunk-based: parallel within chunks (cumsum trick), sequential across.

    decay: (batch, seq_len, dim) in [0, 1] — input-dependent per token
    x: (batch, seq_len, dim)
    Returns: (batch, seq_len, dim)
    """
    B, T, D = x.shape

    h = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
    h_carry = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        x_chunk = x[:, start:end, :]
        decay_chunk = decay[:, start:end, :]  # (B, chunk_len, D)

        # Cumulative product of decays via log-sum-exp
        log_a = torch.log(decay_chunk.clamp(min=1e-6))  # (B, chunk_len, D)
        log_cum_a = torch.cumsum(log_a, dim=1)           # (B, chunk_len, D)
        cum_a = torch.exp(log_cum_a)                      # (B, chunk_len, D)
        inv_cum_a = 1.0 / (cum_a + 1e-10)                # (B, chunk_len, D)

        # h[t] = cum_a[t] * (h_carry + cumsum(x / cum_a)[t])
        scaled = x_chunk * inv_cum_a
        cumulative = torch.cumsum(scaled, dim=1)
        h_chunk = cum_a * (h_carry.unsqueeze(1) + cumulative)

        h[:, start:end, :] = h_chunk
        h_carry = h_chunk[:, -1, :]

    return h


class DiagRecurrence(nn.Module):
    """Diagonal linear recurrence layer with input-dependent decay."""

    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.chunk_size = config.chunk_size
        self.proj_decay = Linear(dim, dim, bias=False)  # input-dependent decay
        self.proj_in = Linear(dim, dim, bias=False)
        self.proj_gate = Linear(dim, dim, bias=False)
        self.proj_out = Linear(dim, dim, bias=False)

    def forward(self, x, return_last_state=False):
        decay = torch.sigmoid(self.proj_decay(x))  # (B, T, dim) — per-token
        inp = self.proj_in(x)
        gate = torch.sigmoid(self.proj_gate(x))
        gated = inp * gate
        h = parallel_scan_chunked(decay, gated, self.chunk_size)
        out = self.proj_out(h)
        if return_last_state:
            return out, h[:, -1, :]  # return final hidden state for step() continuation
        return out

    def step(self, x, h_prev):
        """Single-step for autoregressive inference. O(1) per token."""
        decay = torch.sigmoid(self.proj_decay(x))  # (1, dim)
        inp = self.proj_in(x)
        gate = torch.sigmoid(self.proj_gate(x))
        h_new = decay * h_prev + inp * gate
        return h_new, self.proj_out(h_new)


# ---------------------------------------------------------------------------
# MLP (identical to nanochat GPT)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Block = Recurrence + MLP
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rec = DiagRecurrence(config)
        self.mlp = MLP(config)

    def forward(self, x, return_last_state=False):
        if return_last_state:
            rec_out, h_last = self.rec(norm(x), return_last_state=True)
            x = x + rec_out
            x = x + self.mlp(norm(x))
            return x, h_last
        x = x + self.rec(norm(x))
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, h_prev):
        """Single-step inference."""
        h_new, rec_out = self.rec.step(norm(x), h_prev)
        x = x + rec_out
        x = x + self.mlp(norm(x))
        return h_new, x


# ---------------------------------------------------------------------------
# Full model — same interface as nanochat GPT
# ---------------------------------------------------------------------------

class Rechat(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Match nanochat's structure: transformer.wte, transformer.h, lm_head
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding and unembedding (same as nanochat GPT)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            # Recurrence
            torch.nn.init.uniform_(block.rec.proj_decay.weight, -s, s)
            torch.nn.init.uniform_(block.rec.proj_in.weight, -s, s)
            torch.nn.init.uniform_(block.rec.proj_gate.weight, -s, s)
            torch.nn.init.zeros_(block.rec.proj_out.weight)

            # MLP (same as nanochat GPT)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Cast embeddings to COMPUTE_DTYPE (same as nanochat GPT)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """FLOPs per token (forward + backward). No attention FLOPs!"""
        nparams = sum(p.numel() for p in self.parameters())
        embed_params = self.transformer.wte.weight.numel()
        matmul_params = nparams - embed_params
        return 6 * matmul_params

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': 0,  # rechat has no value embeddings
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': 0,  # no more fixed decay scalars
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # All block params are now 2D matrices → Muon
        matrix_params = []
        for block in self.transformer.h:
            for name, p in block.named_parameters():
                assert p.dim() >= 2, f"Unexpected 1D param: {name}"
                matrix_params.append(p)

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        total_params = len(matrix_params) + len(embedding_params) + len(lm_head_params)
        assert total_params == len(list(self.parameters())), f"Parameter count mismatch: {total_params} vs {len(list(self.parameters()))}"

        # Scale LR by model dim (same as nanochat GPT)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            # AdamW groups
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]
        # Muon groups (all matrix params, grouped by shape)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # kv_cache is accepted but ignored (recurrence doesn't use KV cache)
        B, T = idx.size()

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        for block in self.transformer.h:
            x = block(x)

        x = norm(x)

        # Logit soft-capping (same as nanochat GPT)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive generation: parallel prefill, then O(1) per token."""
        assert isinstance(tokens, list)
        device = self.get_device()

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Parallel prefill — run full forward pass to get hidden states
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        x = self.transformer.wte(ids)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        h_states = []
        for block in self.transformer.h:
            x, h_last = block(x, return_last_state=True)
            h_states.append(h_last)
        # x is (1, T, dim) — take last position, squeeze to (1, dim) for step mode
        x = x[:, -1, :]  # (1, dim)

        # Generate tokens (O(1) per token)
        for _ in range(max_tokens):
            out = norm(x)
            logits = self.lm_head(out)
            logits = logits[..., :self.config.vocab_size].float()
            logits = 15 * torch.tanh(logits / 15)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng).item()
            else:
                next_id = logits.argmax(dim=-1).item()

            yield next_id

            tok_emb = self.transformer.wte(torch.tensor([next_id], device=device))
            x = norm(tok_emb)
            new_h = []
            for i, block in enumerate(self.transformer.h):
                h_new, x = block.step(x, h_states[i])
                new_h.append(h_new)
            h_states = new_h

"""
Classic minGRU — "Were RNNs All We Needed?" (Feng et al., 2024)

Drop-in replacement with same interface as Rechat/GPT:
- Same forward(idx, targets) signature
- Same init_weights(), setup_optimizer(), num_scaling_params(), estimate_flops()

minGRU recurrence:
    z[t] = sigmoid(Linear_z(x[t]))           # gate (input-dependent)
    h_tilde[t] = Linear_h(x[t])              # candidate
    h[t] = (1 - z[t]) * h[t-1] + z[t] * h_tilde[t]   # tied gates (sum to 1)

vs Rechat (decoupled):
    decay[t] = sigmoid(proj_decay(x[t]))      # forget gate
    gate[t]  = sigmoid(proj_gate(x[t]))        # input gate (independent)
    h[t] = decay[t] * h[t-1] + proj_in(x[t]) * gate[t]
    out  = proj_out(h[t])                      # output projection

Key differences: minGRU has 2 projections (tied gates, no output proj) vs Rechat's 4.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class MinGRUConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    chunk_size: int = 128


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# ---------------------------------------------------------------------------
# Parallel scan (same as Rechat — works for any input-dependent decay)
# ---------------------------------------------------------------------------

def parallel_scan_chunked(decay, x, chunk_size=128):
    """
    Diagonal linear recurrence: h[t] = decay[t] * h[t-1] + x[t]
    decay: (batch, seq_len, dim) in [0, 1]
    x: (batch, seq_len, dim)
    Returns: (batch, seq_len, dim)
    """
    B, T, D = x.shape
    h = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
    h_carry = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        x_chunk = x[:, start:end, :]
        decay_chunk = decay[:, start:end, :]

        log_a = torch.log(decay_chunk.clamp(min=1e-6))
        log_cum_a = torch.cumsum(log_a, dim=1)
        cum_a = torch.exp(log_cum_a)
        inv_cum_a = 1.0 / (cum_a + 1e-10)

        scaled = x_chunk * inv_cum_a
        cumulative = torch.cumsum(scaled, dim=1)
        h_chunk = cum_a * (h_carry.unsqueeze(1) + cumulative)

        h[:, start:end, :] = h_chunk
        h_carry = h_chunk[:, -1, :]

    return h


# ---------------------------------------------------------------------------
# minGRU recurrence
# ---------------------------------------------------------------------------

class MinGRURecurrence(nn.Module):
    """Classic minGRU: z and (1-z) tied gates, no output projection."""

    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.chunk_size = config.chunk_size
        self.proj_z = Linear(dim, dim, bias=False)       # gate
        self.proj_h_tilde = Linear(dim, dim, bias=False)  # candidate

    def forward(self, x, return_last_state=False):
        z = torch.sigmoid(self.proj_z(x))        # (B, T, dim)
        h_tilde = self.proj_h_tilde(x)            # (B, T, dim)
        # h[t] = (1-z[t]) * h[t-1] + z[t] * h_tilde[t]
        decay = 1.0 - z       # forget factor
        inp = z * h_tilde      # gated input
        h = parallel_scan_chunked(decay, inp, self.chunk_size)
        if return_last_state:
            return h, h[:, -1, :]
        return h

    def step(self, x, h_prev):
        """Single-step for autoregressive inference. O(1) per token."""
        z = torch.sigmoid(self.proj_z(x))
        h_tilde = self.proj_h_tilde(x)
        h_new = (1.0 - z) * h_prev + z * h_tilde
        return h_new, h_new  # output IS the hidden state (no proj_out)


# ---------------------------------------------------------------------------
# MLP (identical to nanochat GPT / Rechat)
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
# Block = minGRU + MLP
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rec = MinGRURecurrence(config)
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
# Full model — same interface as Rechat / GPT
# ---------------------------------------------------------------------------

class MinGRU(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            # minGRU recurrence
            torch.nn.init.uniform_(block.rec.proj_z.weight, -s, s)
            torch.nn.init.uniform_(block.rec.proj_h_tilde.weight, -s, s)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
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
            'value_embeds': 0,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': 0,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = []
        for block in self.transformer.h:
            for name, p in block.named_parameters():
                assert p.dim() >= 2, f"Unexpected 1D param: {name}"
                matrix_params.append(p)

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        total_params = len(matrix_params) + len(embedding_params) + len(lm_head_params)
        assert total_params == len(list(self.parameters())), f"Parameter count mismatch: {total_params} vs {len(list(self.parameters()))}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        ]
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
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        for block in self.transformer.h:
            x = block(x)

        x = norm(x)

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

        # Parallel prefill
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        x = self.transformer.wte(ids)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        h_states = []
        for block in self.transformer.h:
            x, h_last = block(x, return_last_state=True)
            h_states.append(h_last)
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

"""
Hydra model — multi-head conv1d recurrence.

Drop-in replacement for Rechat/GPT with the same interface.

Key idea: replace Linear projections in the recurrence with multiple
parallel (conv1d_extend → SiLU → conv1d_collapse) paths. Each path
captures local patterns through its conv kernel while expanding and
contracting channels. A separate gate conv controls what enters memory.

Architecture per block:
  1. RMSNorm
  2. Causal Conv1d extend (dim → g * mid_dim, kernel=k) — local context + expand
  3. SiLU
  4. Grouped Conv1d collapse (g * mid_dim → dim, groups=g) — contract per head
  5. Causal Conv1d gate (dim → dim, kernel=k) → sigmoid
  6. Parallel scan (diagonal recurrence)
  7. Linear proj_out (full rank)
  8. RMSNorm
  9. MLP (full rank, same as Rechat)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class HydraConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    chunk_size: int = 128
    num_groups: int = 8    # number of parallel conv heads
    kernel_size: int = 3   # conv kernel size


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# ---------------------------------------------------------------------------
# Parallel scan (identical to Rechat)
# ---------------------------------------------------------------------------

def parallel_scan_chunked(decay, x, chunk_size=128):
    """
    Diagonal linear recurrence: h[t] = decay * h[t-1] + x[t]
    Chunk-based: parallel within chunks (cumsum trick), sequential across.
    """
    B, T, D = x.shape

    positions = torch.arange(chunk_size, device=x.device, dtype=x.dtype)
    log_a = torch.log(decay.clamp(min=1e-6))
    log_powers = positions.unsqueeze(-1) * log_a.unsqueeze(0)
    powers = torch.exp(log_powers)

    h = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
    h_carry = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_len = end - start
        x_chunk = x[:, start:end, :]

        p = powers[:chunk_len]
        inv_p = 1.0 / (p + 1e-10)

        scaled = x_chunk * inv_p.unsqueeze(0)
        cumulative = torch.cumsum(scaled, dim=1)
        h_chunk = cumulative * p.unsqueeze(0)

        carry_contrib = h_carry.unsqueeze(1) * (decay.unsqueeze(0) * p).unsqueeze(0)
        h_chunk = h_chunk + carry_contrib

        h[:, start:end, :] = h_chunk
        h_carry = h_chunk[:, -1, :]

    return h


# ---------------------------------------------------------------------------
# Hydra Recurrence — conv1d extend/collapse + scan
# ---------------------------------------------------------------------------

class HydraRecurrence(nn.Module):
    """
    Multiple parallel (extend → SiLU → collapse) conv1d paths replace Linear.
    Each head captures local context through causal conv kernels.
    Gate conv controls what enters recurrence memory.
    Full-rank Linear for output projection.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        g = config.num_groups
        k = config.kernel_size
        self.kernel_size = k
        self.chunk_size = config.chunk_size
        self.num_groups = g

        head_dim = dim // g
        mid_dim = head_dim * 2  # expand ratio = 2
        total_mid = g * mid_dim

        # Input path: g parallel (extend → collapse) conv pairs
        # Extend: all dims interact, causal conv captures local context
        self.inp_extend = nn.Conv1d(dim, total_mid, k, bias=False)
        # Collapse: each head contracts independently
        self.inp_collapse = nn.Conv1d(total_mid, dim, 1, groups=g, bias=False)

        # Gate path: causal conv → sigmoid
        self.gate_conv = nn.Conv1d(dim, dim, k, bias=False)

        # Per-dim decay (same as Rechat)
        self.decay_raw = nn.Parameter(torch.zeros(dim))

        # Full-rank output projection
        self.proj_out = Linear(dim, dim, bias=False)

    def _conv1d(self, x, conv):
        """Conv1d with weight cast to match input dtype."""
        return F.conv1d(x, conv.weight.to(dtype=x.dtype), groups=conv.groups)

    def forward(self, x):
        B, T, D = x.shape
        # Causal padding: pad left by (kernel_size - 1)
        xt = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))

        # Input: extend → SiLU → collapse
        inp = self._conv1d(xt, self.inp_extend)
        inp = F.silu(inp)
        inp = self._conv1d(inp, self.inp_collapse).transpose(1, 2)  # (B, T, dim)

        # Gate: conv → sigmoid
        gate = torch.sigmoid(self._conv1d(xt, self.gate_conv).transpose(1, 2))  # (B, T, dim)

        # Scan
        decay = torch.sigmoid(self.decay_raw)
        h = parallel_scan_chunked(decay, inp * gate, self.chunk_size)
        return self.proj_out(h)

    def step(self, x, h_prev, conv_state):
        """Single-step inference with conv state buffer."""
        # x: (1, dim), conv_state: (1, dim, kernel_size - 1)
        x_col = x.unsqueeze(-1)  # (1, dim, 1)
        conv_input = torch.cat([conv_state, x_col], dim=-1)  # (1, dim, k)

        # Input: extend → SiLU → collapse
        inp = F.conv1d(conv_input, self.inp_extend.weight.to(dtype=x.dtype))
        inp = F.silu(inp)
        inp = F.conv1d(inp, self.inp_collapse.weight.to(dtype=x.dtype),
                       groups=self.inp_collapse.groups).squeeze(-1)  # (1, dim)

        # Gate
        gate = torch.sigmoid(
            F.conv1d(conv_input, self.gate_conv.weight.to(dtype=x.dtype)).squeeze(-1)
        )  # (1, dim)

        # Recurrence
        decay = torch.sigmoid(self.decay_raw)
        h_new = decay * h_prev + inp * gate

        # Update conv state: drop oldest, keep newest
        new_conv_state = conv_input[:, :, 1:]  # (1, dim, k-1)

        return h_new, self.proj_out(h_new), new_conv_state


# ---------------------------------------------------------------------------
# MLP (full rank — same as Rechat)
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
# Block = HydraRecurrence + MLP
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rec = HydraRecurrence(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.rec(norm(x))
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, h_prev, conv_state):
        """Single-step inference with recurrence + conv states."""
        h_new, rec_out, new_conv_state = self.rec.step(norm(x), h_prev, conv_state)
        x = x + rec_out
        x = x + self.mlp(norm(x))
        return h_new, x, new_conv_state


# ---------------------------------------------------------------------------
# Full Hydra model — same interface as Rechat/GPT
# ---------------------------------------------------------------------------

class Hydra(nn.Module):
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
            # Conv input path
            torch.nn.init.uniform_(block.rec.inp_extend.weight, -s, s)
            torch.nn.init.zeros_(block.rec.inp_collapse.weight)
            # Conv gate path
            torch.nn.init.uniform_(block.rec.gate_conv.weight, -s, s)
            # Recurrence
            torch.nn.init.zeros_(block.rec.decay_raw)  # sigmoid(0) = 0.5
            # Output projection
            torch.nn.init.zeros_(block.rec.proj_out.weight)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        embed_params = self.transformer.wte.weight.numel()
        decay_params = sum(b.rec.decay_raw.numel() for b in self.transformer.h)
        matmul_params = nparams - embed_params - decay_params
        return 6 * matmul_params

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = sum(b.rec.decay_raw.numel() for b in self.transformer.h)
        total = wte + lm_head + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': 0,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices - scalars,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = []   # 2D → Muon
        conv_params = []     # 3D (conv weights) → AdamW
        scalar_params = []   # 1D → AdamW
        for block in self.transformer.h:
            for name, p in block.named_parameters():
                if p.dim() == 1:
                    scalar_params.append(p)
                elif p.dim() == 2:
                    matrix_params.append(p)
                elif p.dim() == 3:
                    conv_params.append(p)

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        total_params = len(matrix_params) + len(conv_params) + len(scalar_params) + len(embedding_params) + len(lm_head_params)
        assert total_params == len(list(self.parameters())), f"Parameter count mismatch: {total_params} vs {len(list(self.parameters()))}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=scalar_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        ]
        # Conv weights (3D) → AdamW (Muon Newton-Schulz requires 2D)
        if conv_params:
            param_groups.append(
                dict(kind='adamw', params=conv_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.01),
            )
        # Muon for 2D matrix params (proj_out, MLP)
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
        """Autoregressive generation using step-mode (O(1) per token)."""
        assert isinstance(tokens, list)
        device = self.get_device()
        dim = self.config.n_embd
        n_layers = self.config.n_layer
        k = self.config.kernel_size

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Initialize states: recurrence + conv buffer per layer
        h_states = [torch.zeros(1, dim, device=device) for _ in range(n_layers)]
        conv_states = [torch.zeros(1, dim, k - 1, device=device) for _ in range(n_layers)]

        # Process prefix
        for tid in tokens:
            tok_emb = self.transformer.wte(torch.tensor([tid], device=device))
            x = norm(tok_emb)
            for i, block in enumerate(self.transformer.h):
                h_new, x, new_conv = block.step(x, h_states[i], conv_states[i])
                h_states[i] = h_new
                conv_states[i] = new_conv

        # Generate tokens
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
            for i, block in enumerate(self.transformer.h):
                h_new, x, new_conv = block.step(x, h_states[i], conv_states[i])
                h_states[i] = h_new
                conv_states[i] = new_conv

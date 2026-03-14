"""
Rechat + Recall — reconstruction-regularized recurrence.

Identical to Rechat, but with an auxiliary loss that trains the model to
reconstruct past tokens from the hidden state. This teaches the decay gates
to selectively preserve important information.

Auxiliary loss:
  For random positions t, sample a distant past position k.
  From h[t] (last recurrence layer), predict token[k].
  aux_loss = cross_entropy(reconstruct_head(h[t]), token[k])
  total_loss = main_loss + aux_weight * aux_loss
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class RechatRecallConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    chunk_size: int = 128
    aux_weight: float = 0.1       # weight of reconstruction loss
    aux_samples: int = 4          # number of (t, k) pairs per batch element
    aux_min_distance: int = 32    # minimum distance between t and k


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# ---------------------------------------------------------------------------
# Parallel scan (identical to rechat.py)
# ---------------------------------------------------------------------------

def parallel_scan_chunked(decay, x, chunk_size=128):
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
# DiagRecurrence — returns h states when needed for aux loss
# ---------------------------------------------------------------------------

class DiagRecurrence(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd
        self.chunk_size = config.chunk_size
        self.proj_decay = Linear(dim, dim, bias=False)
        self.proj_in = Linear(dim, dim, bias=False)
        self.proj_gate = Linear(dim, dim, bias=False)
        self.proj_out = Linear(dim, dim, bias=False)

    def forward(self, x, return_h=False):
        decay = torch.sigmoid(self.proj_decay(x))
        inp = self.proj_in(x)
        gate = torch.sigmoid(self.proj_gate(x))
        gated = inp * gate
        h = parallel_scan_chunked(decay, gated, self.chunk_size)
        out = self.proj_out(h)
        if return_h:
            return out, h  # return full h sequence for aux loss
        return out

    def step(self, x, h_prev):
        decay = torch.sigmoid(self.proj_decay(x))
        inp = self.proj_in(x)
        gate = torch.sigmoid(self.proj_gate(x))
        h_new = decay * h_prev + inp * gate
        return h_new, self.proj_out(h_new)


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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rec = DiagRecurrence(config)
        self.mlp = MLP(config)

    def forward(self, x, return_h=False):
        if return_h:
            rec_out, h = self.rec(norm(x), return_h=True)
            x = x + rec_out
            x = x + self.mlp(norm(x))
            return x, h
        x = x + self.rec(norm(x))
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, h_prev):
        h_new, rec_out = self.rec.step(norm(x), h_prev)
        x = x + rec_out
        x = x + self.mlp(norm(x))
        return h_new, x


# ---------------------------------------------------------------------------
# Full model with reconstruction auxiliary loss
# ---------------------------------------------------------------------------

class RechatRecall(nn.Module):
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
        # Reconstruction head: from h state, predict past token
        self.reconstruct_head = Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.reconstruct_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            torch.nn.init.uniform_(block.rec.proj_decay.weight, -s, s)
            torch.nn.init.uniform_(block.rec.proj_in.weight, -s, s)
            torch.nn.init.uniform_(block.rec.proj_gate.weight, -s, s)
            torch.nn.init.zeros_(block.rec.proj_out.weight)
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
        reconstruct = sum(p.numel() for p in self.reconstruct_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + reconstruct + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': 0,
            'lm_head': lm_head,
            'reconstruct_head': reconstruct,
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
        reconstruct_params = list(self.reconstruct_head.parameters())

        total_params = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(reconstruct_params)
        assert total_params == len(list(self.parameters())), f"Parameter count mismatch: {total_params} vs {len(list(self.parameters()))}"

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            # Reconstruct head — same LR as unembedding (it's also a vocab-sized projection)
            dict(kind='adamw', params=reconstruct_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
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

    def _compute_aux_loss(self, h, idx):
        """
        From h[t] (hidden state at position t), predict token idx[k] for distant k < t.
        h: (B, T, dim) — hidden states from last recurrence layer
        idx: (B, T) — input token ids
        """
        B, T, D = h.shape
        min_dist = self.config.aux_min_distance
        n_samples = self.config.aux_samples

        if T <= min_dist + 1:
            return torch.tensor(0.0, device=h.device)

        # Sample random (t, k) pairs: t in [min_dist, T), k in [0, t - min_dist)
        # Bias t toward later positions (more history to reconstruct from)
        device = h.device
        t_positions = torch.randint(min_dist, T, (B, n_samples), device=device)
        # k is at least min_dist before t
        max_k = (t_positions - min_dist).clamp(min=0)
        k_positions = (torch.rand(B, n_samples, device=device) * (max_k.float() + 1)).long().clamp(max=max_k)

        # Gather h[t] and target tokens[k]
        # h_at_t: (B, n_samples, D)
        t_expanded = t_positions.unsqueeze(-1).expand(-1, -1, D)
        h_at_t = h.gather(1, t_expanded)

        # target_tokens: (B, n_samples)
        target_tokens = idx.gather(1, k_positions)

        # Predict: reconstruct_head(h_at_t) → logits over vocab
        logits = self.reconstruct_head(h_at_t)  # (B, n_samples, vocab)
        logits = logits[..., :self.config.vocab_size].float()

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            reduction='mean'
        )
        return loss

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Run all blocks; get h from the last recurrence layer for aux loss
        h_last = None
        for i, block in enumerate(self.transformer.h):
            if i == len(self.transformer.h) - 1 and targets is not None:
                # Last block: return h for aux loss (only during training)
                x, h_last = block(x, return_h=True)
            else:
                x = block(x)

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            # Auxiliary reconstruction loss
            if h_last is not None and self.config.aux_weight > 0:
                aux_loss = self._compute_aux_loss(h_last, idx)
                loss = main_loss + self.config.aux_weight * aux_loss
            else:
                loss = main_loss

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
            x, h = block(x, return_h=True)
            h_states.append(h[:, -1, :])  # keep last state per layer
        x = x[:, -1, :]

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
            new_h = []
            for i, block in enumerate(self.transformer.h):
                h_new, x = block.step(x, h_states[i])
                new_h.append(h_new)
            h_states = new_h

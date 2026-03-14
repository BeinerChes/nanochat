"""
Benchmark inference speed: GPT vs Rechat at different prompt lengths.
Measures tokens/sec for generating 50 tokens after a prompt of varying length.
"""

import time
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.gpt import GPT, GPTConfig
from nanochat.rechat import Rechat, RechatConfig


def bench_model(model, name, prompt_lengths, gen_tokens=50, warmup=True):
    device = model.get_device()
    results = []

    for plen in prompt_lengths:
        # Fake prompt tokens (just random valid token ids)
        tokens = list(range(1, plen + 1))

        # Warmup run
        if warmup:
            for _ in model.generate(tokens[:min(16, plen)], max_tokens=5, temperature=0.0):
                pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmup = False

        # Timed run
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        count = 0
        for _ in model.generate(tokens, max_tokens=gen_tokens, temperature=0.0):
            count += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tok_per_sec = count / elapsed
        results.append((plen, count, elapsed, tok_per_sec))
        print(f"  {name:12s} | prompt={plen:5d} | generated={count:3d} | {elapsed:.3f}s | {tok_per_sec:.1f} tok/s")

    return results


def main():
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    prompt_lengths = [16, 64, 128, 256, 512]
    gen_tokens = 50

    # --- GPT ---
    print("\n=== GPT (depth=4, dim=768) ===")
    gpt_config = GPTConfig(n_layer=4, n_embd=768, sequence_len=1024)
    gpt = GPT(gpt_config).to(device)
    gpt.init_weights()
    gpt.eval()
    gpt_results = bench_model(gpt, "GPT-d4", prompt_lengths, gen_tokens)
    del gpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Rechat depth=4 ---
    print("\n=== Rechat (depth=4, dim=768) ===")
    rec4_config = RechatConfig(n_layer=4, n_embd=768, sequence_len=1024)
    rec4 = Rechat(rec4_config).to(device)
    rec4.init_weights()
    rec4.eval()
    rec4_results = bench_model(rec4, "Rechat-d4", prompt_lengths, gen_tokens)
    del rec4
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Rechat depth=11 ---
    print("\n=== Rechat (depth=11, dim=768) ===")
    rec11_config = RechatConfig(n_layer=11, n_embd=768, sequence_len=1024)
    rec11 = Rechat(rec11_config).to(device)
    rec11.init_weights()
    rec11.eval()
    rec11_results = bench_model(rec11, "Rechat-d11", prompt_lengths, gen_tokens)
    del rec11

    # --- Summary ---
    print("\n" + "=" * 75)
    print(f"{'Prompt':>8s} | {'GPT-d4':>12s} | {'Rechat-d4':>12s} | {'Rechat-d11':>12s} | {'R4/GPT':>8s} | {'R11/GPT':>8s}")
    print("-" * 75)
    for i, plen in enumerate(prompt_lengths):
        g = gpt_results[i][3]
        r4 = rec4_results[i][3]
        r11 = rec11_results[i][3]
        print(f"{plen:>8d} | {g:>10.1f}/s | {r4:>10.1f}/s | {r11:>10.1f}/s | {r4/g:>7.2f}x | {r11/g:>7.2f}x")


if __name__ == "__main__":
    main()

"""
Interactive generation for any model type (GPT, Rechat, MinGRU, RechatRecall).

Usage:
  uv run python -m scripts.generate --model rechat_d11
  uv run python -m scripts.generate --model d4 --prompt "Once upon a time"
  uv run python -m scripts.generate --model mingru_d13 --temperature 0.5
  uv run python -m scripts.generate --model rechat_d4 --prompt-file story.txt

Loads from base_checkpoints/<model>/ by default.
"""

import os
import sys
import json
import argparse
import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.rechat import Rechat, RechatConfig
from nanochat.rechat_recall import RechatRecall, RechatRecallConfig
from nanochat.mingru import MinGRU, MinGRUConfig

# Model registry: dirname prefix -> (class, config_class)
MODEL_REGISTRY = {
    "d": (GPT, GPTConfig),               # d4, d12, etc.
    "rechat": (Rechat, RechatConfig),     # rechat_d4, rechat_d11
    "recall": (RechatRecall, RechatRecallConfig),  # recall_d4
    "mingru": (MinGRU, MinGRUConfig),     # mingru_d13
}


def detect_model_type(dirname):
    """Detect model class from checkpoint directory name."""
    for prefix, (cls, cfg) in MODEL_REGISTRY.items():
        if dirname.startswith(prefix):
            return cls, cfg
    raise ValueError(f"Unknown model type for '{dirname}'. Known prefixes: {list(MODEL_REGISTRY.keys())}")


def load_model(checkpoint_dir, step, device):
    """Load any model type from checkpoint, auto-detecting type from dir name."""
    dirname = os.path.basename(checkpoint_dir)
    model_class, config_class = detect_model_type(dirname)

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")

    with open(meta_path) as f:
        meta = json.load(f)

    config = config_class(**meta["model_config"])
    model_data = torch.load(model_path, map_location=device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    with torch.device("meta"):
        model = model_class(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    print(f"Loaded {model_class.__name__} from {checkpoint_dir} (step {step})")
    print(f"  Config: {meta['model_config']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    return model


def find_latest_step(checkpoint_dir):
    """Find the latest checkpoint step in a directory."""
    steps = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("meta_") and f.endswith(".json"):
            step = int(f.replace("meta_", "").replace(".json", ""))
            steps.append(step)
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)


def generate_streaming(model, tokenizer, prompt_text, max_tokens, temperature, top_k, seed):
    """Generate tokens and print them as they come."""
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(prompt_text, prepend=bos)
    print(f"\n[{len(tokens)} input tokens, generating up to {max_tokens}...]\n")
    print(prompt_text, end="", flush=True)

    generated = []
    for tok in model.generate(tokens, max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=seed):
        generated.append(tok)
        # Decode incrementally
        text = tokenizer.decode(generated)
        # Print only newly decoded characters
        prev_text = tokenizer.decode(generated[:-1]) if len(generated) > 1 else ""
        new_chars = text[len(prev_text):]
        print(new_chars, end="", flush=True)

    print("\n")
    return generated


def interactive_loop(model, tokenizer, args):
    """Interactive prompt loop."""
    print("\nInteractive mode. Type your prompt and press Enter.")
    print("Commands: /quit, /temp <val>, /topk <val>, /tokens <val>, /seed <val>\n")

    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens
    seed = args.seed

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            temperature = float(prompt.split()[1])
            print(f"  temperature = {temperature}")
            continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1])
            print(f"  top_k = {top_k}")
            continue
        if prompt.startswith("/tokens "):
            max_tokens = int(prompt.split()[1])
            print(f"  max_tokens = {max_tokens}")
            continue
        if prompt.startswith("/seed "):
            seed = int(prompt.split()[1])
            print(f"  seed = {seed}")
            continue

        generate_streaming(model, tokenizer, prompt, max_tokens, temperature, top_k, seed)


def main():
    parser = argparse.ArgumentParser(description="Generate text with any nanochat model")
    parser.add_argument("--model", type=str, required=True, help="Model directory name (e.g. d4, rechat_d11, mingru_d13)")
    parser.add_argument("--step", type=int, default=-1, help="Checkpoint step (-1 = latest)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read prompt from file")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    tokenizer = get_tokenizer()
    base_dir = get_base_dir()

    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model)
    if not os.path.exists(checkpoint_dir):
        print(f"Error: {checkpoint_dir} not found")
        available = os.listdir(os.path.join(base_dir, "base_checkpoints"))
        print(f"Available: {sorted(available)}")
        sys.exit(1)

    step = args.step if args.step > 0 else find_latest_step(checkpoint_dir)
    model = load_model(checkpoint_dir, step, device)

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
        generate_streaming(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.seed)
    elif args.prompt:
        generate_streaming(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.seed)
    else:
        interactive_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()

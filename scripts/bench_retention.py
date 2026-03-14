"""
Memory retention test: can the model remember a character name through filler text?

Setup:
  "Once upon a time there was a girl named Zephyr. She loved to paint."
  + [N tokens of filler from TinyStories]
  → generate 100 tokens
  → does "Zephyr" appear in the output?

Tests Rechat and GPT at varying filler lengths.
"""

import os
import json
import torch
from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.rechat import Rechat, RechatConfig
from nanochat.rechat_recall import RechatRecall, RechatRecallConfig


def load_model_generic(checkpoint_dir, step, device, model_class, config_class):
    """Load any model type from checkpoint."""
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    config_kwargs = meta["model_config"]
    config = config_class(**config_kwargs)
    model_data = torch.load(model_path, map_location=device)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    with torch.device("meta"):
        model = model_class(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model


def get_filler_tokens(tokenizer, num_tokens):
    """Get filler text tokens from TinyStories-like content."""
    # Generic children's story filler that won't mention "Zephyr"
    filler_sentences = [
        "The sun was shining bright in the sky.",
        "There were many trees in the park.",
        "The birds were singing a happy song.",
        "A little cat sat on the fence and watched.",
        "The flowers were red and yellow and blue.",
        "It was a very nice day to play outside.",
        "The children were running and laughing.",
        "A big bear was sleeping under the tree.",
        "The wind blew softly through the leaves.",
        "There was a pond with little fish swimming.",
        "The butterfly flew from flower to flower.",
        "A rabbit hopped across the green grass.",
        "The clouds looked like fluffy white pillows.",
        "Some ducks were swimming in the lake.",
        "The ice cream truck came down the street.",
        "A friendly squirrel climbed up the oak tree.",
        "The garden had many pretty roses growing.",
        "Two frogs were sitting on a lily pad.",
        "The old bridge crossed over the river.",
        "A ladybug landed on a big green leaf.",
    ]
    # Tokenize all filler and repeat until we have enough
    all_tokens = []
    i = 0
    while len(all_tokens) < num_tokens:
        tokens = tokenizer.encode(filler_sentences[i % len(filler_sentences)])
        all_tokens.extend(tokens)
        i += 1
    return all_tokens[:num_tokens]


def run_retention_test(model, tokenizer, filler_lengths, gen_tokens=100, num_tries=3):
    """Test if the model retains 'Zephyr' through varying amounts of filler."""
    bos = tokenizer.get_bos_token_id()
    prompt = "Once upon a time there was a little dog. The dog loved to play with his big red ball."
    prompt_tokens = tokenizer.encode(prompt, prepend=bos)

    results = []
    for filler_len in filler_lengths:
        filler = get_filler_tokens(tokenizer, filler_len)
        input_tokens = prompt_tokens + filler

        hits = 0
        outputs = []
        for trial in range(num_tries):
            generated = []
            for tok in model.generate(input_tokens, max_tokens=gen_tokens, temperature=0.8, top_k=40, seed=42 + trial):
                generated.append(tok)
            text = tokenizer.decode(generated)
            has_name = "dog" in text.lower()
            if has_name:
                hits += 1
            outputs.append(text)

        total_input = len(input_tokens)
        results.append((filler_len, total_input, hits, num_tries, outputs))
        hit_str = f"{hits}/{num_tries}"
        print(f"  filler={filler_len:4d} tokens (total input={total_input:4d}) | dog recalled: {hit_str}")
        # Show first output
        print(f"    → {outputs[0][:150]}")

    return results


def main():
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    tokenizer = get_tokenizer()
    base_dir = get_base_dir()
    checkpoints = os.path.join(base_dir, "base_checkpoints")

    filler_lengths = [0, 50, 100, 200, 400]

    # --- GPT d4 ---
    gpt_dir = os.path.join(checkpoints, "d4")
    if os.path.exists(gpt_dir):
        print("\n=== GPT d4 ===")
        gpt = load_model_generic(gpt_dir, 5000, device, GPT, GPTConfig)
        run_retention_test(gpt, tokenizer, filler_lengths)
        del gpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Rechat d4 ---
    rechat4_dir = os.path.join(checkpoints, "rechat_d4")
    if os.path.exists(rechat4_dir):
        print("\n=== Rechat d4 ===")
        rec4 = load_model_generic(rechat4_dir, 5000, device, Rechat, RechatConfig)
        run_retention_test(rec4, tokenizer, filler_lengths)
        del rec4
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Rechat d11 ---
    rechat11_dir = os.path.join(checkpoints, "rechat_d11")
    if os.path.exists(rechat11_dir):
        print("\n=== Rechat d11 ===")
        rec11 = load_model_generic(rechat11_dir, 5000, device, Rechat, RechatConfig)
        run_retention_test(rec11, tokenizer, filler_lengths)
        del rec11
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- RechatRecall d4 ---
    recall_dir = os.path.join(checkpoints, "recall_d4")
    if os.path.exists(recall_dir):
        print("\n=== RechatRecall d4 (aux retention loss) ===")
        recall = load_model_generic(recall_dir, 5000, device, RechatRecall, RechatRecallConfig)
        run_retention_test(recall, tokenizer, filler_lengths)
        del recall


if __name__ == "__main__":
    main()

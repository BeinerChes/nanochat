"""
Deterministic memory retention test via logit probing.

No generation, no sampling, no randomness. One forward pass per test.

Setup:
  "Once upon a time there was a girl named Lily." + [N filler tokens] + "The girl's name was"
  → check: what rank/probability does "Lily" get in next-token logits?

Tests multiple fact types, averages across them, plots smooth decay curves.
"""

import math
import os
import json
import torch
import torch.nn.functional as F
from nanochat.common import compute_init, autodetect_device_type, get_base_dir, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.rechat import Rechat, RechatConfig
from nanochat.rechat_recall import RechatRecall, RechatRecallConfig
from nanochat.mingru import MinGRU, MinGRUConfig


def find_latest_step(checkpoint_dir):
    """Find the latest checkpoint step in a directory."""
    steps = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("meta_") and f.endswith(".json"):
            step = int(f.replace("meta_", "").replace(".json", ""))
            steps.append(step)
    return max(steps) if steps else None


def load_model_generic(checkpoint_dir, step, device, model_class, config_class):
    """Load any model type from checkpoint. step=-1 for latest."""
    if step == -1:
        step = find_latest_step(checkpoint_dir)
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


# --- Test cases: (setup, filler-free probe, target_word) ---
# Each test: feed [setup + filler + probe], check P(target) at the last position.
# target_word must be a single token (with leading space).
TEST_CASES = [
    {
        "setup": "Once upon a time there was a girl named Lily.",
        "probe": " The girl's name was",
        "target": " Lily",
        "label": "name:Lily",
    },
    {
        "setup": "There was a boy called Max who liked to run.",
        "probe": " The boy's name was",
        "target": " Max",
        "label": "name:Max",
    },
    {
        "setup": "Once upon a time there was a boy named Tim.",
        "probe": " The boy's name was",
        "target": " Tim",
        "label": "name:Tim",
    },
    {
        "setup": "Sam had a big red ball that he loved.",
        "probe": " The ball was",
        "target": " red",
        "label": "color:red",
    },
    {
        "setup": "Mia found a blue flower in the garden.",
        "probe": " The flower was",
        "target": " blue",
        "label": "color:blue",
    },
    {
        "setup": "Ben had a small cat named Rex.",
        "probe": " Ben's pet was a",
        "target": " cat",
        "label": "animal:cat",
    },
    {
        "setup": "Lily went to the park with her dog.",
        "probe": " Lily went to the",
        "target": " park",
        "label": "place:park",
    },
    {
        "setup": "The happy bird sang a song in the tree.",
        "probe": " The bird was",
        "target": " happy",
        "label": "mood:happy",
    },
]


def get_filler_tokens(tokenizer, num_tokens):
    """Get filler text tokens."""
    filler_sentences = [
        "The sun was shining bright in the sky.",
        "There were many trees in the forest.",
        "The birds were singing a beautiful song.",
        "A little squirrel sat on the fence and watched.",
        "The flowers were pink and yellow and white.",
        "It was a very nice day to play outside.",
        "The children were running and laughing.",
        "A bear was sleeping under the old oak.",
        "The wind blew softly through the leaves.",
        "There was a pond with fish swimming around.",
        "The butterfly flew from petal to petal.",
        "A rabbit hopped across the green grass.",
        "The clouds looked like fluffy white pillows.",
        "Some ducks were swimming in the lake.",
        "The ice cream truck came down the street.",
        "A friendly mouse climbed up the wooden stairs.",
        "The garden had many pretty roses growing.",
        "Two frogs were sitting on a lily pad.",
        "The old bridge crossed over the river.",
        "A ladybug landed on a big green leaf.",
    ]
    all_tokens = []
    i = 0
    while len(all_tokens) < num_tokens:
        tokens = tokenizer.encode(filler_sentences[i % len(filler_sentences)])
        all_tokens.extend(tokens)
        i += 1
    return all_tokens[:num_tokens]


def bits_retained(rank, vocab_size=32768):
    """How many bits of the original info survive. 15 = perfect, 0 = random."""
    max_bits = math.log2(vocab_size)
    return max(0, max_bits - math.log2(rank))


@torch.inference_mode()
def probe_retention(model, tokenizer, test_cases, filler_lengths):
    """
    For each test case and filler length:
    - Encode: [BOS] + setup + filler + probe
    - Get logits at last position
    - Report rank, probability, log-prob, and bits retained
    """
    bos = tokenizer.get_bos_token_id()
    device = model.get_device()
    vocab_size = 32768
    results = {}  # filler_len -> list of (rank, prob, logprob, bits, label)

    for filler_len in filler_lengths:
        filler = get_filler_tokens(tokenizer, filler_len)
        results[filler_len] = []

        for tc in test_cases:
            setup_tokens = tokenizer.encode(tc["setup"], prepend=bos)
            probe_tokens = tokenizer.encode(tc["probe"])
            target_tokens = tokenizer.encode(tc["target"])

            if len(target_tokens) != 1:
                print(f"  WARNING: '{tc['target']}' is {len(target_tokens)} tokens, skipping")
                continue
            target_id = target_tokens[0]

            input_tokens = setup_tokens + filler + probe_tokens
            ids = torch.tensor([input_tokens], dtype=torch.long, device=device)

            logits = model(ids)  # (1, T, vocab)
            last_logits = logits[0, -1, :]

            probs = F.softmax(last_logits, dim=-1)
            target_prob = probs[target_id].item()
            logprob = math.log2(max(target_prob, 1e-10))

            rank = (last_logits > last_logits[target_id]).sum().item() + 1
            bits = bits_retained(rank, vocab_size)

            results[filler_len].append((rank, target_prob, logprob, bits, tc["label"]))

        ranks = [r[0] for r in results[filler_len]]
        bits_list = [r[3] for r in results[filler_len]]
        logprobs = [r[2] for r in results[filler_len]]
        avg_bits = sum(bits_list) / len(bits_list)
        avg_logprob = sum(logprobs) / len(logprobs)
        median_rank = sorted(ranks)[len(ranks) // 2]

        detail = "  ".join(f"{r[4]}:{r[3]:.1f}b" for r in results[filler_len])
        print(f"  filler={filler_len:4d} | bits={avg_bits:5.1f}/15  log2p={avg_logprob:6.1f}  med_rank={median_rank:5d} | {detail}")

    return results


def main():
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    tokenizer = get_tokenizer()
    base_dir = get_base_dir()
    checkpoints = os.path.join(base_dir, "base_checkpoints")

    filler_lengths = [0, 32, 64, 128, 256, 512, 1024]

    models_to_test = [
        ("GPT d4 (37M)", "d4", GPT, GPTConfig),
        ("Rechat d6 (far, 48M)", "rechat_d6", Rechat, RechatConfig),
        ("Rechat d11 (nofar, 112M)", "rechat_d11_nofar", Rechat, RechatConfig),
        ("Rechat d11 (far, 135M)", "rechat_d11", Rechat, RechatConfig),
        ("Rechat d4 (far, 28M)", "rechat_d4", Rechat, RechatConfig),
        ("RechatRecall d4 (28M)", "recall_d4", RechatRecall, RechatRecallConfig),
        ("MinGRU d13", "mingru_d13", MinGRU, MinGRUConfig),
    ]

    all_results = {}
    for name, dirname, model_class, config_class in models_to_test:
        model_dir = os.path.join(checkpoints, dirname)
        if not os.path.exists(model_dir):
            continue
        print(f"\n=== {name} ===")
        model = load_model_generic(model_dir, -1, device, model_class, config_class)
        all_results[name] = probe_retention(model, tokenizer, TEST_CASES, filler_lengths)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table: avg bits retained
    print("\n" + "=" * 70)
    print("SUMMARY: Average bits retained (out of 15 = perfect)")
    print("=" * 70)
    header = f"{'Filler':>8}"
    for name in all_results:
        short = name.split("(")[0].strip()[:12]
        header += f" | {short:>12}"
    print(header)
    print("-" * len(header))
    for fl in filler_lengths:
        row = f"{fl:>8}"
        for name, results in all_results.items():
            if fl in results:
                bits_list = [r[3] for r in results[fl]]
                avg_bits = sum(bits_list) / len(bits_list)
                row += f" | {avg_bits:>11.1f}b"
            else:
                row += f" | {'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()

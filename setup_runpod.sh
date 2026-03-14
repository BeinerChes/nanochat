#!/bin/bash
# RunPod setup for Rechat d20 training (3× A100 PCIe)
# Assumes PyTorch is already installed (RunPod template)
set -e

cd /workspace

# 1. Clone repo
git clone https://github.com/BeinerChes/nanochat.git
cd nanochat
git checkout pre_nano_chat

# 2. Install Python deps (PyTorch already present, just add the rest)
pip install wandb datasets transformers tokenizers tiktoken regex scipy tabulate zstandard psutil maturin

# 3. Build Rust tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
maturin develop --release --manifest-path rustbpe/Cargo.toml

# 4. Train tokenizer
python -m scripts.tok_train --max_chars=2000000000

# 5. Download data (240 shards for d20, ~24GB)
python -m nanochat.dataset -n 240

# 6. Download eval bundle
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip && rm eval_bundle.zip
mv eval_bundle "$NANOCHAT_BASE_DIR/"

# 7. Quick sanity check (20 steps)
echo "=== Sanity check: 20 steps ==="
python -m scripts.base_train_rechat \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=512 \
    --num-iterations=20 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1

echo ""
echo "=== Setup complete! Run: ==="
echo ""
echo "torchrun --standalone --nproc_per_node=3 -m scripts.base_train_rechat \\"
echo "    --depth=20 \\"
echo "    --device-batch-size=8 \\"
echo "    --target-param-data-ratio=20 \\"
echo "    --run=rechat-d20 \\"
echo "    --model-tag=rechat-d20 \\"
echo "    --save-every=2000 \\"
echo "    --eval-every=250 \\"
echo "    --core-metric-every=5000 \\"
echo "    --far-weight=0.01 --far-k-min=2 --far-k-max=64"

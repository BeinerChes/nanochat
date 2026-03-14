"""
Convert a text file (one document per line) to parquet format
compatible with nanochat's dataloader.

Usage:
    python -m scripts.convert_tinystories --input /path/to/tinystories.txt

This creates parquet files in base_data_climbmix/ (or the configured data dir),
replacing the default FineWeb dataset. The last shard is used as validation.
"""

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

parser = argparse.ArgumentParser(description="Convert text file to nanochat parquet format")
parser.add_argument("--input", type=str, required=True, help="Path to text file (one doc per line)")
parser.add_argument("--val-frac", type=float, default=0.01, help="Fraction of docs for validation")
parser.add_argument("--rows-per-group", type=int, default=1000, help="Rows per parquet row group")
parser.add_argument("--data-dir", type=str, default=None, help="Output directory (default: base_data_climbmix)")
args = parser.parse_args()

# Read documents
print(f"Reading {args.input}...")
documents = []
with open(args.input, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(line)
print(f"Loaded {len(documents)} documents")

# Split into train and val
import random
random.seed(42)
random.shuffle(documents)
n_val = max(1, int(len(documents) * args.val_frac))
val_docs = documents[:n_val]
train_docs = documents[n_val:]
print(f"Train: {len(train_docs)} docs, Val: {n_val} docs")

# Output directory
base_dir = get_base_dir()
data_dir = args.data_dir or os.path.join(base_dir, "base_data_climbmix")
os.makedirs(data_dir, exist_ok=True)

def write_parquet(docs, filepath, rows_per_group):
    """Write documents to a single parquet file with row groups."""
    writer = None
    schema = pa.schema([("text", pa.string())])
    for i in range(0, len(docs), rows_per_group):
        batch = docs[i:i + rows_per_group]
        table = pa.table({"text": batch}, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(filepath, schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    pf = pq.ParquetFile(filepath)
    print(f"  Written {filepath}: {len(docs)} docs, {pf.num_row_groups} row groups")

# Write train shard (shard_00000.parquet)
train_path = os.path.join(data_dir, "shard_00000.parquet")
write_parquet(train_docs, train_path, args.rows_per_group)

# Write val shard (shard_00001.parquet) — must be the LAST file alphabetically
val_path = os.path.join(data_dir, "shard_00001.parquet")
write_parquet(val_docs, val_path, args.rows_per_group)

print(f"\nDone! Parquet files written to {data_dir}")
print(f"nanochat dataloader will use shard_00000 for train, shard_00001 for val")

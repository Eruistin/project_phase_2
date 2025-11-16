#!/usr/bin/env python
"""
prepare_data_shadow.py
Generate training data for shadow models with different seeds
"""

import os, json, random, argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

MIN_TOKENS = 25

def set_seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_text_column(ds: Dataset, src: str) -> Dataset:
    if src == "wikitext103":
        assert "text" in ds.column_names
        return ds.remove_columns([c for c in ds.column_names if c != "text"])
    raise ValueError(src)

def basic_clean(ds: Dataset) -> Dataset:
    ds = ds.filter(lambda ex: isinstance(ex.get("text", None), str) and len(ex["text"].strip()) > 0)
    def _strip_map(ex):
        return {"text": " ".join(ex["text"].split())}
    return ds.map(_strip_map, batched=False)

def filter_by_tokens(ds: Dataset, tok, min_tokens: int) -> Dataset:
    def _len_map(batch):
        enc = tok(batch["text"], add_special_tokens=False)
        return {"_tok_len": [len(ids) for ids in enc["input_ids"]]}
    ds = ds.map(_len_map, batched=True)
    ds = ds.filter(lambda ex: ex["_tok_len"] >= min_tokens)
    return ds.remove_columns(["_tok_len"])

def sample_n(ds: Dataset, n: int, seed: int):
    """Sample n items from dataset with given seed"""
    n = min(n, len(ds))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    take = sorted(idx[:n])
    return ds.select(take), set(take)

def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this shadow model")
    parser.add_argument("--shadow_id", type=int, required=True, help="Shadow model ID (0, 1, 2, ...)")
    parser.add_argument("--total_samples", type=int, default=50000, help="Total samples in the pool")
    parser.add_argument("--samples_per_shadow", type=int, default=10000, help="Samples for this shadow")
    parser.add_argument("--output_dir", type=str, default="wiki_json/train", help="Output directory")
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS)
    args = parser.parse_args()
    
    set_seed_all(args.seed)
    
    print(f"Preparing shadow_{args.shadow_id} with seed={args.seed}")
    print(f"  Samples: {args.samples_per_shadow} / {args.total_samples}")
    
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Load and filter WikiText-103
    print("Loading WikiText-103...")
    wiki_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", trust_remote_code=True)["train"]
    wiki = ensure_text_column(wiki_raw, "wikitext103")
    wiki = basic_clean(wiki)
    wiki = filter_by_tokens(wiki, tok, args.min_tokens)
    
    print(f"Available samples after filtering: {len(wiki)}")
    
    # Sample for this shadow model
    wiki_train, wiki_train_idx = sample_n(wiki, args.samples_per_shadow, args.seed)
    
    # Save training data
    out_dir = Path(args.output_dir)
    train_json = [{"text": ex["text"]} for ex in wiki_train]
    output_file = out_dir / f"train_shadow_{args.shadow_id}.json"
    dump_json(output_file, train_json)
    
    # Save indices for reference
    indices_file = out_dir / f"train_shadow_{args.shadow_id}_indices.json"
    dump_json(indices_file, {
        "seed": args.seed,
        "shadow_id": args.shadow_id,
        "selected_indices": sorted(list(wiki_train_idx)),
        "num_samples": len(wiki_train_idx)
    })
    
    print(f"✓ Saved to {output_file}")
    print(f"✓ Indices saved to {indices_file}")

if __name__ == "__main__":
    main()
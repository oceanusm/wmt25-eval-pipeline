#!/usr/bin/env python3
import json

IN_PATH = "wmt25-genmt.jsonl"
OUT_DIR = "pair_splits"

# data is a list of dictionaries, with each dict corresponding to one document translation.
data = []
with open(IN_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

split_dict = {}
for ex in data:
    # Build the path of the jsonl it should go to
    output_path = f"{ex['src_lang']}--{ex['tgt_lang']}.jsonl"

    # Initialize if not yet seen
    if output_path not in split_dict:
        split_dict[output_path] = []
    # Append to the corresponding json contents list
    split_dict[output_path].append(ex)

# Write the list of dicts out.
for key, items in split_dict.items():
    print(f"{key} will contain {len(items)} examples.")
    with open(f"{OUT_DIR}/{key}", 'w', encoding='utf-8') as fout:
        for ex in items:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

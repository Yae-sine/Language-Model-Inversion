"""Select N prompts from wentingzhao/one-million-instructions for Llama-2 extraction.

Selection rules:
- Prompts without a system prompt only, so the text is a plain instruction, like Dolly.
  The text is `user.strip()`.
- Drop prompts from Dolly (by source name, and any text equal to an instruction in
  --exclude_instructions). This keeps the Dolly validation prompts unseen.
- Drop duplicate texts.
- Keep only prompts with <= --max_t5_tokens T5 tokens (incl. </s>), the same rule as
  build_dolly_llama2_dataset.py, so no GPU time is spent on prompts dropped later.
- Take a random sample of N (seeded).

Writes to --out_dir:
    instructions.npy  unicode array (N,)   the selected prompts
    sources.npy       unicode array (N,)   their source dataset
    selection.json    counts for every filtering step
    vectors.npy       float32 (N, 32000)   created empty (zeros), to be filled by
                                           extract_llama2_logprobs.py
"""

import argparse
import collections
import json
import os

import datasets
import numpy as np
import transformers

VOCAB_SIZE = 32000  # Llama-2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_t5_tokens", type=int, default=64)
    p.add_argument("--t5", default="t5-base")
    p.add_argument("--exclude_instructions", default=None,
                   help="instructions.npy whose texts must not be selected (e.g. Dolly)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = datasets.load_dataset("wentingzhao/one-million-instructions", split="train")
    users, systems, sources = ds["user"], ds["system"], ds["source"]
    stats = {"total": len(users)}
    stats["sources_total"] = dict(collections.Counter(sources).most_common())

    exclude = set()
    if args.exclude_instructions:
        exclude = {str(s).strip() for s in np.load(args.exclude_instructions)}
    stats["exclude_list_size"] = len(exclude)

    seen = set()
    candidates = []  # (text, source)
    dropped = collections.Counter()
    for user, system, source in zip(users, systems, sources):
        text = (user or "").strip()
        if (system or "").strip():
            dropped["has_system_prompt"] += 1
        elif not text:
            dropped["empty"] += 1
        elif "dolly" in (source or "").lower() or text in exclude:
            dropped["dolly"] += 1
        elif text in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(text)
            candidates.append((text, source))
    stats["dropped"] = dict(dropped)
    stats["candidates"] = len(candidates)
    print(json.dumps(stats, indent=1)[:3000], flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.t5)
    order = np.random.default_rng(args.seed).permutation(len(candidates))
    chosen, too_long, i = [], 0, 0
    while len(chosen) < args.n and i < len(order):
        block = [candidates[j] for j in order[i : i + 20000]]
        i += 20000
        lengths = [len(x) for x in tokenizer([t for t, _ in block])["input_ids"]]
        for (text, source), length in zip(block, lengths):
            if length > args.max_t5_tokens:
                too_long += 1
            elif len(chosen) < args.n:
                chosen.append((text, source))
    assert len(chosen) == args.n, f"only {len(chosen)} prompts available"
    stats["too_long_seen"] = too_long
    stats["selected"] = len(chosen)
    stats["sources_selected"] = dict(
        collections.Counter(s for _, s in chosen).most_common()
    )

    np.save(os.path.join(args.out_dir, "instructions.npy"), np.array([t for t, _ in chosen]))
    np.save(os.path.join(args.out_dir, "sources.npy"), np.array([s for _, s in chosen]))
    np.lib.format.open_memmap(
        os.path.join(args.out_dir, "vectors.npy"), mode="w+",
        dtype=np.float32, shape=(args.n, VOCAB_SIZE),
    ).flush()
    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("selected", len(chosen), "| sources:", stats["sources_selected"], flush=True)


if __name__ == "__main__":
    main()

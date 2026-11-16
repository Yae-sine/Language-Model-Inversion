"""Compute Llama-2-7B next-token log-probs for a list of prompts.

Same method as the teammate's llama2/llama2_extraction.ipynb, which produced the
Dolly-15k vectors:
- Llama-2-7B loaded in 8-bit (bitsandbytes LLM.int8), fp16 compute, one GPU
- pad token = </s>, left padding, truncation at 512 tokens, <s> prepended
- logits at the last position, log_softmax in float32

Differences from the notebook. None of them changes the numbers beyond rounding noise:
- `num_logits_to_keep=1` makes Llama compute logits for the last position only.
  These are the same logits as `outputs.logits[:, -1, :]`, using less memory.
- With --sort_by_length, each batch holds prompts of similar length, capped at
  --max_batch_tokens. That means less padding and faster batches. LLM.int8 picks
  outlier columns per batch, so batch composition can shift values by rounding
  noise only.
- A row whose vector is not finite (NaN/inf) is recomputed alone, without padding,
  and counted in the log. The notebook would have stored it as NaN.

Rows are written in place into an existing vectors.npy (float32, N x 32000), e.g.
the one created by select_one_million_prompts.py. --shard k/n handles every n-th
row starting at k, so n processes (one per GPU) can fill the same file.
"""

import argparse
import os
import time

import numpy as np
import torch
import transformers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--instructions", required=True)
    p.add_argument("--vectors", required=True, help="existing .npy, filled in place")
    p.add_argument("--rows", default=None, help="start:end (default: all)")
    p.add_argument("--shard", default="0/1", help="k/n: rows with index %% n == k")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_batch_tokens", type=int, default=4096,
                   help="cap on batch_size x longest prompt (with --sort_by_length)")
    p.add_argument("--sort_by_length", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50, help="batches")
    args = p.parse_args()

    instructions = np.load(args.instructions, mmap_mode="r")
    vectors = np.load(args.vectors, mmap_mode="r+")
    assert vectors.shape[0] == len(instructions) and vectors.dtype == np.float32
    start, end = 0, len(instructions)
    if args.rows:
        a, b = args.rows.split(":")
        start, end = int(a or 0), int(b or len(instructions))
    k, n = (int(x) for x in args.shard.split("/"))
    rows = [i for i in range(start, end) if i % n == k]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.eval()

    texts = {i: str(instructions[i]) for i in rows}
    if args.sort_by_length:
        lengths = {i: min(len(ids), 512) for i, ids in zip(
            rows, tokenizer([texts[i] for i in rows])["input_ids"])}
        rows = sorted(rows, key=lambda i: -lengths[i])
        # Longest first: a batch's padded size is set by its first prompt.
        batches, cur = [], []
        for i in rows:
            if cur and (len(cur) == args.batch_size
                        or (len(cur) + 1) * lengths[cur[0]] > args.max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    else:
        batches = [rows[i : i + args.batch_size]
                   for i in range(0, len(rows), args.batch_size)]
    print(f"shard {k}/{n}: {len(rows)} prompts, {len(batches)} batches", flush=True)

    def log_probs_for(batch_texts):
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, num_logits_to_keep=1)
        logits = outputs.logits[:, -1, :]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    t0 = time.time()
    n_retried, n_bad = 0, 0
    for b, batch in enumerate(batches):
        lp = log_probs_for([texts[i] for i in batch])
        # A row that is not finite is recomputed alone (no padding in its batch).
        for j in np.where(~np.isfinite(lp).all(axis=1))[0]:
            n_retried += 1
            lp[j] = log_probs_for([texts[batch[j]]])[0]
            if not np.isfinite(lp[j]).all():
                n_bad += 1
                print(f"row {batch[j]} still not finite when run alone", flush=True)
        vectors[batch] = lp
        if (b + 1) % args.log_every == 0 or b + 1 == len(batches):
            done = sum(len(x) for x in batches[: b + 1])
            rate = done / (time.time() - t0)
            eta = (len(rows) - done) / rate
            print(f"shard {k}/{n}: {done}/{len(rows)} prompts, {rate:.1f}/s, "
                  f"ETA {eta / 60:.1f} min, recomputed alone: {n_retried}", flush=True)
    vectors.flush()
    print(f"shard {k}/{n}: done in {(time.time() - t0) / 60:.1f} min; "
          f"{n_retried} rows recomputed alone, {n_bad} still not finite", flush=True)


if __name__ == "__main__":
    main()

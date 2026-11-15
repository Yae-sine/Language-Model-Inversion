"""Build a vec2text training dataset from precomputed Llama-2 next-token log-probs.

Inputs (in --data_dir):
    instructions.npy  unicode array, shape (N,)        -- the prompts
    vectors.npy       float32 array, shape (N, 32000)  -- log_softmax of Llama-2's
                                                          next-token logits per prompt

Output: a DatasetDict saved with `save_to_disk`, with `train` and `validation`
splits and exactly the columns vec2text's own pipeline produces
(`Experiment._load_train_dataset_uncached` + `embed_dataset_batch`):

    input_ids, attention_mask, labels, length,
    embedder_input_ids, embedder_attention_mask, frozen_embeddings, idx

Point training at it with:
    VEC2TEXT_TRAIN_DATASET_PATH=<out> python vec2text/run.py \
        --experiment inversion_from_logits_emb --dataset_name dolly \
        --use_frozen_embeddings_as_input 1 --mock_embedder 1 ...

Example:
    python experiment_scripts/logits/build_dolly_llama2_dataset.py \
        --data_dir ~/data/llama2_dolly15k --out ~/data/dolly_llama2_inv
"""

import argparse
import math
import os

import datasets
import numpy as np
import transformers

from vec2text.tokenize_data import tokenize_function

VOCAB_SIZE = 32000  # Llama-2


def load_tokenizers(t5_name: str, embedder_name: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(t5_name)
    embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(
        embedder_name, token=os.environ.get("LLAMA_TOKEN")
    )
    # Same as vec2text.models.model_utils.load_embedder_and_tokenizer.
    embedder_tokenizer.pad_token = embedder_tokenizer.eos_token
    return tokenizer, embedder_tokenizer


def generate_examples(
    data_dir: str,
    t5_name: str,
    embedder_name: str,
    max_seq_length: int,
    rows: list,
    chunk_size: int,
):
    # Everything is (re)loaded here from plain arguments so that `datasets`
    # can fingerprint this generator without pickling the 2 GB vector array.
    instructions = np.load(os.path.join(data_dir, "instructions.npy"), mmap_mode="r")
    vectors = np.load(os.path.join(data_dir, "vectors.npy"), mmap_mode="r")
    tokenizer, embedder_tokenizer = load_tokenizers(t5_name, embedder_name)
    tokenize = tokenize_function(
        tokenizer, embedder_tokenizer, "text", max_seq_length, padding=False
    )

    for start in range(0, len(rows), chunk_size):
        batch_rows = rows[start : start + chunk_size]
        texts = [str(instructions[i]) for i in batch_rows]
        out = tokenize({"text": texts})
        emb = np.asarray(vectors[batch_rows], dtype=np.float32)
        for j in range(len(batch_rows)):
            ex = {}
            for k, v in out.items():
                x = v[j]
                ex[k] = x.tolist() if hasattr(x, "tolist") else x
            ex["frozen_embeddings"] = emb[j]
            yield ex


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t5", default="t5-base")
    parser.add_argument("--embedder", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    out_path = os.path.expanduser(args.out)

    instructions = np.load(os.path.join(data_dir, "instructions.npy"), mmap_mode="r")
    vectors = np.load(os.path.join(data_dir, "vectors.npy"), mmap_mode="r")
    n = len(instructions)
    assert vectors.shape == (n, VOCAB_SIZE), vectors.shape
    assert vectors.dtype == np.float32, vectors.dtype
    print(f"Loaded {n} prompts and vectors of shape {vectors.shape}")

    # Keep only prompts whose T5 tokenization (incl. </s>) fits in
    # max_seq_length, so the T5 target is never truncated while the Llama
    # vector was computed from the full prompt.
    tokenizer, _ = load_tokenizers(args.t5, args.embedder)
    texts = [str(s) for s in instructions]
    t5_lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    rows = [
        i
        for i, (t, length) in enumerate(zip(texts, t5_lengths))
        if t.strip() and length <= args.max_seq_length
    ]
    print(
        f"Keeping {len(rows)}/{n} prompts with <= {args.max_seq_length} T5 tokens "
        f"(dropped {n - len(rows)})"
    )

    features = datasets.Features(
        {
            "input_ids": datasets.Sequence(datasets.Value("int64")),
            "attention_mask": datasets.Sequence(datasets.Value("int64")),
            "labels": datasets.Sequence(datasets.Value("int64")),
            "length": datasets.Value("int64"),
            "embedder_input_ids": datasets.Sequence(datasets.Value("int64")),
            "embedder_attention_mask": datasets.Sequence(datasets.Value("int64")),
            "frozen_embeddings": datasets.Sequence(
                datasets.Value("float32"), length=VOCAB_SIZE
            ),
        }
    )
    dataset = datasets.Dataset.from_generator(
        generate_examples,
        features=features,
        gen_kwargs={
            "data_dir": data_dir,
            "t5_name": args.t5,
            "embedder_name": args.embedder,
            "max_seq_length": args.max_seq_length,
            "rows": rows,
            "chunk_size": args.chunk_size,
        },
    )

    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    dataset_dict = datasets.DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )
    for key in dataset_dict:
        dataset_dict[key] = dataset_dict[key].add_column(
            "idx", range(len(dataset_dict[key]))
        )
        dataset_dict[key].set_format("pt")
    print(dataset_dict)

    print("Saving to", out_path)
    dataset_dict.save_to_disk(out_path, max_shard_size="2GB")

    # Sanity checks on the saved copy.
    reloaded = datasets.load_from_disk(out_path)
    ex = reloaded["validation"][0]
    fe = ex["frozen_embeddings"].double()
    lse = (fe.max() + (fe - fe.max()).exp().sum().log()).item()
    labels = [t for t in ex["labels"].tolist() if t != -100]
    print("frozen_embeddings shape:", tuple(ex["frozen_embeddings"].shape))
    print(f"logsumexp of first validation vector: {lse:.5f} (expected ~0)")
    print("decoded label:", tokenizer.decode(labels, skip_special_tokens=True))
    assert ex["frozen_embeddings"].shape[-1] == VOCAB_SIZE
    assert math.isclose(lse, 0.0, abs_tol=1e-3), lse
    print("Done.")


if __name__ == "__main__":
    main()

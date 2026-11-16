"""Evaluate logits inverters on a saved dataset of precomputed LM log-probs.

Scores each model on every example of one split (default: validation) using the
same text metrics as training (vec2text BaseTrainer._text_comparison_metrics),
under several decoding settings, plus the eval loss.

A model is either a local checkpoint folder (config.json + model.safetensors, e.g.
saves/.../checkpoint-2628) or a Hugging Face repo id, e.g. the authors'
jxm/t5-base__llama-7b__one-million-instructions__emb. The LM (embedder) is never
loaded: its weights are skipped (only the shards holding inverter weights are
downloaded), and it is replaced by MockEmbedder, so only frozen_embeddings are used.

Run from a folder where the embedder name (meta-llama/Llama-2-7b-hf) resolves to a
local copy of its tokenizer + config, as for training.

Example:
    python experiment_scripts/logits/eval_inverter.py \
        --dataset_dir /tmp/dolly_llama2_inv --out_dir /kaggle/working/eval \
        --model ours_ep6=/path/checkpoint-2628 \
        --model authors=jxm/t5-base__llama-7b__one-million-instructions__emb
"""

import argparse
import json
import os
import time
import types

# The embedder is built without weights (see vec2text.models.model_utils).
os.environ["VEC2TEXT_EMBEDDER_NO_WEIGHTS"] = "1"

import datasets  # noqa: E402
import evaluate  # noqa: E402
import huggingface_hub  # noqa: E402
import pandas as pd  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402

import vec2text.models  # noqa: E402
from vec2text.models.config import InversionConfig  # noqa: E402
from vec2text.trainers.base import BaseTrainer  # noqa: E402
from vec2text.utils import MockEmbedder  # noqa: E402

DECODING = {
    # The setting used during training-time evaluation.
    "greedy": {"num_beams": 1, "do_sample": False, "no_repeat_ngram_size": 0},
    "greedy_norep3": {"num_beams": 1, "do_sample": False, "no_repeat_ngram_size": 3},
    "beam4_norep3": {"num_beams": 4, "do_sample": False, "no_repeat_ngram_size": 3},
}
SUMMARY_KEYS = ["bleu_score", "token_set_f1", "exact_match", "rouge_score"]


def load_state(source: str):
    """Returns (config, state_dict without embedder.* weights)."""
    if os.path.isdir(source):
        config = InversionConfig.from_pretrained(source)
        index_path = os.path.join(source, "model.safetensors.index.json")
        if os.path.exists(index_path):
            weight_map = json.load(open(index_path))["weight_map"]
            files = [os.path.join(source, f) for f in sorted(set(weight_map.values()))]
        else:
            files = [os.path.join(source, "model.safetensors")]
    else:
        config = InversionConfig.from_pretrained(source)
        index_path = huggingface_hub.hf_hub_download(
            source, "model.safetensors.index.json"
        )
        weight_map = json.load(open(index_path))["weight_map"]
        needed = sorted(
            {f for k, f in weight_map.items() if not k.startswith("embedder.")}
        )
        print(f"{source}: downloading {len(needed)} of "
              f"{len(set(weight_map.values()))} shards (skipping the LM weights)")
        files = [huggingface_hub.hf_hub_download(source, f) for f in needed]
    state = {}
    for f in files:
        for k, v in safetensors.torch.load_file(f).items():
            if not k.startswith("embedder."):
                state[k] = v
    return config, state


def load_model(source: str, device: str):
    config, state = load_state(source)
    arch = config.architectures[0]
    model = getattr(vec2text.models, arch)(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # T5's embedding matrix is stored once; its tied aliases show up as missing.
    tied = ("shared.weight", "embed_tokens.weight", "lm_head.weight")
    bad_missing = [
        k for k in missing if not k.startswith("embedder.") and not k.endswith(tied)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{source}: missing={bad_missing[:10]} unexpected={unexpected[:10]}"
        )
    del model.embedder  # weightless (meta) LM; never called with frozen embeddings
    model.embedder = MockEmbedder(embedder_dim=model.embedder_dim)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_model(model, data, tokenizer, metrics_ns, batch_size, device):
    """data: dict with frozen_embeddings (Dataset, torch format), input_ids and
    labels (unpadded lists of token ids)."""
    max_length = model.config.max_seq_length
    fe_ds, n = data["frozen_embeddings"], len(data["input_ids"])
    results, texts = {}, {}

    # Eval loss (same per-batch mean as the HF Trainer).
    losses = []
    for i in range(0, n, batch_size):
        lab = data["labels"][i : i + batch_size]
        width = max(len(x) for x in lab)
        labels = torch.tensor([x + [-100] * (width - len(x)) for x in lab])
        out = model(
            input_ids=None,
            attention_mask=None,
            labels=labels.to(device),
            frozen_embeddings=fe_ds[i : i + batch_size]["frozen_embeddings"].to(device),
        )
        losses.append(out.loss.item())
    results["loss"] = sum(losses) / len(losses)

    refs_str = tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)
    refs_ids = [list(x) + [tokenizer.pad_token_id] * (max_length - len(x))
                for x in data["input_ids"]]
    for name, gen_kwargs in DECODING.items():
        t0 = time.time()
        preds_ids = []
        for i in range(0, n, batch_size):
            fe = fe_ds[i : i + batch_size]["frozen_embeddings"].to(device)
            out = model.generate(
                inputs={"frozen_embeddings": fe},
                generation_kwargs={**gen_kwargs, "early_stopping": False,
                                   "max_length": max_length},
            )
            for row in out.cpu().tolist():
                preds_ids.append(row + [tokenizer.pad_token_id] * (max_length - len(row)))
        preds_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        m = BaseTrainer._text_comparison_metrics(
            metrics_ns, preds_ids, preds_str, refs_ids, refs_str
        )
        m = {k: float(v) for k, v in m.items()}
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        texts[name] = (preds_str, list(metrics_ns.bleu_results))
        print(f"  {name:14s} " + "  ".join(f"{k}={m[k]:.4f}" for k in SUMMARY_KEYS)
              + f"  ({m['seconds']}s)", flush=True)
    return results, refs_str, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--model", action="append", required=True,
                   help="name=path_or_hf_repo (repeatable)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_examples", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    raw = datasets.load_from_disk(args.dataset_dir)[args.split]
    plain = raw.with_format(None)
    data = {
        "frozen_embeddings": raw.with_format("torch", columns=["frozen_embeddings"]),
        "input_ids": plain["input_ids"],
        "labels": plain["labels"],
    }
    n = len(raw)
    print(f"{args.split}: {n} examples")
    metrics_ns = types.SimpleNamespace(
        metric_bleu=evaluate.load("sacrebleu"),
        metric_rouge=evaluate.load("rouge"),
        additional_metrics=[],
    )

    all_results, rows = {}, None
    for spec in args.model:
        name, source = spec.split("=", 1)
        print(f"\n=== {name}: {source}", flush=True)
        model = load_model(source, device)
        res, refs_str, texts = evaluate_model(
            model, data, model.tokenizer, metrics_ns, args.batch_size, device
        )
        res["source"] = source
        all_results[name] = res
        if rows is None:
            rows = {"idx": plain["idx"], "true": refs_str}
        for dec, (preds, bleus) in texts.items():
            rows[f"{name}|{dec}"] = preds
            rows[f"{name}|{dec}|bleu"] = bleus
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    print("\n=== Summary (" + args.split + f", n={n})")
    print(f"{'model':12s} {'decoding':14s} {'loss':>6s} " +
          " ".join(f"{k:>12s}" for k in SUMMARY_KEYS))
    for name, res in all_results.items():
        for dec in DECODING:
            m = res[dec]
            print(f"{name:12s} {dec:14s} {res['loss']:6.3f} " +
                  " ".join(f"{m[k]:12.4f}" for k in SUMMARY_KEYS))

    print(f"\n=== First {args.n_examples} examples (beam4_norep3)")
    for i in range(min(args.n_examples, n)):
        print(f"\n[true]     {rows['true'][i]}")
        for name in all_results:
            print(f"[{name}] {rows[f'{name}|beam4_norep3'][i]}")


if __name__ == "__main__":
    main()

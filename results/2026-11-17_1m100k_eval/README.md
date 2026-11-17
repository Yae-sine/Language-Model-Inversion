# T5-base inverter trained on 100k one-million-instructions prompts: evaluation

Kaggle kernel `vec2text-1m-t5base` (version 2). Training files are in `../2026-11-17_t5base_1m100k_train/`.

## Setup
- Inverter: T5-base, `inversion_from_logits_emb`, with Llama-2-7B next-token log-probs as input.
- Data: 100k training prompts from one-million-instructions, plus 1,010 held-out prompts (`om_val`).
- Training: 30,000 steps (9.6 epochs) in 8 h 11 min on one T4, with a constant LR of 2e-4 after warmup.
- Models evaluated:
  - `best_1m`: checkpoint-28500, chosen by eval BLEU.
  - `final_1m`: checkpoint-30000.
  - `dolly_ep40`: the earlier model trained on 14k Dolly prompts (epoch 40).
- Eval sets: `om_val` (1,010 prompts) and `dolly_val` (the same 737 Dolly prompts as in `../2026-11-16_dolly_eval/`).

## Main results (greedy / beam4_norep3 BLEU)

| eval set  | best_1m       | final_1m      | dolly_ep40    | authors' model (from 2026-11-16) |
|-----------|---------------|---------------|---------------|----------------------------------|
| om_val    | 25.4 / 25.8   | 25.0 / 26.0   | 4.4 / 4.6     | not evaluated                    |
| dolly_val | 14.4 / 14.9   | 13.8 / 13.9   | 22.5 / 23.2   | 56.1 / 57.5                      |

`summary.csv` has the full table: loss, token F1, exact match and ROUGE.

## Analysis
1. **In-distribution, the 100k model works far better than the Dolly model** (25.4 vs 4.4 BLEU on om_val).
   - It wins on 91.5% of prompts.
   - Paired mean per-prompt BLEU difference: +21.0, with a 95% bootstrap CI of [19.7, 22.3].
   - It often recovers the template and structure of the prompt ("Answer the following question: …", "Produce a detailed sentence about a restaurant using the following words: …") but not the content words. Exact match is only 0.8%.
2. **On Dolly, it does worse than the 14k Dolly model** (14.4 vs 22.5 BLEU).
   - It loses on 59.6% of prompts.
   - Paired difference: −8.1, with a 95% CI of [−9.9, −6.5].
   - So 100k general prompts do not beat 14k in-domain prompts on Dolly. Prompt distribution matters more than 7x more data at this scale.
3. **Loss and BLEU disagree on Dolly.** The 100k model has a lower eval loss on Dolly (2.87 vs 3.72) but a lower BLEU. The 40-epoch Dolly model is overfit: its predictions are overconfident, which raises the loss, but it has memorised Dolly's phrasing. Use BLEU/F1 rather than loss to compare across training sets.
4. **The run had not converged.**
   - Eval BLEU on om_val rose almost every 1,500 steps: 6.6 → 10.9 → 13.9 → … → 23.8 at 22.5k → 24.9 at 28.5k.
   - Train loss fell to 1.28 while eval loss flattened around 1.64. The gap is growing, but eval BLEU had not degraded yet.
   - With a constant LR, an LR decay or more steps would probably still add a few BLEU points. More data (the authors train on the full one-million-instructions set) is likely the larger lever.
5. **Accuracy drops for long prompts.** On om_val, per-prompt BLEU falls from 29 (≤10 words) to 25 (21–40 words) and 13 (41–80 words). Dolly prompts are shorter (median 9 words vs 18), yet the 100k model scores lower there, which confirms the gap is about domain, not length.
6. **Decoding barely matters.** Beam-4 with no-repeat-3 adds 0.1–1 BLEU over greedy. Best and final checkpoints are within noise of each other (BLEU SEM ≈ 0.7).
7. **We are still far from the authors' model on Dolly** (about 14–23 vs 56 BLEU). Training-set size (100k vs the full dataset) and training length are the most likely gaps. This run does not isolate either one.

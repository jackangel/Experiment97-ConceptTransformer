# Concept Transformer

A small GPT-style language model with one extra idea: an auxiliary **concept head** that fights overfitting by making the model *predict the future, not just the next token*.

## The idea

During normal next-token training, a transformer on a small dataset eventually memorizes the training text and validation loss climbs. This experiment adds a second training objective:

> At every position, a small MLP head must predict the **direction of the mean embedding of the next `N_HORIZON = 6` tokens** — a coarse "concept" of where the text is heading.

Total loss:

```
loss = next_token_cross_entropy + lambda_c * concept_loss
```

## Why it works

- **Un-memorizable target** — the realized future is stochastic given the context, so the concept task can't be crammed by memorization. Gradient spent on it only rewards generalizable structure.
- **Denoised signal** — averaging 6 embeddings cancels token-level noise (~2.4× SNR gain) while keeping the learnable "what is this text about" direction.
- **Dense, self-annealing gradients** — unlike sparse one-hot CE gradients, the cosine-regression gradient is dense and fades as the task is learned.

Result: validation loss stays flat much longer than the identical baseline without the concept head.

## Guardrails included

- **EMA target embeddings** (BYOL/data2vec-style) — targets come from a slow teacher copy, not the live embedding table.
- **Target centering** — removes the corpus-centroid shortcut so the head can't win by predicting a constant vector.
- **Skill metric** — logs how much the head beats a constant predictor (detects a vacuous head).
- **λ warmup + LR schedule** — the concept signal ramps in only after the model has basic language structure.

## Usage

```bash
pip install torch tiktoken
python ConceptTransformer.py
```

- Trains on `input.txt` (a dummy dataset is created if missing), saves `concept_transformer_ckpt.pt` at the end.
- Run again with a checkpoint present → **chat mode**: type a prompt, the model continues it (`/temp 0.8`, `/exit`).
- Delete the checkpoint to retrain.

## Knobs

| Setting | Default | Meaning |
|---|---|---|
| `N_HORIZON` | 6 | How many future tokens form the concept target |
| `LAMBDA_C` | 2.0 | Weight of the concept loss vs next-token loss |
| `CONCEPT_LOSS_TYPE` | `"mse"` | `"infonce"` switches to CPC-style contrastive |
| `USE_EMA_TARGETS` | `True` | Teacher embedding table for stable targets |

See `research/related_work_concept_head.md` for the literature survey behind the design.

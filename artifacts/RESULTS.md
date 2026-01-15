# Phase-1 Results (Recorded Runs)

This document reports the recorded Phase-1 evaluation runs in `neuronx_paper_grade_multiseed_holdout_v2.json`.

**Design (as encoded in the run ledger):**
- **Models:** EleutherAI/gpt-neo-1.3B; facebook/opt-2.7b; EleutherAI/gpt-j-6B
- **Seeds:** 0–4
- **Template holdouts:** 0–3
- **Runs:** 60 total (20 per model; 5 seeds × 4 holdouts)

## Aggregated performance by model (mean ± SD across 20 runs)

| Model | Runs | Test AUROC (mean ± SD) | Perm p(AUC) median [min,max] | Entity-paired acc (mean ± SD) | Mean Δ (mean ± SD) | Selected layers | Selected pooling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI/gpt-j-6B | 20 | 0.9777 ± 0.0220 | 0.007299 [0.002000, 0.062388] | 0.9800 ± 0.0225 | 172.40 ± 66.26 | L8×1, L9×7, L10×8, L11×4 | last×20 |
| facebook/opt-2.7b | 20 | 0.9788 ± 0.0177 | 0.003799 [0.000400, 0.011598] | 0.9710 ± 0.0286 | 49.15 ± 18.14 | L19×13, L20×4, L21×3 | last×20 |
| EleutherAI/gpt-neo-1.3B | 20 | 0.8303 ± 0.0593 | 0.000600 [0.000200, 0.003199] | 0.8230 ± 0.0653 | 56.80 ± 23.48 | L9×9, L11×3, L12×4, L23×4 | last×20 |

## Template holdout breakdown (test AUROC; mean ± SD across 5 seeds)

| Model | Holdout tpl | n | Test AUROC (mean ± SD) | Range [min,max] |
| --- | --- | --- | --- | --- |
| EleutherAI/gpt-j-6B | 0 | 5 | 0.9626 ± 0.0235 | [0.9284, 0.9868] |
| EleutherAI/gpt-j-6B | 1 | 5 | 0.9935 ± 0.0038 | [0.9880, 0.9976] |
| EleutherAI/gpt-j-6B | 2 | 5 | 0.9622 ± 0.0233 | [0.9336, 0.9832] |
| EleutherAI/gpt-j-6B | 3 | 5 | 0.9924 ± 0.0041 | [0.9880, 0.9976] |
| EleutherAI/gpt-neo-1.3B | 0 | 5 | 0.8006 ± 0.0439 | [0.7472, 0.8500] |
| EleutherAI/gpt-neo-1.3B | 1 | 5 | 0.8723 ± 0.0293 | [0.8456, 0.9140] |
| EleutherAI/gpt-neo-1.3B | 2 | 5 | 0.7727 ± 0.0438 | [0.7264, 0.8244] |
| EleutherAI/gpt-neo-1.3B | 3 | 5 | 0.8754 ± 0.0448 | [0.8176, 0.9304] |
| facebook/opt-2.7b | 0 | 5 | 0.9681 ± 0.0120 | [0.9532, 0.9828] |
| facebook/opt-2.7b | 1 | 5 | 0.9914 ± 0.0064 | [0.9836, 0.9992] |
| facebook/opt-2.7b | 2 | 5 | 0.9622 ± 0.0178 | [0.9360, 0.9800] |
| facebook/opt-2.7b | 3 | 5 | 0.9938 ± 0.0047 | [0.9884, 1.0000] |

## Layer and pooling selection behavior

Across all recorded runs:
- `selected_token` is **last** in **60/60** runs.
- Validation-selected layers cluster by model (see “Selected layers” column in the aggregated table).

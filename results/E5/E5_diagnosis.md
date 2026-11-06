# E5 diagnostic interpretation (exploratory)

All test analyses refer to the already inspected October 8 mixtures (one test day).
Macro-8 and mixture exact match are distinct; check both, including the always-all baseline.

## ExtraTrees3 (seed 42)
- Train mixture exact: 1.000; Sept 28 val: 0.144; Oct 8 test: 0.143.
- Test mixture-conditioned oracle exact: 0.316 (not deployable).
- **Hypothesis:** generalization gap already within campaign 130; inspect recipe support and date composition.

## CNN (seed 42)
- Train mixture exact: 0.461; Sept 28 val: 0.256; Oct 8 test: 0.286.
- Test mixture-conditioned oracle exact: 0.143 (not deployable).
- **Hypothesis:** poor fit even on observed training mixtures; inspect class weighting, optimization, capacity and overlapping signals.

## ConvTran (seed 42)
- Train mixture exact: 0.299; Sept 28 val: 0.120; Oct 8 test: 0.418.
- Test mixture-conditioned oracle exact: 0.143 (not deployable).
- **Hypothesis:** poor fit even on observed training mixtures; inspect class weighting, optimization, capacity and overlapping signals.

## Interpretation constraints
- If the same-day holdout is strong but next-day weak, temporal/recipe distribution differences are plausible, not proven sensor drift.
- If 32-example memorization fails, first inspect optimization and implementation; success says nothing about cross-day generalization.
- If 1-NN mixture-only performs well but end-to-end models fail, evaluate class imbalance and mixture gating before claiming physical indistinguishability.
- If AUROC is high but exact match is low, thresholds/ranking should be investigated; do not optimize on test.
- Small per-recipe samples and one mixture test day prevent a population-level confidence interval.
- E5 is an exploratory reanalysis; avoid replacing the original E5 result as if it were a new untouched benchmark.

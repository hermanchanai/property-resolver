# Property resolver results

Frozen on the PHX tune split: Splink threshold 0.11, LLM band 0.02-0.9. Tampa uses the Phoenix model unchanged (no re-estimation, no new threshold).

Cells are precision / recall / F1. *raw* = on the labeled pairs; *pop* = re-weighted by stratum to the full candidate-pair population.

## PHX tune: 150 labeled pairs, 83 matches
Label sources: {'rule_draft': 98, 'manual_draft': 39, 'herman': 13}; excluded (unsure): 0

| Layer | raw P / R / F1 | pop P / R / F1 |
|---|---|---|
| L1 | 1.000 / 0.687 / 0.814 | 1.000 / 0.507 / 0.673 |
| L1+L2 | 0.988 / 0.952 / 0.969 | 0.992 / 0.975 / 0.983 |
| L1+L2+L3 (claude-haiku-5-5) | 1.000 / 0.940 / 0.969 | 1.000 / 0.967 / 0.983 |
| L1+L2+L3 (claude-sonnet-5-5) | 1.000 / 0.940 / 0.969 | 1.000 / 0.967 / 0.983 |

| LLM | alone on unsettled labeled pairs (P / R / F1, n) | residue sent | share of candidate pairs | $ residue | $ all calls |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 0.786 / 0.846 / 0.815, n=93 | 197 of 4110 | 4.8% | 0.0404 | 0.0624 |
| claude-sonnet-5-5 | 0.920 / 0.885 / 0.902, n=93 | 197 of 4110 | 4.8% | 0.4010 | 0.6200 |

## PHX holdout: 50 labeled pairs, 22 matches
Label sources: {'rule_draft': 32, 'manual_draft': 14, 'herman': 4}; excluded (unsure): 0

| Layer | raw P / R / F1 | pop P / R / F1 |
|---|---|---|
| L1 | 1.000 / 0.636 / 0.778 | 1.000 / 0.693 / 0.819 |
| L1+L2 | 0.955 / 0.955 / 0.955 | 0.967 / 0.958 / 0.963 |
| L1+L2+L3 (claude-haiku-5-5) | 0.955 / 0.955 / 0.955 | 0.967 / 0.958 / 0.963 |
| L1+L2+L3 (claude-sonnet-5-5) | 1.000 / 0.955 / 0.977 | 1.000 / 0.958 / 0.979 |

| LLM | alone on unsettled labeled pairs (P / R / F1, n) | residue sent | share of candidate pairs | $ residue | $ all calls |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 0.667 / 1.000 / 0.800, n=36 | 197 of 4110 | 4.8% | 0.0404 | 0.0624 |
| claude-sonnet-5-5 | 0.889 / 1.000 / 0.941, n=36 | 197 of 4110 | 4.8% | 0.4010 | 0.6200 |

## TPA test: 52 labeled pairs, 31 matches
Label sources: {'manual_draft': 50, 'herman': 2}; excluded (unsure): 0

| Layer | raw P / R / F1 | pop P / R / F1 |
|---|---|---|
| L1 | 1.000 / 0.806 / 0.893 | 1.000 / 0.588 / 0.741 |
| L1+L2 | 0.769 / 0.968 / 0.857 | 0.747 / 0.990 / 0.851 |
| L1+L2+L3 (claude-haiku-5-5) | 0.939 / 1.000 / 0.969 | 0.888 / 1.000 / 0.941 |
| L1+L2+L3 (claude-sonnet-5-5) | 0.912 / 1.000 / 0.954 | 0.879 / 1.000 / 0.935 |

| LLM | alone on unsettled labeled pairs (P / R / F1, n) | residue sent | share of candidate pairs | $ residue | $ all calls |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 0.750 / 1.000 / 0.857, n=27 | 92 of 1617 | 5.7% | 0.0173 | 0.0186 |
| claude-sonnet-5-5 | 0.667 / 1.000 / 0.800, n=27 | 92 of 1617 | 5.7% | 0.1834 | 0.2050 |


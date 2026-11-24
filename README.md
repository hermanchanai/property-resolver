# property-resolver

One canonical property ID across four public sources that name the same building differently:
rent-tracker communities (scraped leasing sites), public-filing sold comps, county parcels and
multifamily building permits. Phoenix (PHX) and Tampa (TPA).

## Layers

1. **L1 deterministic:** shared normalized APN, or a normalized address key (house number, street, ZIP5).
2. **L2 probabilistic:** [Splink](https://github.com/moj-analytical-services/splink) on name, address and
   distance over blocked candidate pairs. Threshold frozen on the Phoenix tune split (0.11).
3. **L3 LLM:** only the uncertain band (L2 probability 0.02-0.9) goes to Claude, about 5% of candidate
   pairs, with each answer cached in `labels/`. A full Phoenix residue costs $0.04 with Haiku.

Tampa uses the Phoenix model unchanged (no re-estimation, no new threshold), as a transfer test.

## Results (`results/summary.md`)

Precision / recall on labeled pairs:

| Split | L1 | L1+L2 | L1+L2+L3 (Haiku) | L1+L2+L3 (Sonnet) |
|---|---|---|---|---|
| PHX holdout (50 pairs) | 1.000 / 0.636 | 0.955 / 0.955 | 0.955 / 0.955 | 1.000 / 0.955 |
| TPA test, Phoenix model (52 pairs) | 1.000 / 0.806 | 0.769 / 0.968 | 0.939 / 1.000 | 0.912 / 1.000 |

The LLM layer earns its keep on transfer: the Phoenix-tuned Splink threshold over-merges in Tampa
(precision 0.77), and the LLM on the uncertain band brings it back to 0.94. Labels: rule drafts from
explicit evidence, manual drafts, and pairs reviewed by hand (`labels/<mkt>_herman.csv`, which wins).
The Phoenix sample is frozen with a sealed 25% holdout; Tampa labels are test-only.

Output: `output/property_xref.parquet` (one row per source record: canonical_id, source, source_id,
market, layer, confidence, run_id) and `output/merge_log.csv` (merges / splits versus the previous run).
Large components are real: Florida condo and townhome communities carry one parcel per unit.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
# full rebuild (needs the snapshot and the source database, see Data):
.venv/bin/python -m tools.build_inputs
.venv/bin/python -m tools.candidates --market PHX && .venv/bin/python -m tools.candidates --market TPA
.venv/bin/python -m tools.layer2_splink --market PHX
.venv/bin/python -m tools.layer2_splink --market TPA --model-from models/splink_phx.json
.venv/bin/python -m tools.layer3_llm --market PHX --model claude-haiku-5-5 --band 0.02 0.9 --max-usd 1
.venv/bin/python -m tools.report
.venv/bin/python -m tools.resolve --market PHX --market TPA
```
Each tool's docstring is its SOP. Order matters: rebuilding inputs drops the candidate pairs and L2 scores.

## Data

Inputs come from the frozen snapshot of [text-to-sql-eval](https://github.com/hermanchanai/text-to-sql-eval) (not distributed; see its
README) and from county-parcel tables in a private Postgres database, read through a read-only session
(`tools/db.py` asserts `transaction_read_only = on`; the DSN comes from `SOURCE_ENV_FILE` or a gitignored
`.env`). Neither is in this repo, so a full rebuild needs your own copy. The labels, cached LLM answers,
results and the resolved crosswalk are included.

MIT licensed.

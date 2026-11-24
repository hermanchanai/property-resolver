"""Write results/summary.md + results/summary.json: P/R per layer, per market, per LLM model.

Frozen parameters (chosen on the PHX tune split): threshold 0.11, LLM band 0.02-0.9.
Usage: python -m tools.report
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.evaluate import report

REPO = Path(__file__).resolve().parents[1]
RUNS = [("PHX", "tune"), ("PHX", "holdout"), ("TPA", "test")]
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
THR, BAND = 0.11, (0.02, 0.9)


def fmt(v: dict) -> str:
    return f"{v['precision']:.3f} / {v['recall']:.3f} / {v['f1']:.3f}"


def main() -> None:
    out, md = {}, ["# Property resolver results", "",
                   f"Frozen on the PHX tune split: Splink threshold {THR}, LLM band {BAND[0]}-{BAND[1]}. "
                   "Tampa uses the Phoenix model unchanged (no re-estimation, no new threshold).", "",
                   "Cells are precision / recall / F1. *raw* = on the labeled pairs; *pop* = re-weighted by "
                   "stratum to the full candidate-pair population.", ""]
    for mk, sp in RUNS:
        rs = {m: report(mk, sp, THR, BAND, m) for m in MODELS}
        out[f"{mk}_{sp}"] = rs
        h = rs[MODELS[0]]
        md += [f"## {mk} {sp}: {h['n_labeled']} labeled pairs, {h['n_positive']} matches",
               f"Label sources: {h['label_sources']}; excluded (unsure): {h['excluded_unlabeled']}", "",
               "| Layer | raw P / R / F1 | pop P / R / F1 |", "|---|---|---|"]
        for k in ("L1", "L1+L2"):
            v = h["layers"][k]
            md.append(f"| {k} | {fmt(v['raw'])} | {fmt(v['pop_weighted'])} |")
        for m in MODELS:
            v = rs[m]["layers"]["L1+L2+L3"]
            md.append(f"| L1+L2+L3 ({m}) | {fmt(v['raw'])} | {fmt(v['pop_weighted'])} |")
        md += ["", "| LLM | alone on unsettled labeled pairs (P / R / F1, n) | residue sent | share of candidate pairs | $ residue | $ all calls |",
               "|---|---|---|---|---|---|"]
        for m in MODELS:
            a, r = rs[m]["llm_alone_on_unsettled"], rs[m]["llm_residue"]
            md.append(f"| {m} | {fmt(a)}, n={a['n']} | {r['sent_to_llm']} of {r['candidate_pairs']} | "
                      f"{r['share']:.1%} | {r['usd_for_residue']:.4f} | {r['usd_all_calls_incl_eval']:.4f} |")
        md.append("")
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    (REPO / "results/summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

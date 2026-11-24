"""Draw the stratified labeling sample from `pairs` and draft a label for each pair.

Strata are mutually exclusive, assigned in this order (first hit wins):
  apn_conflict     both sides have APNs, none shared, and (<= 150 m or name_sim >= 0.6)
  apn_far          share an APN but > 150 m apart (does the APN key hold at distance?)
  easy             <= 150 m and name_sim >= 0.6
  near_dissimilar  <= 150 m and name_sim < 0.6   (the 749-pair problem from the plan)
  near_miss        150-500 m, half drawn from name_sim >= 0.5
  other            no usable distance (city-centroid geocode) -> address or APN block only
A quarter of each stratum is sealed as holdout (never used to tune a threshold).
Population counts per stratum are saved so metrics can be re-weighted to the full pair set.

Draft labels come from explicit evidence rules (draft_label); 'uncertain' drafts go to Herman.

Usage: python -m tools.sample_labels --market PHX --out labels/phx_sample.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED = 20261122
QUOTA = {"apn_conflict": 20, "apn_far": 20, "easy": 50, "near_dissimilar": 70, "near_miss": 40, "other": 10}
HOLDOUT_SHARE = 0.25


def stratum(p) -> str:
    d, s = p.distance_m, p.name_sim
    if p.apn_conflict and ((pd.notna(d) and d <= 150) or s >= 0.6):
        return "apn_conflict"
    if p.apn_shared > 0 and pd.notna(d) and d > 150:
        return "apn_far"
    if pd.isna(d):
        return "other"
    if d <= 150:
        return "easy" if s >= 0.6 else "near_dissimilar"
    return "near_miss"


def draft_label(p) -> tuple[str, str]:
    """(draft, why). draft in {'match', 'non_match', 'uncertain'}."""
    strong_name = p.name_match or p.name_sim >= 0.85 or p.county_name_sim >= 0.85 or p.t_county_name_sim >= 0.85
    addr = p.addr_rel == "exact"
    if p.phase_conflict and strong_name:
        return "uncertain", f"same name stem, phase markers differ ({p.phase_r} vs {p.phase_t})"
    if p.apn_shared > 0 and p.r_apn_conf in ("high", "medium") and (strong_name or addr):
        return "match", "shared APN + " + ("name" if strong_name else "address")
    if strong_name and addr:
        return "match", "name + exact address"
    if strong_name and pd.notna(p.distance_m) and p.distance_m <= 100:
        return "match", f"name + {p.distance_m:.0f} m"
    weak_name = p.name_sim < 0.3 and p.county_name_sim < 0.5 and p.t_county_name_sim < 0.5
    if weak_name and p.addr_rel in ("none", "street") and p.apn_shared == 0:
        return "non_match", f"names differ, addresses differ ({p.addr_rel}), no shared APN"
    if weak_name and p.apn_conflict and not addr:
        return "non_match", "names differ, APNs disagree"
    return "uncertain", "mixed evidence"


def sample(con, market: str, scale: float = 1.0, test_only: bool = False) -> tuple[pd.DataFrame, dict]:
    df = con.execute("select * from pairs where market=?", [market]).df()
    df["stratum"] = df.apply(stratum, axis=1)
    pop = df.stratum.value_counts().to_dict()
    picks = []
    for st, q in QUOTA.items():
        q = round(q * scale)
        g = df[df.stratum == st]
        if st == "near_miss":
            hi, lo = g[g.name_sim >= 0.5], g[g.name_sim < 0.5]
            k = min(q // 2, len(hi))
            part = pd.concat([hi.sample(k, random_state=SEED), lo.sample(min(q - k, len(lo)), random_state=SEED)])
        else:
            part = g.sample(min(q, len(g)), random_state=SEED)
        part = part.sample(frac=1, random_state=SEED)     # shuffle, then seal the first quarter
        if test_only:     # generalization market: nothing is tuned here, every label is a test label
            part = part.assign(split="test")
        else:
            n_hold = round(len(part) * HOLDOUT_SHARE)
            part = part.assign(split=["holdout"] * n_hold + ["tune"] * (len(part) - n_hold))
        picks.append(part)
    s = pd.concat(picks).reset_index(drop=True)
    drafts = s.apply(draft_label, axis=1, result_type="expand")
    s["draft"], s["draft_why"] = drafts[0], drafts[1]
    return s, {"population": pop, "sampled": s.stratum.value_counts().to_dict(), "seed": SEED,
               "scale": scale, "test_only": test_only}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()
    con = duckdb.connect(str(REPO / "data/inputs.duckdb"), read_only=True)
    s, meta = sample(con, a.market, a.scale, a.test_only)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the sample is frozen once drawn (delete it deliberately to redraw)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, index=False)
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(s.groupby(["stratum", "draft"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

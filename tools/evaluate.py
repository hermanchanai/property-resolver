"""Precision / recall per layer on the labeled sample.

Label precedence: Herman's answer (labels/<mkt>_herman.csv) > manual draft (labels/<mkt>_drafts_manual.csv,
keyed by sample row) > rule draft (column `draft` in the sample). Pairs still 'uncertain' or labeled
'unsure' are excluded and counted. Every report states how many labels Herman confirmed.

Layers (cumulative, each adds positives to the one before):
  L1      shared normalized APN, rent-side parcel match high/medium confidence
  L1+L2   L1 OR Splink match_probability >= threshold (threshold chosen on the TUNE split only)
  L1+L2+L3  same, but pairs inside the uncertain band take the LLM verdict (labels/<mkt>_llm.csv)

Metrics are reported two ways: raw on the labeled pairs, and re-weighted to the full candidate
population by stratum (the sample over-represents hard strata on purpose).

Usage: python -m tools.evaluate --market PHX [--split tune|holdout|all] [--threshold 0.9]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_labels(market: str) -> pd.DataFrame:
    m = market.lower()
    s = pd.read_csv(REPO / f"labels/{m}_sample.csv")
    s["row"] = range(len(s))
    s["label"], s["label_source"] = s.draft, "rule_draft"
    man = REPO / f"labels/{m}_drafts_manual.csv"
    if man.exists():
        md = pd.read_csv(man).set_index("i")
        hit = s.row.isin(md.index)
        s.loc[hit, "label"] = s.loc[hit, "row"].map(md.draft)
        unsure = s.row.map(md.sure).eq("n") & hit
        s.loc[hit, "label_source"] = "manual_draft"
        s.loc[unsure, ["label", "label_source"]] = ["unsure", "manual_unsure"]   # waits for Herman
    her = REPO / f"labels/{m}_herman.csv"
    if her.exists():
        h = pd.read_csv(her).drop_duplicates("pair_id", keep="last").set_index("pair_id")
        hit = s.pair_id.isin(h.index)
        s.loc[hit, "label"] = s.loc[hit, "pair_id"].map(h.label)
        s.loc[hit, "label_source"] = "herman"
    return s


def prf(y: pd.Series, p: pd.Series, w: pd.Series | None = None) -> dict:
    w = pd.Series(1.0, index=y.index) if w is None else w
    tp = float((w * (y & p)).sum())
    fp = float((w * (~y & p)).sum())
    fn = float((w * (y & ~p)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": round(tp, 1), "fp": round(fp, 1), "fn": round(fn, 1)}


def frame(market: str, llm_model: str | None = None) -> tuple[pd.DataFrame, dict]:
    s = load_labels(market)
    con = duckdb.connect(str(REPO / "data/inputs.duckdb"), read_only=True)
    l2 = con.execute("select community_id, property_id, match_probability from l2_scores where market=?",
                     [market]).df()
    # layer-1 features come from the CURRENT pairs table: the sample CSV froze them when it was drawn
    cur = con.execute("select pair_id, apn_shared, r_apn_conf from pairs where market=?", [market]).df()
    con.close()
    s = s.drop(columns=["apn_shared", "r_apn_conf"]).merge(cur, on="pair_id", how="left")
    s["apn_shared"] = s.apn_shared.fillna(0)
    s = s.merge(l2, on=["community_id", "property_id"], how="left")
    s["match_probability"] = s.match_probability.fillna(0.0)
    s["l1"] = (s.apn_shared > 0) & s.r_apn_conf.isin(["high", "medium"])
    if llm_model:
        llm = REPO / f"labels/{market.lower()}_llm_{llm_model}.jsonl"
        lv = pd.read_json(llm, lines=True).drop_duplicates("pair_id", keep="last").set_index("pair_id")
        s["llm_same"] = s.pair_id.map(lv.same_property)
    meta = json.loads((REPO / f"labels/{market.lower()}_sample.meta.json").read_text())
    pop, smp = meta["population"], s.stratum.value_counts().to_dict()
    s["w"] = s.stratum.map(lambda st: pop.get(st, 0) / smp[st])
    return s, meta


def layer_preds(s: pd.DataFrame, thr: float, band: tuple[float, float] | None) -> dict[str, pd.Series]:
    out = {"L1": s.l1, "L1+L2": s.l1 | (s.match_probability >= thr)}
    if band and "llm_same" in s:
        inband = ~s.l1 & s.match_probability.between(*band) & s.llm_same.notna()
        l3 = out["L1+L2"].copy()
        l3[inband] = s.loc[inband, "llm_same"].astype(bool)
        out["L1+L2+L3"] = l3
    return out


def choose_threshold(s: pd.DataFrame) -> float:
    t = s[(s.split == "tune") & s.label.isin(["match", "non_match"])]
    y = t.label == "match"
    best = max(np.arange(0.05, 1.0, 0.01),
               key=lambda th: (prf(y, t.l1 | (t.match_probability >= th))["f1"], th))
    return round(float(best), 2)


def report(market: str, split: str, thr: float | None, band: tuple[float, float] | None,
           llm_model: str | None = None) -> dict:
    s, meta = frame(market, llm_model)
    thr = choose_threshold(s) if thr is None else thr
    lab = s[s.label.isin(["match", "non_match"])]
    if split != "all":
        lab = lab[lab.split == split]
    y = lab.label == "match"
    out = {"market": market, "split": split, "threshold": thr, "band": band, "llm_model": llm_model,
           "n_labeled": int(len(lab)), "n_positive": int(y.sum()),
           "label_sources": lab.label_source.value_counts().to_dict(),
           "excluded_unlabeled": int((~s.label.isin(["match", "non_match"])).sum()), "layers": {}}
    for name, p in layer_preds(lab, thr, band).items():
        out["layers"][name] = {"raw": prf(y, p), "pop_weighted": prf(y, p, lab.w)}
    if band and llm_model:
        # the LLM judged on its own, on every labeled pair layer 1 did not settle (diagnostic)
        u = lab[~lab.l1 & lab.llm_same.notna()]
        out["llm_alone_on_unsettled"] = prf(u.label == "match", u.llm_same.astype(bool))
        out["llm_alone_on_unsettled"]["n"] = int(len(u))
        from tools.layer3_llm import PRICES, select_pairs
        res = select_pairs(market, band, labeled_only=False)
        con = duckdb.connect(str(REPO / "data/inputs.duckdb"), read_only=True)
        n_pairs = con.execute("select count(*) from pairs where market=?", [market]).fetchone()[0]
        con.close()
        calls = pd.read_json(REPO / f"labels/{market.lower()}_llm_{llm_model}.jsonl", lines=True)
        rc = calls[calls.pair_id.isin(res.pair_id)]
        pin, pout = PRICES[llm_model]
        out["llm_residue"] = {"candidate_pairs": int(n_pairs), "sent_to_llm": int(len(res)),
                              "share": round(len(res) / n_pairs, 4),
                              "usd_for_residue": round(float((rc.input_tokens * pin + rc.output_tokens * pout).sum()) / 1e6, 4),
                              "usd_all_calls_incl_eval": round(float((calls.input_tokens * pin + calls.output_tokens * pout).sum()) / 1e6, 4),
                              "labeled_pairs_in_band": int((~lab.l1 & lab.match_probability.between(*band)).sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--split", default="tune", choices=["tune", "holdout", "test", "all"])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--band", type=float, nargs=2)
    ap.add_argument("--llm-model")
    a = ap.parse_args()
    print(json.dumps(report(a.market, a.split, a.threshold, tuple(a.band) if a.band else None, a.llm_model),
                     indent=2))


if __name__ == "__main__":
    main()

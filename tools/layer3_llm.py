"""Layer 3: LLM tiebreak for the residue that layers 1-2 cannot settle.

Only pairs NOT settled by layer 1 and with a Splink match_probability inside the band go to the model.
Each call sees both records side by side (names, addresses, distance, county assessor names, county
vs sold-comp unit counts and years) and must answer JSON {same_property, confidence, reason}.

Model ids and prices were checked in the Anthropic docs (anthropic-dev-docs MCP, 2026-11-22):
  claude-haiku-5-5  $0.10 in / $0.50 out per MTok (prompts <= 100k tokens)
  claude-sonnet-5-5 $2 in / $10 out per MTok
The key is read in-process from SOURCE_ENV_FILE (ANTHROPIC_API_KEY = the "BFR Intel" key per that
file's comment); its label is logged, the value never is. Every response is cached in
labels/<mkt>_llm_<model>.jsonl so a re-run costs nothing for pairs already judged.

Usage: python -m tools.layer3_llm --market PHX --model claude-haiku-5-5 --band 0.02 0.9 [--labeled-only]
       [--max-usd 2.0] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import duckdb
import pandas as pd
from dotenv import dotenv_values

from tools.db import DEFAULT_SOURCE_ENV

log = logging.getLogger("layer3")
REPO = Path(__file__).resolve().parents[1]
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}   # USD per MTok in/out
KEY_LABEL = "BFR Intel (ANTHROPIC_API_KEY in the source .env)"
MAX_TOKENS = 1500

SYSTEM = ("You decide whether two real-estate records describe the SAME multifamily or build-to-rent "
          "property. Record A comes from a rent tracker that scrapes leasing websites (current marketing "
          "name). Record B is a sold comp from public deed filings (name at the time of sale). Properties "
          "are often renamed after a sale, so different names alone do not mean different properties; "
          "two neighbours on the same street, or two phases of one development sold separately, are "
          "different properties. County assessor names, unit counts and year built come from public parcel "
          "records. Answer with JSON only: "
          '{"same_property": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}')


def _fmt(v) -> str:
    return "unknown" if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else str(v)


def prompt(p) -> str:
    dom = re.sub(r"^https?://(www\.)?", "", _fmt(p.r_url)).split("/")[0]
    return "\n".join([
        "Record A (rent tracker):",
        f"  name: {_fmt(p.r_name)}", f"  address: {_fmt(p.r_address)}", f"  website domain: {dom}",
        f"  county assessor complex name on its matched parcel: {_fmt(p.county_names)}",
        f"  county units / year built: {_fmt(p.r_county_units)} / {_fmt(p.r_county_year)}",
        "Record B (sold comp):",
        f"  name: {_fmt(p.t_name)}", f"  address: {_fmt(p.t_address)}",
        f"  county assessor complex name on its parcels: {_fmt(p.t_county_names)}",
        f"  units / year built: {_fmt(p.t_units)} / {_fmt(p.t_year)}",
        f"Distance between the two geocodes: {_fmt(p.distance_m)} m",
        "Do A and B describe the same property?"])


def parse(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    d = json.loads(m.group(0) if m else t)
    return {"same_property": bool(d["same_property"]), "confidence": float(d.get("confidence", 0)),
            "reason": str(d.get("reason", ""))[:300]}


def client() -> anthropic.Anthropic:
    env = Path(os.environ.get("SOURCE_ENV_FILE", DEFAULT_SOURCE_ENV))
    key = dotenv_values(env).get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(f"ANTHROPIC_API_KEY missing from {env}")
    log.info("Anthropic key: %s", KEY_LABEL)
    return anthropic.Anthropic(api_key=key, max_retries=6)


def call(c: anthropic.Anthropic, model: str, p) -> dict:
    for attempt in range(4):
        try:
            r = c.messages.create(model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
                                  output_config={"effort": "low"},
                                  messages=[{"role": "user", "content": prompt(p)}])
            text = "".join(b.text for b in r.content if b.type == "text")
            out = {"pair_id": p.pair_id, "model": model, "input_tokens": r.usage.input_tokens,
                   "output_tokens": r.usage.output_tokens, "raw": text[:600]}
            try:
                out.update(parse(text))
            except (ValueError, KeyError) as e:
                out.update({"same_property": None, "confidence": None, "reason": f"PARSE_ERROR {e}"})
            return out
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("rate-limited 4 times")


def select_pairs(market: str, band: tuple[float, float], labeled_only: bool,
                 include_labeled: bool = False) -> pd.DataFrame:
    """Band pairs not settled by layer 1. include_labeled adds every labeled pair not settled by layer 1
    whatever its score (evaluation of the model itself; not part of the production residue)."""
    con = duckdb.connect(str(REPO / "data/inputs.duckdb"), read_only=True)
    df = con.execute("""
        select p.*, coalesce(s.match_probability, 0) match_probability from pairs p
        left join l2_scores s using (market, community_id, property_id)
        where p.market = ? and not (p.apn_shared > 0 and p.r_apn_conf in ('high','medium'))""",
                     [market]).df()
    con.close()
    lab = pd.read_csv(REPO / f"labels/{market.lower()}_sample.csv", usecols=["pair_id"])
    inband = df.match_probability.between(*band)
    keep = inband | (df.pair_id.isin(lab.pair_id) if include_labeled else False)
    df = df[keep]
    if labeled_only:
        df = df[df.pair_id.isin(lab.pair_id)]
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--model", required=True, choices=list(PRICES))
    ap.add_argument("--band", type=float, nargs=2, required=True)
    ap.add_argument("--labeled-only", action="store_true")
    ap.add_argument("--include-labeled", action="store_true")
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cache = REPO / f"labels/{a.market.lower()}_llm_{a.model}.jsonl"
    done = {json.loads(l)["pair_id"] for l in cache.open()} if cache.exists() else set()
    todo = select_pairs(a.market, tuple(a.band), a.labeled_only, a.include_labeled)
    todo = todo[~todo.pair_id.isin(done)]
    pin, pout = PRICES[a.model]
    est = len(todo) * (700 * pin + 400 * pout) / 1e6
    log.info("%d pairs to judge (%d cached); estimate $%.3f with %s", len(todo), len(done), est, a.model)
    if a.dry_run:
        print(prompt(next(todo.itertuples())) if len(todo) else "nothing to do")
        return
    if est > a.max_usd:
        raise SystemExit(f"estimate ${est:.2f} exceeds --max-usd {a.max_usd}")
    c = client()
    spent = 0.0
    with cache.open("a") as f, ThreadPoolExecutor(max_workers=4) as ex:
        for out in ex.map(lambda p: call(c, a.model, p), todo.itertuples()):
            f.write(json.dumps(out) + "\n")
            f.flush()
            spent += (out["input_tokens"] * pin + out["output_tokens"] * pout) / 1e6
    rows = [json.loads(l) for l in cache.open()]
    tot = sum((r["input_tokens"] * pin + r["output_tokens"] * pout) / 1e6 for r in rows)
    log.info("this run $%.4f; cache total %d calls $%.4f; parse errors %d", spent, len(rows), tot,
             sum(r["same_property"] is None for r in rows))


if __name__ == "__main__":
    main()

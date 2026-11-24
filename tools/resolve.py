"""Resolve every record to one canonical property ID and write the cross-reference table.

Members: rent communities, sold properties, county parcels, MF permits.
Edges, applied in priority order (union-find):
  1. rent <-> sold     layer 1 (shared APN), else layer 2 (Splink p >= threshold), with layer 3 (LLM)
                       overriding pairs inside the uncertain band. Each sold property joins at most ONE
                       rent community (the strongest edge).
  2. county parcels    Maricopa parcels sharing an economic unit
  3. rent  -> parcel   county-parcel match, high/medium confidence, parcel not shared with another community
  4. sold  -> parcel   same normalized APN
  5. permit -> parcel  same normalized APN (Maricopa only; FL permits carry no matching key)
Chaining guard: an edge that would put two DIFFERENT rent communities into one component is refused and
logged (two phases of one development stay two properties unless a rent<->sold edge says otherwise).

Canonical ID = 'prop_' + sha1(anchor)[:12], anchor = the member with the lowest (source rank, source_id):
rent community, then sold property, then parcel, then permit. A run diffs itself against the previous
xref and logs merges (one new id gathers members of several old ids) and splits (the reverse).

Output: output/property_xref.parquet (+ .csv), output/merge_log.csv, output/resolve_run.json
Usage: python -m tools.resolve --market PHX --market TPA [--threshold 0.11] [--band 0.02 0.9]
       [--llm-model claude-haiku-5-5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output"
RANK = {"rent_community": 0, "sold_property": 1, "county_parcel": 2, "permit": 3}


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}
        self.rents: dict[str, set] = {}

    def add(self, x: str):
        if x not in self.p:
            self.p[x] = x
            self.rents[x] = {x} if x.startswith("rent_community|") else set()

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        """False (and no merge) when both sides already hold different rent communities."""
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.rents[ra] and self.rents[rb] and self.rents[ra] != self.rents[rb]:
            return False
        self.p[rb] = ra
        self.rents[ra] |= self.rents.pop(rb)
        return True


def rent_sold_edges(con, market: str, thr: float, band: tuple[float, float], llm_model: str | None):
    df = con.execute("""
        select p.community_id, p.property_id, p.apn_shared, p.r_apn_conf,
               coalesce(s.match_probability, 0) prob
        from pairs p left join l2_scores s using (market, community_id, property_id)
        where p.market = ?""", [market]).df()
    df["l1"] = (df.apn_shared > 0) & df.r_apn_conf.isin(["high", "medium"])
    df["layer"], df["conf"] = None, df.prob
    df.loc[df.l1, ["layer", "conf"]] = ["L1_apn", 1.0]
    l2 = ~df.l1 & (df.prob >= thr)
    df.loc[l2, "layer"] = "L2_splink"
    if llm_model:
        f = REPO / f"labels/{market.lower()}_llm_{llm_model}.jsonl"
        if f.exists():
            lv = pd.read_json(f, lines=True).drop_duplicates("pair_id", keep="last")
            lv[["m", "c", "p"]] = lv.pair_id.str.split(":", n=2, expand=True)
            lv["p"] = lv.p.astype(int)
            v = {(c, p): (s, conf) for c, p, s, conf in zip(lv.c, lv.p, lv.same_property, lv.confidence)}
            inband = ~df.l1 & df.prob.between(*band)
            for i in df.index[inband]:
                hit = v.get((df.at[i, "community_id"], df.at[i, "property_id"]))
                if hit and hit[0] is not None:
                    df.at[i, "layer"] = "L3_llm" if hit[0] else None
                    df.at[i, "conf"] = float(hit[1] or 0) if hit[0] else df.at[i, "conf"]
    e = df[df.layer.notna()].copy()
    e["rank"] = e.layer.map({"L1_apn": 0, "L2_splink": 1, "L3_llm": 1})
    # each sold property keeps only its strongest rent edge
    e = e.sort_values(["property_id", "rank", "conf"], ascending=[True, True, False])
    dropped = int(e.duplicated("property_id").sum())
    return e.drop_duplicates("property_id"), dropped


def resolve(markets: list[str], thr: float, band: tuple[float, float], llm_model: str | None) -> dict:
    con = duckdb.connect(str(REPO / "data/inputs.duckdb"), read_only=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows, stats = [], {}
    for mk in markets:
        uf, layer_of, conf_of = UF(), {}, {}
        refused = Counter()

        def key(src, sid):
            return f"{src}|{mk}|{sid}"

        def link(a, b, layer, conf, kind):
            if uf.union(a, b):          # first edge that touches a member records how it joined
                for x in (a, b):
                    layer_of.setdefault(x, layer)
                    conf_of.setdefault(x, conf)
            else:
                refused[kind] += 1

        for (cid,) in con.execute("select community_id from rent_community where market=?", [mk]).fetchall():
            uf.add(key("rent_community", cid))
        for (pid,) in con.execute("select property_id from sold_property where market=?", [mk]).fetchall():
            uf.add(key("sold_property", pid))

        rs, dropped = rent_sold_edges(con, mk, thr, band, llm_model)
        for r in rs.itertuples():
            a, b = key("rent_community", r.community_id), key("sold_property", r.property_id)
            link(a, b, r.layer, round(float(r.conf), 4), "rent_sold")
            layer_of[b] = r.layer
            conf_of[b] = round(float(r.conf), 4)

        parcels = con.execute("select county, apn_norm, econ_unit from county_parcel where market=? "
                              "and apn_norm is not null", [mk]).df()
        by_econ = defaultdict(list)
        for c, a, e in zip(parcels.county, parcels.apn_norm, parcels.econ_unit):
            if isinstance(e, str) and e:
                by_econ[(c, e)].append(a)
        for (c, _), apns in by_econ.items():
            for a in apns[1:]:
                link(key("county_parcel", f"{c}:{apns[0]}"), key("county_parcel", f"{c}:{a}"),
                     "parcel_econ_unit", 1.0, "econ_unit")

        cp = con.execute("""select community_id, county, apn_norm, match_confidence from community_parcel
            where market=? and match_confidence in ('high','medium') and not shared_parcel
              and apn_norm is not null""", [mk]).fetchall()
        for cid, c, a, mc in cp:
            link(key("rent_community", cid), key("county_parcel", f"{c}:{a}"), "rent_parcel_match",
                 1.0 if mc == "high" else 0.8, "rent_parcel")
        sp = con.execute("""select sa.property_id, sa.county, sa.apn_norm from sold_property_apn sa
            where sa.market=? and sa.apn_norm is not null""", [mk]).fetchall()
        known_parcels = {f"{c}:{a}" for c, a in zip(parcels.county, parcels.apn_norm)}
        for pid, c, a in sp:
            if f"{c}:{a}" in known_parcels or key("county_parcel", f"{c}:{a}") in uf.p:
                link(key("sold_property", pid), key("county_parcel", f"{c}:{a}"), "sold_parcel_apn", 1.0,
                     "sold_parcel")
        if mk == "PHX":
            pm = con.execute("""select permit_id, apn_norm from permit where market=? and apn_norm is not null""",
                             [mk]).fetchall()
            for pid, a in pm:
                pk = key("county_parcel", f"maricopa_az:{a}")
                if pk in uf.p:
                    link(key("permit", pid), pk, "permit_parcel_apn", 1.0, "permit_parcel")

        comps = defaultdict(list)
        for x in uf.p:
            comps[uf.find(x)].append(x)
        sizes = Counter()
        for members in comps.values():
            srcs = {m.split("|")[0] for m in members}
            if not srcs & {"rent_community", "sold_property"}:
                continue          # parcel/permit-only groups are not properties we track
            anchor = min(members, key=lambda m: (RANK[m.split("|")[0]], m))
            cid = "prop_" + hashlib.sha1(anchor.encode()).hexdigest()[:12]
            sizes[len(members)] += 1
            for m in members:
                src, _, sid = m.split("|", 2)
                rows.append({"canonical_id": cid, "source": src, "source_id": sid, "market": mk,
                             "layer": "anchor" if m == anchor else layer_of.get(m, "anchor"),
                             "confidence": 1.0 if m == anchor else conf_of.get(m, 1.0), "run_id": run_id})
        x = pd.DataFrame([r for r in rows if r["market"] == mk])
        stats[mk] = {"canonical_properties": int(x.canonical_id.nunique()),
                     "members_by_source": x.source.value_counts().to_dict(),
                     "rent_sold_edges_by_layer": rs.layer.value_counts().to_dict(),
                     "sold_with_extra_rent_edges_dropped": dropped,
                     "refused_merges_chaining_guard": dict(refused),
                     "rent_linked_to_sold": int(x[x.source == "sold_property"].canonical_id.isin(
                         x[x.source == "rent_community"].canonical_id).sum()),
                     "component_size_max": max(sizes) if sizes else 0}
    con.close()
    return {"run_id": run_id, "rows": pd.DataFrame(rows), "stats": stats,
            "params": {"threshold": thr, "band": band, "llm_model": llm_model, "markets": markets}}


def merge_log(new: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    if prev is None or prev.empty:
        return pd.DataFrame(columns=["event", "canonical_id", "other_ids", "n_members"])
    k = ["source", "source_id", "market"]
    j = new.merge(prev[k + ["canonical_id"]].rename(columns={"canonical_id": "prev_id"}), on=k, how="left")
    out = []
    for cid, g in j.groupby("canonical_id"):
        olds = sorted(set(g.prev_id.dropna()))
        if len(olds) > 1:
            out.append({"event": "merge", "canonical_id": cid, "other_ids": ";".join(olds), "n_members": len(g)})
    for pid, g in j.dropna(subset=["prev_id"]).groupby("prev_id"):
        news = sorted(set(g.canonical_id))
        if len(news) > 1:
            out.append({"event": "split", "canonical_id": pid, "other_ids": ";".join(news), "n_members": len(g)})
    return pd.DataFrame(out, columns=["event", "canonical_id", "other_ids", "n_members"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="append", required=True)
    ap.add_argument("--threshold", type=float, default=0.11)
    ap.add_argument("--band", type=float, nargs=2, default=[0.02, 0.9])
    ap.add_argument("--llm-model", default="claude-haiku-5-5")
    a = ap.parse_args()
    res = resolve(a.market, a.threshold, tuple(a.band), a.llm_model)
    OUT.mkdir(exist_ok=True)
    prev_path = OUT / "property_xref.parquet"
    prev = pd.read_parquet(prev_path) if prev_path.exists() else None
    ml = merge_log(res["rows"], prev)
    df = res["rows"].sort_values(["market", "canonical_id", "source", "source_id"]).reset_index(drop=True)
    df.to_parquet(prev_path, index=False)
    df.to_csv(OUT / "property_xref.csv", index=False)
    ml.to_csv(OUT / "merge_log.csv", index=False)
    meta = {"run_id": res["run_id"], "params": res["params"], "stats": res["stats"],
            "merge_log": ml.event.value_counts().to_dict(), "rows": len(df)}
    (OUT / "resolve_run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()

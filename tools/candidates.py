"""Candidate pairs (rent community x sold property) with comparison features.

Blocking (union of three keys; a pair enters if ANY holds):
  near   both coordinates street-level or better and haversine distance <= NEAR_M
  addr   same parsed (house number, street) key   -- catches city-centroid geocodes
  apn    the community's county-parcel match shares a normalized APN with the sold property

Writes table `pairs` into data/inputs.duckdb (replaced each run) and prints the blocking-recall check:
of the pairs that share an APN, how many each geometric key alone would have caught.

Usage: python -m tools.candidates --market PHX
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import duckdb
import pandas as pd

from tools.normalize import address_relation, name_match, name_sim, parse_address, phase_marker

log = logging.getLogger("candidates")
REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data/inputs.duckdb"
NEAR_M = 500.0
IMPRECISE = {"city", "zip_approx", None}


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def _str_list(v) -> list[str]:
    try:
        return [x for x in v if isinstance(x, str)]
    except TypeError:          # NA scalar
        return []


def build_pairs(con: duckdb.DuckDBPyConnection, market: str) -> pd.DataFrame:
    r = con.execute("select * from rent_community where market=?", [market]).df()
    t = con.execute("select * from sold_property where market=?", [market]).df()
    r["addr"] = r.address.map(parse_address)
    t["addr"] = t.apply(lambda x: parse_address(", ".join(v for v in [x.address, x.city, x.zip] if isinstance(v, str) and v)), axis=1)

    # R -> APNs (county-parcel match), R -> county complex name / units / year
    rp = con.execute("""select cp.community_id, list(distinct cp.apn_norm) apns,
            any_value(cp.match_confidence) conf, any_value(cp.county_units) cu,
            any_value(cp.year_built_min) cy, bool_or(cp.shared_parcel) shared,
            list(distinct pc.complex_name) filter (where pc.complex_name is not null) cnames
        from community_parcel cp left join county_parcel pc
          on pc.market = cp.market and pc.apn_norm = cp.apn_norm
        where cp.market=? group by 1""", [market]).df().set_index("community_id")
    tp = con.execute("""select property_id, list(distinct apn_norm) apns from sold_property_apn
        where market=? and apn_norm is not null group by 1""", [market]).df().set_index("property_id")
    t_apns = {pid: set(a) for pid, a in tp.apns.items()}
    tcn = con.execute("""select sa.property_id, list(distinct pc.complex_name) cnames from sold_property_apn sa
        join county_parcel pc on pc.market = sa.market and pc.apn_norm = sa.apn_norm
        where sa.market=? and pc.complex_name is not null group by 1""", [market]).df()
    t_cnames = {pid: _str_list(v) for pid, v in zip(tcn.property_id, tcn.cnames)}
    r_apns = {cid: set(a) for cid, a in rp.apns.items()}

    # spatial candidates via a coarse grid join in DuckDB, exact distance in Python
    near = con.execute("""select r.community_id, t.property_id from rent_community r join sold_property t
          on r.market = t.market and abs(r.lat - t.lat) < 0.0055 and abs(r.lng - t.lng) < 0.0065
        where r.market = ? and coalesce(t.geo_precision,'') not in ('city','zip_approx')""", [market]).fetchall()
    keys: dict[tuple, set] = {}
    for k in near:
        keys.setdefault(k, set()).add("near")
    r_by_addr: dict[str, list] = {}
    for cid, a in zip(r.community_id, r.addr):
        if a.key:
            r_by_addr.setdefault(a.key, []).append(cid)
    for pid, a in zip(t.property_id, t.addr):
        for cid in r_by_addr.get(a.key, []) if a.key else []:
            keys.setdefault((cid, pid), set()).add("addr")
    apn_to_t: dict[str, list] = {}
    for pid, s in t_apns.items():
        for a in s:
            apn_to_t.setdefault(a, []).append(pid)
    for cid, s in r_apns.items():
        conf = rp.conf.get(cid) if cid in rp.index else None
        for a in s:
            for pid in apn_to_t.get(a, []):
                keys.setdefault((cid, pid), set()).add("apn" if conf in ("high", "medium") else "apn_low")

    R = r.set_index("community_id")
    T = t.set_index("property_id")
    rows = []
    for (cid, pid), why in keys.items():
        if cid not in R.index or pid not in T.index:   # matched community since superseded / ungeocoded
            continue
        a, b = R.loc[cid], T.loc[pid]
        precise = (b.geo_precision not in IMPRECISE) and pd.notna(a.lat) and pd.notna(b.lat)
        d = haversine_m(a.lat, a.lng, b.lat, b.lng) if precise else None
        if why == {"near"} and (d is None or d > NEAR_M):
            continue
        rinfo = rp.loc[cid] if cid in rp.index else None
        ra, ta = r_apns.get(cid, set()), t_apns.get(pid, set())
        cnames = _str_list(rinfo.cnames) if rinfo is not None else []
        pa, pb = phase_marker(a["name"]), phase_marker(b["name"])
        cu = rinfo.cu if rinfo is not None else None
        cy = rinfo.cy if rinfo is not None else None
        rows.append({
            "market": market, "community_id": cid, "property_id": int(pid),
            "block": "+".join(sorted(why)),
            "r_name": a["name"], "t_name": b["name"], "r_address": a.address,
            "t_address": ", ".join(v for v in [b.address, b.city, b.zip] if isinstance(v, str) and v),
            "r_url": a.url, "r_product": a["product"], "t_product": b["product"],
            "t_geo_precision": b.geo_precision,
            "distance_m": round(d, 1) if d is not None else None,
            "name_sim": round(name_sim(a["name"], b["name"]), 3),
            "name_match": name_match(a["name"], b["name"]),
            "county_name_sim": round(max([name_sim(c, b["name"]) for c in cnames] or [0.0]), 3),
            "county_names": "; ".join(cnames[:3]) or None,
            "t_county_names": "; ".join(t_cnames.get(pid, [])[:3]) or None,
            "t_county_name_sim": round(max([name_sim(a["name"], c) for c in t_cnames.get(pid, [])] or [0.0]), 3),
            "addr_rel": address_relation(a.addr, b.addr),
            "zip_same": bool(a.addr.zip5 and b.addr.zip5 and a.addr.zip5 == b.addr.zip5),
            "r_has_apn": bool(ra), "t_has_apn": bool(ta),
            "apn_shared": len(ra & ta),
            "apn_conflict": bool(ra and ta and not (ra & ta)),
            "r_apn_conf": rinfo.conf if rinfo is not None and isinstance(rinfo.conf, str) else None,
            "r_county_units": None if cu is None or pd.isna(cu) else int(cu),
            "t_units": None if pd.isna(b.units) else int(b.units),
            "r_county_year": None if cy is None or pd.isna(cy) else int(cy),
            "t_year": None if pd.isna(b.year_built) else int(b.year_built),
            "phase_r": pa, "phase_t": pb,
            "phase_conflict": bool((pa or pb) and pa != pb),
        })
    df = pd.DataFrame(rows)
    df["units_ratio"] = [
        (min(x, y) / max(x, y)) if x and y else None for x, y in zip(df.r_county_units, df.t_units)]
    df["year_gap"] = [abs(x - y) if x and y else None for x, y in zip(df.r_county_year, df.t_year)]
    df["pair_id"] = [f"{m}:{c}:{p}" for m, c, p in zip(df.market, df.community_id, df.property_id)]
    return df.sort_values(["community_id", "property_id"]).reset_index(drop=True)


def blocking_recall(df: pd.DataFrame) -> dict:
    apn = df[df.apn_shared > 0]
    n = len(apn)
    return {"apn_pairs": n,
            "within_150m": int((apn.distance_m <= 150).sum()),
            "within_500m": int((apn.distance_m <= 500).sum()),
            "addr_key": int(apn.block.str.contains("addr").sum()),
            "near_or_addr": int(((apn.distance_m <= 500) | apn.block.str.contains("addr")).sum()),
            "no_distance": int(apn.distance_m.isna().sum())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    a = ap.parse_args()
    con = duckdb.connect(str(DB))
    df = build_pairs(con, a.market)
    con.execute("create table if not exists pairs as select * from df limit 0")
    con.execute("delete from pairs where market = ?", [a.market])
    con.execute("insert into pairs by name select * from df")
    log.info("pairs %s: %d  blocks: %s", a.market, len(df), df.block.value_counts().to_dict())
    log.info("blocking recall on APN-shared pairs: %s", blocking_recall(df))


if __name__ == "__main__":
    main()

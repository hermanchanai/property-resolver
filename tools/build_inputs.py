"""Build data/inputs.duckdb: the public source records the resolver links.

Sources (all public; read-only):
  * frozen snapshot (default ../text-to-sql-eval/snapshot/cre_snapshot.duckdb, opened read_only):
      rents.communities  -> rent_community   (scraped leasing sites; name, address, lat/lng only)
      trades.property    -> sold_property    (public-filing sold comps: deed_az / ga_owner_diff / news)
      trades.property_parcel -> sold_property_apn
      county_permits.mf_permits -> permit
  * source Postgres (read-only transaction, asserted):
      county_parcels.community_matches (latest run, matched) -> community_parcel
        vendor_units / vendor_year_built are licensed and are NOT copied.
      capital.parcels (Maricopa Assessor apartment master) -> county_parcel
      county_parcels.fl_parcels (Tampa-area FL DOR, MF + referenced parcels) -> county_parcel

Usage: python -m tools.build_inputs [--snapshot PATH] [--out data/inputs.duckdb]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from tools.apn import norm_apn
from tools.fl_altkey import altkey_map
from tools.db import query

log = logging.getLogger("build_inputs")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = REPO.parent / "text-to-sql-eval/snapshot/cre_snapshot.duckdb"

MARKETS = {"PHX": {"msa": "Phoenix", "permit_msa": "phoenix", "fips": ["04013"]},
           "TPA": {"msa": "Tampa", "permit_msa": "tampa", "fips": ["12057", "12103", "12101", "12053"]}}
FIPS_TO_COUNTY = {"04013": "maricopa_az", "12057": "fl_hillsborough", "12103": "fl_pinellas",
                  "12101": "fl_pasco", "12053": "fl_hernando"}
MATCH_RUN = "2026-10-05-q2"

DDL = {
    "rent_community": """create table rent_community (market varchar, community_id varchar, name varchar,
        address varchar, url varchar, lat double, lng double, geocode_precision varchar, product varchar)""",
    "sold_property": """create table sold_property (market varchar, property_id bigint, name varchar,
        name_norm varchar, address varchar, city varchar, zip varchar, lat double, lng double,
        geo_precision varchar, product varchar, units integer, year_built integer)""",
    "sold_property_apn": """create table sold_property_apn (market varchar, property_id bigint,
        county varchar, apn varchar, apn_norm varchar)""",
    "community_parcel": """create table community_parcel (market varchar, community_id varchar,
        county varchar, apn varchar, apn_norm varchar, match_confidence varchar, match_rule varchar,
        distance_m double, owner_norm varchar, county_units integer, year_built_min integer,
        year_built_max integer, shared_parcel boolean)""",
    "county_parcel": """create table county_parcel (market varchar, county varchar, apn varchar,
        apn_norm varchar, econ_unit varchar, complex_name varchar, situs varchar, owner varchar,
        units integer, year_built integer, use_code varchar, lat double, lng double)""",
    "permit": """create table permit (market varchar, permit_id varchar, jurisdiction varchar,
        project_name varchar, address varchar, zip varchar, parcel_id varchar, apn_norm varchar,
        lat double, lng double, units integer, owner_name varchar, developer_name varchar,
        date_issued date, dedup_address_key varchar)""",
}


def _insert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    con.register("_df", df)
    cols = [r[0] for r in con.execute(f"select column_name from duckdb_columns() "
                                      f"where table_name='{table}' order by column_index").fetchall()]
    con.execute(f"insert into {table} select {', '.join(cols)} from _df")
    con.unregister("_df")
    return len(df)


def build(snapshot: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    for ddl in DDL.values():
        con.execute(ddl)
    con.execute(f"attach '{snapshot}' as snap (read_only)")
    counts: dict[str, int] = {}

    for mk, cfg in MARKETS.items():
        con.execute("""insert into rent_community select market, id::varchar, name, address, url,
            latitude, longitude, geocode_precision, product from snap.rents.communities
            where market = ? and superseded_by is null and latitude is not null""", [mk])
        con.execute("""insert into sold_property select ?, property_id, current_name, name_norm,
            situs_address, city, zip, lat, lng, geo_precision, product, units, year_built
            from snap.trades.property where msa = ?""", [mk, cfg["msa"]])
        apn = con.execute("""select ?, pp.property_id, pp.county, pp.apn from snap.trades.property_parcel pp
            join snap.trades.property p using (property_id) where p.msa = ?""", [mk, cfg["msa"]]).df()
        apn.columns = ["market", "property_id", "county", "apn"]
        # FL: sold comps may carry the appraiser's own key (Hillsborough folio = DOR ALT_KEY) -> map to the PIN
        xw = {c: altkey_map(c) for c in set(apn.county) if c == "fl_hillsborough"}
        for c, m in xw.items():
            counts[f"altkey_map_{c}"] = len(m)
        apn["apn_norm"] = [norm_apn(c, a, xw.get(c)) for c, a in zip(apn.county, apn.apn)]
        _insert(con, "sold_property_apn", apn)
        pm = con.execute("""select ?, permit_id::varchar, jurisdiction, project_name, address, zip, parcel_id,
            latitude, longitude, units, owner_name, developer_name, date_issued::date, dedup_address_key
            from snap.county_permits.mf_permits where msa = ? and is_mf""", [mk, cfg["permit_msa"]]).df()
        pm.columns = ["market", "permit_id", "jurisdiction", "project_name", "address", "zip", "parcel_id",
                      "lat", "lng", "units", "owner_name", "developer_name", "date_issued", "dedup_address_key"]
        county = FIPS_TO_COUNTY[cfg["fips"][0]] if mk == "PHX" else None
        pm["apn_norm"] = [norm_apn(county, a) if county else None for a in pm.parcel_id]
        _insert(con, "permit", pm)

        cols, rows = query("""select community_id::text, county_fips, unnest(parcel_ids), match_confidence,
            match_rule, distance_m::float, owner_norm, county_units, county_year_built_min,
            county_year_built_max, coalesce(shared_parcel, false)
            from county_parcels.community_matches
            where run_id = %s and market = %s and status = 'matched'""", (MATCH_RUN, mk))
        cp = pd.DataFrame(rows, columns=["community_id", "fips", "apn", "match_confidence", "match_rule",
                                         "distance_m", "owner_norm", "county_units", "year_built_min",
                                         "year_built_max", "shared_parcel"], dtype=object)
        cp["market"] = mk
        cp["county"] = cp.fips.map(FIPS_TO_COUNTY)
        cp["apn_norm"] = [norm_apn(c, a) for c, a in zip(cp.county, cp.apn)]
        _insert(con, "community_parcel", cp)

    # county parcels: Maricopa apartment master
    cols, rows = query("""select apn, econ_unit, complex_name,
        concat_ws(' ', situs_address, situs_city, situs_zip), owner_name, num_units, construction_yr, puc,
        latitude, longitude from capital.parcels where county = 'maricopa_az'""")
    mc = pd.DataFrame(rows, columns=["apn", "econ_unit", "complex_name", "situs", "owner", "units",
                                     "year_built", "use_code", "lat", "lng"], dtype=object)
    mc["market"], mc["county"] = "PHX", "maricopa_az"
    mc["apn_norm"] = [norm_apn("maricopa_az", a) for a in mc.apn]
    counts["county_parcel_PHX"] = _insert(con, "county_parcel", mc)

    # county parcels: FL DOR, MF-10+ parcels plus any parcel a community match or sold comp references
    ref = con.execute("""select distinct county, apn from community_parcel where market='TPA'
                         union select distinct county, apn from sold_property_apn where market='TPA'
                         union select distinct county, apn_norm from sold_property_apn where market='TPA'""").fetchall()
    ref_by_fips: dict[str, list[str]] = {}
    inv = {v: k for k, v in FIPS_TO_COUNTY.items()}
    for county, apn in ref:
        if county in inv:
            ref_by_fips.setdefault(inv[county], []).append(apn)
    for fips in MARKETS["TPA"]["fips"]:
        cols, rows = query("""select parcel_id, null, null, concat_ws(' ', phy_addr1, phy_city, phy_zipcd), own_name,
            no_res_unt, act_yr_blt, dor_uc, st_y(centroid::geometry), st_x(centroid::geometry)
            from county_parcels.fl_parcels where county_fips = %s
              and (dor_uc = '003' or parcel_id = any(%s))""", (fips, ref_by_fips.get(fips, [])))
        fl = pd.DataFrame(rows, columns=["apn", "econ_unit", "complex_name", "situs", "owner", "units",
                                         "year_built", "use_code", "lat", "lng"], dtype=object)
        fl["market"], fl["county"] = "TPA", FIPS_TO_COUNTY[fips]
        fl["apn_norm"] = [norm_apn(FIPS_TO_COUNTY[fips], a) for a in fl.apn]
        counts[f"county_parcel_{fips}"] = _insert(con, "county_parcel", fl)

    for t in DDL:
        for mk in MARKETS:
            counts[f"{t}_{mk}"] = con.execute(f"select count(*) from {t} where market=?", [mk]).fetchone()[0]
    con.execute("detach snap")
    con.close()
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "snapshot": str(snapshot), "snapshot_md5_of_path_size": hashlib.md5(
                    f"{snapshot}:{snapshot.stat().st_size}".encode()).hexdigest(),
                "community_match_run": MATCH_RUN, "counts": counts}
    (REPO / "data").mkdir(exist_ok=True)
    (REPO / "data/inputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", type=Path, default=REPO / "data/inputs.duckdb")
    a = ap.parse_args()
    m = build(a.snapshot, a.out)
    for k, v in sorted(m["counts"].items()):
        log.info("%-32s %7d", k, v)


if __name__ == "__main__":
    main()

"""Layer 2: Fellegi-Sunter match probabilities with Splink on DuckDB (rent community <-> sold property).

Each record carries its own name plus the county assessor's complex name reached through its parcels
(rent side: county_parcels community match; sold side: its own APNs). Comparisons:
  name      best Jaro-Winkler over {own, county} x {own, county} name keys
  address   exact number+street / same street, same side (parity) within 150 / same street other side / else
  distance  great-circle km thresholds 0.05 / 0.15 / 0.3 / 0.5  (NULL for city-centroid geocodes)
  apn       shares a normalized APN / both have APNs, none shared
  units     county units (rent side) vs sold-comp units: ratio >= 0.95 / >= 0.75 / else
Parameters: u by random sampling; m from the TUNE split's labeled matches (default) or unsupervised EM
(--m-source em; collapses onto 'same name + shared APN' and scores renames ~0.01, kept for comparison);
lambda from deterministic rules. Predictions use the same three blocking keys as tools/candidates.py.

Usage: python -m tools.layer2_splink --market PHX   -> table l2_scores in data/inputs.duckdb
"""
from __future__ import annotations

import argparse
import re
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from tools.normalize import name_key, parse_address

log = logging.getLogger("layer2")
REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data/inputs.duckdb"


def _rec(uid, src, name, county_names, address, lat, lng, apns, units, year):
    a = parse_address(address)
    cn = [name_key(c) for c in county_names if isinstance(c, str)]
    cn = [c for c in cn if c]
    own = name_key(name) or None
    return {"unique_id": uid, "source_dataset": src, "name1": own,
            "name2": next((c for c in cn if c != own), cn[0] if cn else None),
            "addr_num": int(a.number) if a.number else None, "addr_street": a.street,
            "addr_parity": (int(a.number) % 2) if a.number else None,
            "lat": lat, "lng": lng, "apns": sorted(apns) or None,
            "units": units, "year": year}


def build_records(con, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = con.execute("""
        select rc.community_id, rc.name, rc.address, rc.lat, rc.lng,
               list(distinct cp.apn_norm) filter (where cp.apn_norm is not null
                    and cp.match_confidence in ('high','medium')) apns,
               list(distinct pc.complex_name) filter (where pc.complex_name is not null) cnames,
               any_value(cp.county_units) units, any_value(cp.year_built_min) yr
        from rent_community rc
        left join community_parcel cp on cp.market = rc.market and cp.community_id = rc.community_id
        left join county_parcel pc on pc.market = cp.market and pc.apn_norm = cp.apn_norm
        where rc.market = ? group by all""", [market]).df()
    t = con.execute("""
        select sp.property_id, sp.name, concat_ws(', ', sp.address, sp.city, sp.zip) address,
               case when sp.geo_precision in ('city','zip_approx') or sp.geo_precision is null then null else sp.lat end lat,
               case when sp.geo_precision in ('city','zip_approx') or sp.geo_precision is null then null else sp.lng end lng,
               list(distinct sa.apn_norm) filter (where sa.apn_norm is not null) apns,
               list(distinct pc.complex_name) filter (where pc.complex_name is not null) cnames,
               any_value(sp.units) units, any_value(sp.year_built) yr
        from sold_property sp
        left join sold_property_apn sa on sa.market = sp.market and sa.property_id = sp.property_id
        left join county_parcel pc on pc.market = sa.market and pc.apn_norm = sa.apn_norm
        where sp.market = ? group by all""", [market]).df()

    def lst(v):
        try:
            return [x for x in v if x is not None]
        except TypeError:
            return []

    def num(v):
        return None if v is None or pd.isna(v) else int(v)

    R = pd.DataFrame([_rec(f"r:{x.community_id}", "rent", x["name"], lst(x.cnames), x.address, x.lat, x.lng,
                           lst(x.apns), num(x.units), num(x.yr)) for _, x in r.iterrows()])
    T = pd.DataFrame([_rec(f"t:{x.property_id}", "sold", x["name"], lst(x.cnames), x.address,
                           None if pd.isna(x.lat) else x.lat, None if pd.isna(x.lng) else x.lng,
                           lst(x.apns), num(x.units), num(x.yr)) for _, x in t.iterrows()])
    return R, T


def _lr(sql: str) -> str:
    """Comparison SQL uses col_l / col_r (blocking rules keep l.col / r.col)."""
    return re.sub(r"\b([lr])\.(\w+)", r"\2_\1", sql)


def _jw(a: str, b: str) -> str:
    return f"coalesce(jaro_winkler_similarity({a}, {b}), 0)"


NAME_BEST = ("greatest(" + ", ".join(_jw(f"l.{x}", f"r.{y}") for x in ("name1", "name2")
                                     for y in ("name1", "name2")) + ")")


def CL(sql: str, label: str):
    return cll.CustomLevel(_lr(sql), label)


def settings(lam: float = 1e-4) -> SettingsCreator:
    name = cl.CustomComparison(output_column_name="name", comparison_levels=[
        CL("(l.name1 is null and l.name2 is null) or (r.name1 is null and r.name2 is null)",
                        "name null").configure(is_null_level=True),
        CL(f"{NAME_BEST} >= 0.97", "name jw >= 0.97"),
        CL(f"{NAME_BEST} >= 0.90", "name jw >= 0.90"),
        CL(f"{NAME_BEST} >= 0.80", "name jw >= 0.80"),
        cll.ElseLevel()])
    # address and distance are strongly correlated (a neighbour across the street is both "same street" and
    # "< 150 m"), so they form ONE comparison; Fellegi-Sunter multiplies independent comparisons.
    D = ("(2 * 6371 * asin(sqrt(power(sin(radians(r.lat - l.lat) / 2), 2) + cos(radians(l.lat)) "
         "* cos(radians(r.lat)) * power(sin(radians(r.lng - l.lng) / 2), 2))))")
    same_st = "l.addr_street = r.addr_street and abs(l.addr_num - r.addr_num) <= 150"
    loc = cl.CustomComparison(output_column_name="location", comparison_levels=[
        CL("(l.addr_street is null or r.addr_street is null or l.addr_num is null or r.addr_num is null) "
           "and (l.lat is null or r.lat is null)", "location null").configure(is_null_level=True),
        CL("l.addr_num = r.addr_num and l.addr_street = r.addr_street", "exact number + street"),
        CL(f"{same_st} and l.addr_parity = r.addr_parity and coalesce({D}, 0) <= 0.15",
           "same street, same side, <= 150 m"),
        CL(f"{same_st} and l.addr_parity <> r.addr_parity", "same street, opposite side"),
        CL(f"{D} <= 0.05", "<= 50 m, address differs"),
        CL(f"{D} <= 0.15", "<= 150 m, address differs"),
        CL(f"{D} <= 0.5 or ({same_st})", "<= 500 m or same street"),
        cll.ElseLevel()])
    apn = cl.CustomComparison(output_column_name="apn", comparison_levels=[
        CL("l.apns is null or r.apns is null", "apn null").configure(is_null_level=True),
        CL("list_has_any(l.apns, r.apns)", "shares an APN"),
        cll.ElseLevel()])
    units = cl.CustomComparison(output_column_name="units", comparison_levels=[
        CL("l.units is null or r.units is null or l.units = 0 or r.units = 0",
                        "units null").configure(is_null_level=True),
        CL("least(l.units, r.units) / greatest(l.units, r.units) >= 0.95", "ratio >= 0.95"),
        CL("least(l.units, r.units) / greatest(l.units, r.units) >= 0.75", "ratio >= 0.75"),
        cll.ElseLevel()])
    return SettingsCreator(
        link_type="link_only",
        comparisons=[name, loc, apn, units],
        blocking_rules_to_generate_predictions=[
            "abs(l.lat - r.lat) < 0.0055 and abs(l.lng - r.lng) < 0.0065",
            block_on("addr_num", "addr_street"),
            "list_has_any(l.apns, r.apns)",
        ],
        probability_two_random_records_match=lam,
        retain_intermediate_calculation_columns=True,
        additional_columns_to_retain=[],
    )


def run(market: str, m_source: str = "labels", model_from: Path | None = None) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect(str(DB), read_only=True)
    R, T = build_records(con, market)
    con.close()
    db = DuckDBAPI()
    sr, st = db.register(R, dataset_display_name="rent"), db.register(T, dataset_display_name="sold")
    if model_from:   # generalization run: the other market's trained model, no re-estimation at all
        linker = Linker([sr, st], str(model_from))
        log.info("applying saved model %s unchanged (no training on %s)", model_from.name, market)
        return _predict(linker, market)
    linker = Linker([sr, st], settings())
    linker.training.estimate_probability_two_random_records_match(
        ["list_has_any(l.apns, r.apns)", "l.addr_num = r.addr_num and l.addr_street = r.addr_street"],
        recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=2e6, seed=1)
    if m_source == "labels":
        # supervised m: labeled MATCHES from the tune split only (holdout never touches training)
        from tools.evaluate import load_labels
        lab = load_labels(market)
        pos = lab[(lab.split == "tune") & (lab.label == "match")]
        lt = pd.DataFrame({"source_dataset_l": "rent", "unique_id_l": "r:" + pos.community_id.astype(str),
                           "source_dataset_r": "sold", "unique_id_r": "t:" + pos.property_id.astype(str)})
        linker.table_management.register_table(lt, "m_labels", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("m_labels")
        log.info("m estimated from %d labeled tune matches (sources: %s)", len(pos),
                 pos.label_source.value_counts().to_dict())
    else:
        # EM pass 1 blocks on proximity (learns name/apn/units); pass 2 blocks on exact name key (learns location)
        linker.training.estimate_parameters_using_expectation_maximisation(
            "abs(l.lat - r.lat) < 0.0055 and abs(l.lng - r.lng) < 0.0065", fix_u_probabilities=False)
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("name1"))
    return _predict(linker, market)


def _predict(linker: Linker, market: str) -> tuple[pd.DataFrame, dict]:
    pred = linker.inference.predict().as_pandas_dataframe()
    pred["community_id"] = [x[2:] if x.startswith("r:") else y[2:] for x, y in zip(pred.unique_id_l, pred.unique_id_r)]
    pred["property_id"] = [int(y[2:]) if y.startswith("t:") else int(x[2:])
                           for x, y in zip(pred.unique_id_l, pred.unique_id_r)]
    pred["market"] = market
    model = json.loads(json.dumps(linker.misc.save_model_to_json(), default=str))
    keep = ["market", "community_id", "property_id", "match_weight", "match_probability"] + \
        [c for c in pred.columns if c.startswith("gamma_")]
    return pred[keep], model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--m-source", choices=["labels", "em"], default="labels")
    ap.add_argument("--model-from", type=Path, help="apply this saved model unchanged (no training)")
    a = ap.parse_args()
    scores, model = run(a.market, a.m_source, a.model_from)
    if not a.model_from:
        (REPO / "models").mkdir(exist_ok=True)
        (REPO / f"models/splink_{a.market.lower()}.json").write_text(json.dumps(model, indent=2) + "\n")
    con = duckdb.connect(str(DB))
    have = con.execute("select count(*) from duckdb_tables() where table_name='l2_scores'").fetchone()[0]
    if have:   # keep other markets' rows; columns may differ between model versions
        con.execute("""create or replace table l2_scores as
            select * from (select * from l2_scores where market <> ?) union all by name select * from scores""",
                    [a.market])
    else:
        con.execute("create table l2_scores as select * from scores")
    log.info("scored %d pairs; p>=0.5: %d", len(scores), int((scores.match_probability >= 0.5).sum()))


if __name__ == "__main__":
    main()

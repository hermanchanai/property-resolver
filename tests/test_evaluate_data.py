"""Data test: the evaluator must score layer 1 from the CURRENT pairs table, not from features frozen into the
sample CSV when it was drawn (skips when data/inputs.duckdb is absent)."""
from pathlib import Path

import duckdb
import pytest

DB = Path(__file__).resolve().parents[1] / "data/inputs.duckdb"


@pytest.mark.skipif(not DB.exists(), reason="inputs not built")
@pytest.mark.parametrize("market", ["PHX", "TPA"])
def test_l1_flag_comes_from_current_pairs(market):
    from tools.evaluate import frame
    s, _ = frame(market)
    con = duckdb.connect(str(DB), read_only=True)
    cur = con.execute("""select pair_id, apn_shared > 0 and r_apn_conf in ('high','medium') l1
                         from pairs where market = ?""", [market]).df().set_index("pair_id").l1
    con.close()
    got = s.set_index("pair_id").l1
    common = got.index.intersection(cur.index)
    diff = (got[common] != cur[common].fillna(False)).sum()
    assert diff == 0, f"{diff} labeled pairs carry a stale layer-1 flag"

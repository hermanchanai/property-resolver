"""Data test on the built inputs (skips when data/inputs.duckdb is absent)."""
from pathlib import Path

import duckdb
import pytest

DB = Path(__file__).resolve().parents[1] / "data/inputs.duckdb"


@pytest.mark.skipif(not DB.exists(), reason="inputs not built")
def test_hillsborough_sold_comp_apns_reach_dor_parcels():
    """Spec: Hillsborough sold comps carry folios; after the ALT_KEY crosswalk, most of their APNs that point at
    an MF (DOR use 003) parcel must join county_parcel. Measured before the fix: 741 / 7,592 joined in total."""
    con = duckdb.connect(str(DB), read_only=True)
    joined, total = con.execute("""
        select count(pc.apn_norm), count(*) from sold_property_apn sa
        left join county_parcel pc on pc.market = sa.market and pc.apn_norm = sa.apn_norm
        where sa.market = 'TPA' and sa.county = 'fl_hillsborough'""").fetchone()
    assert joined / total >= 0.5, f"only {joined}/{total} Hillsborough sold-comp APNs reach a DOR parcel"

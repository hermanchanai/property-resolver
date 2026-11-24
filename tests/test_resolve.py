import pandas as pd

from tools.resolve import UF, merge_log


def test_chaining_guard_refuses_two_rent_communities():
    uf = UF()
    assert uf.union("rent_community|PHX|a", "county_parcel|PHX|p1")
    assert uf.union("rent_community|PHX|b", "county_parcel|PHX|p2")
    # p1 and p2 are one economic unit, but that would merge two tracked communities -> refused
    assert not uf.union("county_parcel|PHX|p1", "county_parcel|PHX|p2")
    assert uf.find("rent_community|PHX|a") != uf.find("rent_community|PHX|b")


def test_sold_property_can_join_a_rent_community():
    uf = UF()
    assert uf.union("rent_community|PHX|a", "sold_property|PHX|1")
    assert uf.union("sold_property|PHX|1", "county_parcel|PHX|p1")
    assert uf.find("county_parcel|PHX|p1") == uf.find("rent_community|PHX|a")


def _x(rows):
    return pd.DataFrame(rows, columns=["canonical_id", "source", "source_id", "market"])


def test_merge_log_detects_merge_and_split():
    prev = _x([("old1", "rent_community", "a", "PHX"), ("old2", "sold_property", "1", "PHX"),
               ("old3", "rent_community", "b", "PHX"), ("old3", "sold_property", "2", "PHX")])
    new = _x([("n1", "rent_community", "a", "PHX"), ("n1", "sold_property", "1", "PHX"),
              ("n2", "rent_community", "b", "PHX"), ("n3", "sold_property", "2", "PHX")])
    ml = merge_log(new, prev)
    assert set(zip(ml.event, ml.canonical_id)) == {("merge", "n1"), ("split", "old3")}

import pytest

from tools.apn import norm_apn


@pytest.mark.parametrize("county,raw,want", [
    ("maricopa_az", "306-04-784", "30604784"),
    ("maricopa_az", "51008553A", "51008553A"),
    ("maricopa_az", "510-08-553a", "51008553A"),
    ("fl_pinellas", "14 30 01 42030 002 0100", "143001420300020100"),
    ("fl_pasco", "36-26-19-0230-0B600-0000", "36261902300B6000000"),
    ("maricopa_az", "", None),
    ("maricopa_az", "PENDING", None),
    ("maricopa_az", "99999999", None),
    ("fl_pasco", "-- ---", None),
    ("maricopa_az", None, None),
])
def test_norm_apn(county, raw, want):
    assert norm_apn(county, raw) == want


def test_maricopa_permit_dash_form_equals_assessor_form():
    assert norm_apn("maricopa_az", "306-04-784") == norm_apn("maricopa_az", "30604784")


def test_hillsborough_folio_maps_to_dor_pin():
    """Spec: a sold comp's Hillsborough FOLIO (10 digits) is the DOR NAL ALT_KEY of a PIN; both forms of one
    parcel must produce the same layer-1 key."""
    xw = {"1528860000": "192833ZZZ000005706900A"}
    assert norm_apn("fl_hillsborough", "1528860000", xwalk=xw) == norm_apn("fl_hillsborough", "192833ZZZ000005706900A")


def test_xwalk_never_rewrites_an_apn_that_is_not_a_key():
    xw = {"1528860000": "192833ZZZ000005706900A"}
    assert norm_apn("fl_hillsborough", "172912D22000000000020A", xwalk=xw) == "172912D22000000000020A"

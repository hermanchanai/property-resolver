"""Per-county APN normalization (layer 1 key).

Rule: upper-case, drop every non-alphanumeric character (Maricopa '306-04-784' == '30604784',
Pinellas '14 30 01 42030 002 0100' == '143001420300020100'). Placeholders become None
(blank, no digit, one repeated digit like '99999999', 'PENDING').
Hillsborough sold comps carry the 10-digit FOLIO while FL DOR carries the PIN ('172713001000000086900U').
The DOR NAL file's ALT_KEY column IS the folio, so `xwalk` (ALT_KEY -> PARCEL_ID, tools/fl_altkey.py) maps a
folio onto its PIN before normalizing.
"""
from __future__ import annotations

import re

_PLACEHOLDER = {"PENDING", "NONE", "NA", "TBD", "UNKNOWN"}


def norm_apn(county: str | None, apn: str | None, xwalk: dict[str, str] | None = None) -> str | None:
    if apn is None:
        return None
    t = re.sub(r"[^A-Z0-9]", "", str(apn).upper())
    if xwalk and t in xwalk:
        t = re.sub(r"[^A-Z0-9]", "", xwalk[t].upper())
    if not t or t in _PLACEHOLDER or not re.search(r"\d", t):
        return None
    digits = re.sub(r"\D", "", t)
    if len(set(digits)) == 1 and len(digits) >= 4:
        return None
    return t

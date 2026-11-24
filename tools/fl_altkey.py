"""ALT_KEY -> PARCEL_ID crosswalk from the Florida DOR NAL tax-roll file (public record).

Hillsborough's property appraiser publishes the FOLIO; DOR publishes the PIN and carries the folio as
ALT_KEY. The NAL zip is read from the local cache the county-parcel loader already downloaded
(~/.cache/sales-comps/fl_dor/<year><F|P>/<County> <co_no> <Final|Preliminary> NAL <year>.zip); nothing is
fetched here. Final roll preferred over Preliminary.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

CACHE = Path.home() / ".cache/sales-comps/fl_dor"
COUNTY_FILE = {"fl_hillsborough": "Hillsborough", "fl_pinellas": "Pinellas", "fl_pasco": "Pasco",
               "fl_hernando": "Hernando"}


def nal_zip(county: str, year: int = 2026) -> Path | None:
    name = COUNTY_FILE[county]
    for stage in ("F", "P"):
        hits = sorted((CACHE / f"{year}{stage}").glob(f"{name} * NAL {year}.zip"))
        if hits:
            return hits[0]
    return None


def altkey_map(county: str, year: int = 2026) -> dict[str, str]:
    """{normalized ALT_KEY: PARCEL_ID}; empty when the NAL file is not cached."""
    z = nal_zip(county, year)
    if z is None:
        return {}
    out: dict[str, str] = {}
    with zipfile.ZipFile(z) as zf:
        member = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        with zf.open(member) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1")):
                k = re.sub(r"[^A-Z0-9]", "", (row.get("ALT_KEY") or "").upper())
                if k and row.get("PARCEL_ID"):
                    out[k] = row["PARCEL_ID"]
    return out

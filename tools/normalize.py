"""Name and address normalization shared by every layer.

Name rules generalize the county-parcel matcher's stop-word list (the source
pipeline's county-parcel name match): marketing words ('apartments', 'luxury', 'homes') carry no identity.
Addresses are parsed with usaddress; the comparison key is (house number, street name, ZIP5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import usaddress
from rapidfuzz import fuzz

NAME_STOP = {
    "the", "at", "on", "of", "and", "a", "an", "in", "by", "apartments", "apartment", "apts", "apt", "homes",
    "townhomes", "townhome", "residences", "residence", "villas", "luxury", "community", "communities",
    "living", "rentals", "lofts", "flats", "llc", "lp", "inc", "llc.", "condominiums", "condos", "condo",
    "rental", "rent", "home", "phase", "ph", "az", "fl",
}
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
PHASE_RE = re.compile(r"\b(phase|ph)\s*([ivx]{1,4}|\d{1,2})\b|\b([ivx]{2,4}|[2-6])$", re.I)
DIRS = {"north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
        "southeast": "se", "southwest": "sw"}
SUFFIX = {"street": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr", "boulevard": "blvd",
          "lane": "ln", "court": "ct", "circle": "cir", "parkway": "pkwy", "place": "pl", "trail": "trl",
          "highway": "hwy", "terrace": "ter", "way": "way", "loop": "loop", "run": "run"}
ORD = re.compile(r"^(\d+)(st|nd|rd|th)$")


def _str(s) -> str:
    """None / NaN / non-strings read as empty text."""
    return s if isinstance(s, str) else ""


def tokens(s: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", _str(s).lower())


def name_key(s: str | None) -> str:
    """Identity tokens only, in order: 'The Palms Apartments II' -> 'palms 2'."""
    out = []
    for t in tokens(s):
        if t in NAME_STOP:
            continue
        out.append(ROMAN.get(t, t))
    return " ".join(out)


def phase_marker(s: str | None) -> str | None:
    """'Encantada Phase II' -> '2'; 'Liv Crossroads' -> None. Used by the phase-chaining guard."""
    m = PHASE_RE.search(_str(s).strip())
    if not m:
        return None
    v = (m.group(2) or m.group(3) or "").lower()
    return ROMAN.get(v, v) or None


def name_sim(a: str | None, b: str | None) -> float:
    """0-1 token-set similarity on identity tokens (0 when either side has none)."""
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return 0.0
    return fuzz.token_set_ratio(ka, kb) / 100.0


def name_match(a: str | None, b: str | None) -> bool:
    """County-parcel matcher rule (S-rules): >=2 shared identity tokens covering 60% of the shorter
    name, or the shorter name IS one distinctive (>=5 char) shared token."""
    ta, tb = set(name_key(a).split()), set(name_key(b).split())
    ta.discard("")
    tb.discard("")
    if not ta or not tb:
        return False
    inter = ta & tb
    if len(inter) >= 2:
        return len(inter) / min(len(ta), len(tb)) >= 0.6
    return len(inter) == 1 and min(len(ta), len(tb)) == 1 and len(next(iter(inter))) >= 5


@dataclass(frozen=True)
class Addr:
    number: str | None
    street: str | None      # identity words of the street name, 'campbell', '52' (ordinals stripped)
    zip5: str | None

    @property
    def key(self) -> str | None:
        return f"{self.number}|{self.street}" if self.number and self.street else None


def _street_words(raw: str) -> str | None:
    words = []
    for t in tokens(raw):
        t = DIRS.get(t, SUFFIX.get(t, t))
        m = ORD.match(t)
        if m:
            t = m.group(1)
        if t in DIRS.values() or t in SUFFIX.values():
            continue
        words.append(t)
    return " ".join(words) or None


def parse_address(s: str | None) -> Addr:
    if not isinstance(s, str) or not s.strip():
        return Addr(None, None, None)
    z = re.search(r"\b(\d{5})(?:-\d{4})?\s*$", s.strip())
    zip5 = z.group(1) if z else None
    try:
        tagged, kind = usaddress.tag(s)
    except usaddress.RepeatedLabelError:
        tagged, kind = {}, "ambiguous"
    num = tagged.get("AddressNumber")
    street = tagged.get("StreetName")
    if not num or not street:     # fallback: leading house number then words up to the first comma
        head = s.split(",")[0]
        m = re.match(r"\s*(\d+)[\-\d]*\s+(.*)", head)
        if not m:
            return Addr(None, None, zip5)
        num, street = m.group(1), m.group(2)
    num = re.match(r"\d+", num)
    num = num.group(0).lstrip("0") or "0" if num else None
    return Addr(num, _street_words(street), zip5 or tagged.get("ZipCode", "")[:5] or None)


def address_relation(a: Addr, b: Addr) -> str:
    """'exact' same number + street; 'street' same street, numbers within 200; 'none' otherwise; 'na' if
    either side cannot be parsed."""
    if not (a.street and b.street):
        return "na"
    same_street = a.street == b.street or (set(a.street.split()) & set(b.street.split()) - {"e", "w", "n", "s"})
    if not same_street:
        return "none"
    if a.number and b.number:
        if a.number == b.number:
            return "exact"
        if abs(int(a.number) - int(b.number)) <= 200:
            return "street"
    return "none" if a.number and b.number else "na"

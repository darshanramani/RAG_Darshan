"""PII scrubbing applied BEFORE anything is chunked, embedded or published.

Policy for the public bot:
  * remove: phone numbers, street addresses, postal/ZIP codes
  * keep:   email address and LinkedIn URL (professional contact details)
"""
from __future__ import annotations

import re

# Phone numbers in many shapes, including odd ones like "+1 519-546 (0671)".
_PHONE = re.compile(
    r"""
    (?<![\w/])                 # not glued to a word or URL path
    (?:\+?\d{1,3}[\s.-]*)?     # optional country code
    (?:\(?\d{3}\)?[\s.-]*)     # area code
    \(?\d{3}\)?[\s.-]*         # exchange
    \(?\d{4}\)?                # line number
    (?![\w])
    """,
    re.VERBOSE,
)

# Canadian postal codes (A1A 1A1) and US ZIP(+4) after a state/province word.
_CA_POSTAL = re.compile(r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b", re.I)

# "20 Example Ave", "127 Main St N" ...
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s){1,4}"
    r"(?:Ave(?:nue)?|St(?:reet)?|Rd|Road|Blvd|Boulevard|Dr(?:ive)?|Cres(?:cent)?|Ct|Court|"
    r"Way|Lane|Ln|Pl(?:ace)?|Terr(?:ace)?|Pkwy|Hwy)\.?(?:\s[NSEW]\b)?",
)


def scrub(text: str) -> str:
    """Return text with phone numbers, street addresses and postal codes removed."""
    text = _STREET.sub("[address removed]", text)
    text = _CA_POSTAL.sub("", text)
    text = _PHONE.sub("", text)
    # tidy separators left behind ("Ontario |  | email" -> "Ontario | email")
    text = re.sub(r"(\s*\|\s*)+(?=\s*\|)", "", text)
    text = re.sub(r"\|\s*\|", "|", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def find_pii(text: str) -> list[str]:
    """Used by tests/CI to assert that nothing sensitive survived."""
    hits: list[str] = []
    for rx in (_PHONE, _CA_POSTAL, _STREET):
        hits += [m.group(0) for m in rx.finditer(text)]
    return hits

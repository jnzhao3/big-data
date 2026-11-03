"""Normalization of cell values, ground-truth values, predictions and target names before comparison.

The rules were derived from the quirks actually present in GDC_new, GDC-SM and GDC-VM
(see the E0 design doc, section 3.3). They are deliberately lexical: no float round-trips.
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation

import pandas as pd

# Structural "no match". Predictions use JSON null or an empty target list; this is the in-memory marker.
NONE = None

_FLOAT_INT = re.compile(r"[+-]?\d+\.0+")

# Spellings of "no match" accepted in schema-matching predictions. GDC column names never collide with these.
SM_NONE_TOKENS = frozenset({"", "none", "null", "nil", "n/a", "na", "nan", "no match", "nomatch", "no_match",
                            "unmatched", "-", "--"})


def num(s: str) -> str:
    """Canonical spelling of an integer written as a decimal; everything else is returned unchanged.

    '1031.0' -> '1031', '-1.0' -> '-1', '+5.00' -> '5', '-0.0' -> '0'.
    Unchanged: '1.50', '5.12e-08', '007', 'na', 'NaN', '>=90', '1_000', 'inf'.
    float() is not used: it accepts '1_000' and ' 12 ', rewrites '1e3', and loses precision above 2**53.
    """
    if _FLOAT_INT.fullmatch(s):
        s = s[: s.index(".")].lstrip("+")
        if s.lstrip("-").strip("0") == "":
            s = "0"
    return s


def canon_value(v) -> str:
    """Scoring key for a source cell, a ground-truth value or a predicted value. '' means a missing cell.

    None, NaN and pd.NA -> ''. Python and numpy floats go through repr(float(x)) first
    (numpy 2 reprs np.float64(1.5) as 'np.float64(1.5)'). Strings are NFC-normalized and stripped;
    inner whitespace is kept because GDC-VM keeps it (e.g. 'R0:  No residual tumor').
    """
    if v is None:
        return ""
    if isinstance(v, float):  # includes numpy.float64
        if math.isnan(v):
            return ""
        v = repr(float(v))
    elif not isinstance(v, str):
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        v = str(v)
    return num(unicodedata.normalize("NFC", v).strip())


def lenient_value(v: str) -> str:
    """Key for the lenient value comparison: inner whitespace collapsed, case folded."""
    return re.sub(r"\s+", " ", v).casefold()


def canon_target_col(v):
    """Predicted schema-matching target -> casefolded GDC column name, or NONE.

    The 736 GDC column names are all distinct after casefolding.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return NONE
    s = unicodedata.normalize("NFC", str(v)).strip().strip("`'\"").strip()
    return NONE if s.casefold() in SM_NONE_TOKENS else s.casefold()


def to_decimal(s):
    """Decimal for a finite number written as a string, else None."""
    try:
        d = Decimal(str(s))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return d if d.is_finite() else None

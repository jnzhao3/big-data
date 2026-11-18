"""The matcher tools as plain Python (the MCP server in server/matcher_server.py wraps these).

Every tool that returns rows pages them with offset/limit and never returns more than the row cap.
Nothing here reads the ground truth: see study.py.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .codebox import CodeError, CodeStore, execute
from .store import SubmissionStore, item_key, now_iso
from .study import StudyView
from hbench.normalize import SM_NONE_TOKENS, canon_value, to_decimal

DEFAULT_CAP = 50
DEFAULT_SEARCH_TOP_K = 5
METHODS = ("exact", "normalized", "ratio", "partial_ratio", "token_sort", "token_set", "jaro_winkler",
           "soundex", "metaphone")
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


class ToolError(ValueError):
    """An error the model sees. Messages never depend on the ground truth."""


# ---------------------------------------------------------------------- helpers

def normalize_text(s) -> str:
    """lowercase, punctuation to spaces, whitespace collapsed"""
    return " ".join(_PUNCT.sub(" ", str(s).lower()).split())


def similarity(a: str, b: str, method: str) -> float:
    """Similarity in [0, 1] between two strings."""
    if method not in METHODS:
        raise ToolError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    if method == "exact":
        return float(a == b)
    na, nb = normalize_text(a), normalize_text(b)
    if method == "normalized":
        return float(na == nb)
    if method == "ratio":
        return fuzz.ratio(na, nb) / 100
    if method == "partial_ratio":
        return fuzz.partial_ratio(na, nb) / 100
    if method == "token_sort":
        return fuzz.token_sort_ratio(na, nb) / 100
    if method == "token_set":
        return fuzz.token_set_ratio(na, nb) / 100
    if method == "jaro_winkler":
        return JaroWinkler.normalized_similarity(na, nb)
    # phonetic: 1.0 when the codes agree, else the ratio of the codes
    enc = jellyfish.soundex if method == "soundex" else jellyfish.metaphone
    ca, cb = (enc(x) if x else "" for x in (na, nb))
    if not ca or not cb:
        return 0.0
    return 1.0 if ca == cb else fuzz.ratio(ca, cb) / 100


def shape(v: str) -> str:
    """Value shape with runs collapsed: 'White' -> 'Aa', 'WHITE' -> 'A', 'pT3a' -> 'aA9a', 'C3L-00004' -> 'A9A-9'."""
    out = []
    for ch in v:
        c = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else " " if ch.isspace() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def casing(v: str) -> str:
    letters = [c for c in v if c.isalpha()]
    if not letters:
        return "no_letters"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    if v.istitle():
        return "title"
    return "mixed"


def _page(items: list, offset: int, limit: int | None, cap: int) -> tuple[list, dict]:
    offset = max(0, int(offset or 0))
    limit = cap if limit is None else max(1, min(int(limit), cap))
    sl = items[offset: offset + limit]
    end = offset + len(sl)
    return sl, {"total": len(items), "offset": offset, "returned": len(sl),
                "next_offset": end if end < len(items) else None, "row_cap": cap}


# ---------------------------------------------------------------------- the tools

class MatcherTools:
    def __init__(self, view: StudyView, store: SubmissionStore, cap: int = DEFAULT_CAP, sql=None,
                 search_only: bool = False, search_top_k: int = DEFAULT_SEARCH_TOP_K, code: CodeStore | None = None):
        self.view = view
        self.store = store
        self.cap = cap
        self.sql = sql  # SqlGuard or None
        self.code = code  # CodeStore or None (code tools disabled)
        # search-only mode: no whole vocabularies; permissible values are reachable only through
        # search_permissible, at most search_top_k per call, so every candidate considered is in the server log
        if search_only and sql is not None:
            raise ValueError("search-only mode cannot be combined with the SQL tool (it exposes permissible_values)")
        self.search_only = search_only
        self.search_top_k = max(1, min(int(search_top_k), cap))
        # (attribute, value) pairs returned by search_permissible; in search-only mode only these can be submitted
        self.searched: set[tuple[str, str]] = set()

    # -- helpers -----------------------------------------------------------

    def _col(self, column: str) -> str:
        if column not in self.view.columns:
            raise ToolError(f"unknown source column {column!r}; describe_study lists the columns")
        return column

    def _attr(self, attribute: str) -> str:
        if not self.view.has_attribute(attribute):
            near = self.search_attributes(attribute, top_k=5)["results"]
            hint = ", ".join(r["attribute"] for r in near)
            raise ToolError(f"unknown GDC attribute {attribute!r}; closest names: {hint}")
        return attribute

    def _pairs(self) -> list[tuple[str, str]]:
        return list(self.view.pairs) if self.view.mode == "oracle" else self.store.mapped_pairs()

    # -- exploration ---------------------------------------------------------

    def describe_study(self, offset: int = 0, limit: int | None = None, pairs_offset: int = 0) -> dict:
        v = self.view
        cols = [{"name": c, "inferred_type": v.inferred_type(c),
                 "n_distinct_values": len(v.value_counts(c)) - (1 if "" in v.value_counts(c) else 0),
                 "n_missing_cells": v.value_counts(c).get("", 0)} for c in v.columns]
        page, meta = _page(cols, offset, limit, self.cap)
        out = {"study": v.study, "mode": v.mode, "n_rows": len(v.table), "columns": page, "columns_page": meta}
        if v.mode == "oracle":
            pairs = [{"source_attribute": s, "target_attribute": t} for s, t in v.pairs]
            ppage, pmeta = _page(pairs, pairs_offset, None, self.cap)
            out.update(pairs=ppage, pairs_page=pmeta,
                       note="Map the values of exactly these attribute pairs.")
        else:
            out["note"] = ("End-to-end mode: choose the GDC attribute(s) for each source column with "
                           "map_attribute, then map the values.")
        if self.search_only:
            out["search_only"] = (f"Permissible values are not listed in this session: find them with "
                                  f"search_permissible (at most {self.search_top_k} results per call).")
        out["returned"] = len(page)
        return out

    def profile_column(self, column: str) -> dict:
        col = self._col(column)
        counts = self.view.value_counts(col)
        cells = [canon_value(x) for x in self.view.table[col]]
        present = [x for x in cells if x != ""]
        top = [{"value": val, "count": n, "missing": val == ""} for val, n in counts.most_common(20)]
        lengths = [len(x) for x in present]
        out = {"column": col, "inferred_type": self.view.inferred_type(col), "n_rows": len(cells),
               "n_distinct_values": len([k for k in counts if k != ""]),
               "null_rate": round(1 - len(present) / len(cells), 4) if cells else None,
               "top_values": top, "returned": len(top)}
        if lengths:
            out["length"] = {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)}
            nums = [float(d) for d in (to_decimal(x) for x in present) if d is not None]
            if nums and len(nums) == len(present):
                out["numeric"] = {"min": min(nums), "max": max(nums), "mean": round(sum(nums) / len(nums), 4)}
            out["shapes"] = [{"shape": s, "count": n} for s, n in Counter(shape(x) for x in present).most_common(10)]
            out["casing"] = dict(Counter(casing(x) for x in present))
            out["with_digits"] = sum(any(c.isdigit() for c in x) for x in present)
            out["with_punctuation"] = sum(bool(_PUNCT.search(x)) for x in present)
        return out

    def sample_rows(self, n: int = 10, seed: int = 0, stratify_by: str | None = None) -> dict:
        n = max(1, min(int(n), self.cap))
        t = self.view.table.reset_index().rename(columns={"index": "row_id"})
        if stratify_by:
            self._col(stratify_by)
            groups = [g for _, g in t.groupby(stratify_by, sort=True)]
            per = max(1, math.ceil(n / max(len(groups), 1)))
            s = pd.concat([g.sample(min(len(g), per), random_state=seed) for g in groups]).head(n)
        else:
            s = t.sample(min(n, len(t)), random_state=seed)
        rows = s.to_dict(orient="records")
        return {"rows": rows, "returned": len(rows), "row_cap": self.cap, "seed": seed, "stratify_by": stratify_by}

    def list_values(self, column: str, offset: int = 0, limit: int | None = None) -> dict:
        col = self._col(column)
        vals = [{"value": val, "count": n, "missing": val == ""}
                for val, n in sorted(self.view.value_counts(col).items(), key=lambda kv: (-kv[1], kv[0]))]
        page, meta = _page(vals, offset, limit, self.cap)
        return {"column": col, "values": page, **meta,
                "note": 'The value "" (missing: true) stands for empty cells; it is a value to map too.'}

    # -- vocabulary ----------------------------------------------------------

    def describe_attribute(self, attribute: str) -> dict:
        a = self._attr(attribute)
        e = self.view.schema[a]
        n = len(e.get("value_data") or {})
        return {"attribute": a, "description": e.get("column_description") or "", "enumerated": n > 0,
                "n_permissible_values": n, "returned": 1}

    def list_permissible(self, attribute: str, offset: int = 0, limit: int | None = None) -> dict:
        if self.search_only:
            raise ToolError("listing permissible values is disabled in this session; use search_permissible")
        a = self._attr(attribute)
        vals = [{"value": val, "description": d} for val, d in self.view.permissible_values(a)]
        page, meta = _page(vals, offset, limit, self.cap)
        out = {"attribute": a, "values": page, **meta}
        if not vals:
            out["note"] = "Not enumerated: any value of the right form (e.g. a number) is accepted."
        return out

    def search_attributes(self, query: str, method: str = "token_set", top_k: int = 10) -> dict:
        top_k = max(1, min(int(top_k), self.cap))
        res = []
        for name, e in self.view.schema.items():
            s_name = similarity(query, name.replace("_", " "), method)
            s_desc = similarity(query, e.get("column_description") or "", method) if method not in (
                "exact", "normalized") else 0.0
            res.append((max(s_name, 0.9 * s_desc), s_name, s_desc, name))
        res.sort(key=lambda r: (-r[0], r[3]))
        out = [{"attribute": n, "score": round(s, 4), "name_score": round(sn, 4), "description_score": round(sd, 4),
                "enumerated": bool(self.view.permissible.get(n))} for s, sn, sd, n in res[:top_k]]
        return {"query": query, "method": method, "results": out, "returned": len(out), "row_cap": self.cap}

    def search_permissible(self, attribute: str, value, method: str = "token_set", top_k: int = 10,
                           min_score: float = 0.0) -> dict:
        a = self._attr(attribute)
        vals = self.view.permissible_values(a)
        if not vals:
            raise ToolError(f"{a!r} is not enumerated; it has no permissible values to search")
        top_k = max(1, min(int(top_k), self.search_top_k if self.search_only else self.cap))
        q = canon_value(value)
        scored = sorted(((similarity(q, v, method), v, d) for v, d in vals), key=lambda r: (-r[0], r[1]))
        out = [{"value": v, "score": round(s, 4), "description": d}
               for s, v, d in scored if s >= float(min_score)][:top_k]
        self.searched.update((a, r["value"]) for r in out)
        return {"attribute": a, "query": q, "method": method, "results": out, "returned": len(out),
                "row_cap": self.search_top_k if self.search_only else self.cap}

    # -- model-written code (only with --allow-code) --------------------------

    def _code(self) -> CodeStore:
        if self.code is None:
            raise ToolError("the code tools are disabled in this session")
        return self.code

    def save_code(self, name: str, code: str, description: str = "") -> dict:
        try:
            saved = self._code().save(name, code, description, now_iso())
        except CodeError as e:
            raise ToolError(str(e)) from None
        return {**saved, "lines": len(code.splitlines()), "returned": 1}

    def list_code(self) -> dict:
        idx = self._code().index()
        out = [{"name": n, "versions": len(v), "latest_description": v[-1]["description"]} for n, v in idx.items()]
        page, meta = _page(out, 0, None, self.cap)
        return {"functions": page, **meta}

    def run_code(self, name: str, inputs=None, columns: list[str] | None = None,
                 attributes: list[str] | None = None, version: int | None = None, offset: int = 0,
                 limit: int | None = None) -> dict:
        try:
            code, v = self._code().load(name, version)
        except CodeError as e:
            raise ToolError(str(e)) from None
        data = {"values": {}, "permissible": {}}
        for c in columns or []:
            data["values"][self._col(c)] = sorted(self.view.value_counts(c))
        for a in attributes or []:
            if self.search_only:
                raise ToolError("permissible values cannot be passed to code in search-only mode; use "
                                "search_permissible")
            data["permissible"][self._attr(a)] = [val for val, _ in self.view.permissible_values(a)]
        try:
            res = execute(code, inputs, data)
        except CodeError as e:
            raise ToolError(str(e)) from None
        if not res.get("ok"):
            raise ToolError(f"{name} v{v} failed: {res.get('error')}" + (
                f"\nprinted before the error:\n{res['stdout'][:2000]}" if res.get("stdout") else ""))
        result = res.get("result")
        out = {"name": name, "version": v, "stdout": res.get("stdout", "")[:5000]}
        if isinstance(result, list):
            page, meta = _page(result, offset, limit, self.cap)
            out.update(result=page, **meta)
        elif isinstance(result, dict):
            items, meta = _page(list(result.items()), offset, limit, self.cap)
            out.update(result=dict(items), **meta)
        else:
            out.update(result=result, returned=1)
        return out

    def query(self, sql: str) -> dict:
        if self.sql is None:
            raise ToolError("the SQL tool is disabled in this session")
        return self.sql.query(sql)

    # -- answers -------------------------------------------------------------

    def _check_target_value(self, ta: str, tv) -> tuple[str | None, str | None]:
        """(value to store or None for NONE, warning)."""
        if tv is None:
            return None, None
        tv = str(tv).strip()
        perm = self.view.permissible.get(ta) or frozenset()
        if tv.casefold() in SM_NONE_TOKENS and tv not in perm:
            return None, f'"{tv}" was recorded as NONE (no match); use null for that'
        if perm:
            if self.search_only:
                # only values a search in this session returned, so every answer traces to a logged search;
                # no "did you mean" hint, which would reveal vocabulary outside a search
                if (ta, tv) not in self.searched:
                    raise ToolError(f"{tv!r} was not returned by any search_permissible call on {ta!r} in this "
                                    "session; search for it first (values are case-sensitive)")
                return tv, None
            if tv in perm:
                return tv, None
            near = [p for p in perm if p.casefold() == tv.casefold()]
            hint = f"; did you mean {near[0]!r}?" if near else "; list_permissible shows the allowed values"
            raise ToolError(f"{tv!r} is not a permissible value of {ta!r} (values are case-sensitive){hint}")
        if to_decimal(tv) is None:
            return tv, f"{ta!r} is not enumerated and {tv!r} is not a number; recorded as given"
        return tv, None

    def submit_value_matches(self, matches: list[dict]) -> dict:
        if not isinstance(matches, list) or not matches:
            raise ToolError("matches must be a non-empty list")
        pairs = set(self._pairs())
        accepted, errors, warnings = 0, [], []
        for i, m in enumerate(matches):
            try:
                sa, ta = m.get("source_attribute"), m.get("target_attribute")
                self._col(sa)
                self._attr(ta)
                if (sa, ta) not in pairs:
                    raise ToolError(
                        f"({sa!r} -> {ta!r}) is not one of the given attribute pairs" if self.view.mode == "oracle"
                        else f"({sa!r} -> {ta!r}) is not mapped; call map_attribute first")
                sv = m.get("source_value")
                if sv is None:
                    sv = ""
                if not self.view.has_value(sa, sv):
                    raise ToolError(f"{canon_value(sv)!r} is not a value of source column {sa!r}")
                conf = m.get("confidence")
                if conf is None or isinstance(conf, bool):
                    raise ToolError("confidence (0-1) is required")
                conf = float(conf)
                if not 0 <= conf <= 1 or math.isnan(conf):
                    raise ToolError("confidence must be between 0 and 1")
                tv, warn = self._check_target_value(ta, m.get("target_value"))
                if warn:
                    warnings.append({"index": i, "warning": warn})
                self.store.add_match(sa, ta, sv, tv, conf, m.get("rationale"))
                accepted += 1
            except ToolError as e:
                errors.append({"index": i, "error": str(e)})
            except (TypeError, ValueError) as e:
                errors.append({"index": i, "error": f"invalid entry: {e}"})
        return {"accepted": accepted, "errors": errors, "warnings": warnings, "returned": accepted}

    def map_attribute(self, source_attribute: str, target_attribute: str) -> dict:
        if self.view.mode != "end_to_end":
            raise ToolError("map_attribute is only available in end-to-end mode")
        sa, ta = self._col(source_attribute), self._attr(target_attribute)
        self.store.set_mapping(sa, ta, True)
        return {"mapped": [{"source_attribute": s, "target_attribute": t} for s, t in self.store.mapped_pairs()
                           if s == sa], "returned": 1}

    def unmap_attribute(self, source_attribute: str, target_attribute: str) -> dict:
        if self.view.mode != "end_to_end":
            raise ToolError("unmap_attribute is only available in end-to-end mode")
        if (source_attribute, target_attribute) not in set(self.store.mapped_pairs()):
            raise ToolError(f"({source_attribute!r} -> {target_attribute!r}) is not mapped")
        self.store.set_mapping(source_attribute, target_attribute, False)
        return {"unmapped": {"source_attribute": source_attribute, "target_attribute": target_attribute},
                "returned": 1}

    def get_progress(self, offset: int = 0, limit: int | None = None) -> dict:
        state = self.store.matches()
        rows, tot = [], Counter()
        for sa, ta in self._pairs():
            vals = set(self.view.value_counts(sa)) if sa in self.view.columns else set()
            subs = [state[k] for k in (item_key(sa, ta, x) for x in vals) if k in state]
            n_none = sum(1 for s in subs if s.get("target_value") is None)
            r = {"source_attribute": sa, "target_attribute": ta, "n_values": len(vals),
                 "submitted_values": len(subs) - n_none, "explicit_none": n_none, "remaining": len(vals) - len(subs)}
            rows.append(r)
            for k in ("n_values", "submitted_values", "explicit_none", "remaining"):
                tot[k] += r[k]
        page, meta = _page(rows, offset, limit, self.cap)
        out = {"pairs": page, **meta, "totals": dict(tot)}
        if self.view.mode == "end_to_end":
            mapped_cols = {s for s, _ in self._pairs()}
            out["unmapped_source_columns"] = [c for c in self.view.columns if c not in mapped_cols][: self.cap]
        return out

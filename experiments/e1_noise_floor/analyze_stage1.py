"""E1 Stage 1 analysis (reads logs only; makes no model calls).

  A  margin survey (1a): decision margins (value span and prefix-inclusive), P_greedy under both endpoints'
     default samplers, duplicates, path classes, accuracy by study
  B  two-lineage enumeration (1b): text/decision flips between the batched source and the single-token replay,
     shifts of the decision margin, shifts of the logit gap at every position (call-clustered bootstrap), sign
     of the shift, and whether the batched pre-send reproduced the source bitwise (P2)
  C  transport (1c): /api/chat cold vs /v1 cold, step by step
  D  audits and validation

Usage: python -I analyze_stage1.py <stage1_out_root> [<out_dir>]

Writes stage1_analysis.json, stage1_decisions.csv and stage1_gap_shifts.csv.gz to out_dir (default: the root).
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO, OUT = Path(__file__).resolve().parents[2], Path(sys.argv[1])
DEST = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
DEST.mkdir(parents=True, exist_ok=True)
JSON_OUT = DEST / "stage1_analysis.json"
sys.path.insert(0, str(REPO))
from hbench.decisions import decision_margin, position_margin  # noqa: E402
from hbench.runlog.writer import BlobStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resample import read_run_events  # noqa: E402

THRESHOLDS = (0.05, 0.1, 0.15, 0.25, 0.5)
report: dict = {}


def run_dirs(root: Path):
    return sorted(p.parent for p in root.glob("**/events.driver.jsonl"))


def p_keep(entry, temperature, top_p):
    """P(sampler picks this position's greedy token): top-k(40) -> top-p on T=1 probs -> temperature.
    Visible top-5 only; the remaining mass acts as one pseudo-candidate."""
    tok, _, top = entry
    cands = sorted([(c, math.exp(lp)) for c, lp in top if lp is not None], key=lambda x: -x[1])
    rest = max(0.0, 1 - sum(p for _, p in cands))
    nucleus, cum = [], 0.0
    for c, p in cands + [("<rest>", rest)]:
        nucleus.append((c, p))
        cum += p
        if cum >= top_p:
            break
    w = {c: p ** (1 / temperature) for c, p in nucleus if p > 0}
    return w.get(tok, 0.0) / sum(w.values()) if w else None


def gaps(tokens):
    return [position_margin(t)[0] for t in tokens]


def margin_of(entry, tok):
    """Logprob of tok minus the best other candidate's at one position; tok need not be the token chosen there,
    so the gap of the source's choice can be read off a replay that diverged at this position."""
    cands = {c: lp for c, lp in (entry[2] or []) if lp is not None}
    if entry[1] is not None:
        cands.setdefault(entry[0], entry[1])
    if tok not in cands or len(cands) < 2:
        return None
    return cands[tok] - max(lp for c, lp in cands.items() if c != tok)


def first_divergence(ta, tb):
    return next((i for i, (x, y) in enumerate(zip(ta, tb)) if x[0] != y[0]), None)


def decision_info(tokens, text):
    """Value-span margin, prefix-inclusive margin (positions 0..end of value), P_greedy for both defaults."""
    out = {"value": None, "margin_span": None, "margin_prefix": None, "margin_prefix_raw": None,
           "margin_prefix_free": None, "end": None,
           "answer_token": None, "skip_positions": [], "p_greedy_apichat": None, "p_greedy_v1": None}
    try:
        value = json.loads(text).get("target")
    except (TypeError, ValueError, AttributeError):
        return out
    if not isinstance(value, str) or not tokens:
        return out
    d = decision_margin(tokens, text, value, "target")
    out["value"] = value
    if not d["found"]:
        return out
    end = max(d["span"])
    g = [x for x in gaps(tokens[: end + 1]) if x is not None]
    # A thinking model's first answer token is forced by the format=json grammar (it wanted a newline), so its
    # raw gap is about -20 for every decision; for such outputs margin_prefix skips constrained positions
    # (hbench margin_free). Outputs without thinking (Stage 1, qwen2.5:14b) keep the raw prefix minimum.
    raw = min(g) if g else None
    out.update(margin_span=d["margin_value"], margin_prefix=d["margin_free"] if d["answer_token"] else raw,
               margin_prefix_raw=raw, margin_prefix_free=d["margin_free"], answer_token=d["answer_token"], end=end,
               skip_positions=d["constrained_positions"] if d["answer_token"] else [],
               p_greedy_apichat=math.prod(p_keep(tokens[i], 0.8, 0.9) for i in range(end + 1)),
               p_greedy_v1=math.prod(p_keep(tokens[i], 1.0, 1.0) for i in range(end + 1)))
    return out


def quantiles(x, qs=(0, 0.01, 0.05, 0.1, 0.25, 0.5)):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))])
    return {f"q{q * 100:g}": (float(np.quantile(x, q)) if len(x) else None) for q in qs}


# ------------------------------------------------------------------ A: margin survey
rows, src_calls, path_counts, audits, natural_ram_restores = [], {}, Counter(), {}, []
for rd in run_dirs(OUT / "1a"):
    events = read_run_events(rd)
    header = next(e for e in events if e["type"] == "run_header")
    cond = header["condition"]
    study, mode = cond["data"]["study"], cond["method"]["mode"]
    bs = BlobStore(rd.parent / "_blobs")
    calls = [e for e in events if e["type"] == "llm_call"]
    for c in calls:
        src_calls[(rd.name, c["span_id"])] = (c, bs)
        path_counts[(mode, c["engine"]["path_class"])] += 1
    preds = {json.loads(line)["source_column"]: json.loads(line) for line in open(rd / "predictions.jsonl")}
    by_span = {c["span_id"]: c for c in calls}
    items = pd.read_csv(rd / "items.gdc-sm.csv", keep_default_na=False)
    correct = dict(zip(items.source_column, items.closed_world_correct.astype(str) == "True"))
    annotated = dict(zip(items.source_column, items.annotated.astype(str) == "True"))
    for col, p in preds.items():
        ev = p.get("evidence") or {}
        c = by_span.get(ev.get("span_id"))
        if c is None:
            continue
        toks = bs.get(c["token_logprobs"]) if c.get("token_logprobs") else None
        info = decision_info(toks, (c.get("output") or {}).get("text"))
        g = gaps(toks[: info["end"] + 1]) if info["end"] is not None else []
        g = [x for i, x in enumerate(g) if i not in set(info["skip_positions"])]  # same positions margin_prefix uses
        rows.append(dict(run=rd.name, study=study, mode=mode, column=col, span_id=c["span_id"],
                         input_sha256=c["input_sha256"], path=c["engine"]["path_class"], status=p["status"],
                         legacy_status=p.get("legacy_status"), annotated=annotated.get(col),
                         correct=correct.get(col), evidence_margin=ev.get("margin"),
                         decision_gaps=json.dumps([x for x in g if x is not None]), **info))
    end = next((e for e in events if e["type"] == "run_end"), {})
    audits[rd.name] = end.get("engine_audit") or {}
    for sid in audits[rd.name].get("ram_restores") or []:
        natural_ram_restores.append({"run": rd.name, "span_id": sid})
df = pd.DataFrame(rows)
dfu = df.drop_duplicates("input_sha256")
A = {"n_decisions": len(df), "n_unique_inputs": int(df.input_sha256.nunique()),
     "n_no_value": int(df.value.isna().sum()), "path_classes": {f"{m}:{p}": n for (m, p), n in path_counts.items()},
     "margin_span_equals_evidence": bool(np.allclose(df.margin_span.astype(float).fillna(-1),
                                                     df.evidence_margin.astype(float).fillna(-1))),
     "by_mode": {}}
for mode, g in dfu.groupby("mode"):
    m = g.margin_prefix.dropna()
    pa, pv = g.p_greedy_apichat.dropna(), g.p_greedy_v1.dropna()
    A["by_mode"][mode] = {
        "n_unique": len(g), "margin_prefix_quantiles": quantiles(m), "margin_span_quantiles": quantiles(g.margin_span),
        "n_prefix_margin_below": {str(t): int((m < t).sum()) for t in THRESHOLDS},
        "n_prefix_lower_than_span": int((g.margin_prefix < g.margin_span - 1e-12).sum()),
        "apichat_default_deterministic_share": float((pa >= 0.9995).mean()),
        "apichat_expected_changes_between_two_runs": [float((2 * pa * (1 - pa)).sum()), float((1 - pa ** 2).sum())],
        "v1_default_deterministic_share": float((pv >= 0.9995).mean()),
        "v1_expected_changes_between_two_runs": [float((2 * pv * (1 - pv)).sum()), float((1 - pv ** 2).sum())],
        "accuracy_annotated": float(g[g.annotated == True].correct.mean()) if (g.annotated == True).any() else None,  # noqa: E712
    }
A["n_all_margin_below"] = {str(t): int((df.margin_prefix < t).sum()) for t in THRESHOLDS}
off_path = df[(df.path != "partial") & (df.path != "cold_full")]
A["natural_off_batched_lineage"] = {
    "single_token_calls": off_path[["run", "span_id", "study", "mode", "column", "value", "margin_prefix"]]
    .to_dict("records"), "ram_restores": natural_ram_restores}
A["lowest_margins"] = (dfu.sort_values("margin_prefix").head(25)[
    ["study", "mode", "column", "value", "margin_span", "margin_prefix", "p_greedy_apichat", "correct"]]
    .round(4).to_dict("records"))
A["accuracy_by_study"] = (df[df.annotated == True].groupby(["mode", "study"]).correct  # noqa: E712
                          .agg(["sum", "count"]).reset_index().to_dict("records"))
report["A_margin_survey"] = A
df.to_csv(DEST / "stage1_decisions.csv", index=False)

# ------------------------------------------------------------------ B: two-lineage enumeration
B = {"n_replay_runs": 0, "n_replayed": 0}
comp_rows, shift_rows, pre_rows, measured = [], [], [], {}
for rd in run_dirs(OUT / "1b"):
    B["n_replay_runs"] += 1
    events = read_run_events(rd)
    header = next(e for e in events if e["type"] == "run_header")
    src_run = Path(header["task"]["source_run"]).name
    bs = BlobStore(rd.parent / "_blobs")
    audits[rd.name] = (next((e for e in events if e["type"] == "run_end"), {}).get("engine_audit") or {})
    restored = set(audits[rd.name].get("ram_restores") or [])
    if (rd / "replay_compare.csv").exists():
        cmp = pd.read_csv(rd / "replay_compare.csv")
        cmp["source_run"] = src_run
        comp_rows.append(cmp)
    for e in events:
        if e["type"] != "llm_call":
            continue
        src_key = (src_run, e.get("replay_of") or e.get("parent_id"))
        if src_key not in src_calls:
            continue
        sc, sbs = src_calls[src_key]
        ta, tb = sbs.get(sc["token_logprobs"]), bs.get(e["token_logprobs"])
        if e.get("caller") == "replay:pre":  # batched re-send: P2 says bitwise identical to the source
            pre_rows.append({"source": src_key, "path": e["engine"]["path_class"], "bitwise": ta == tb,
                             "cached_source": sc["engine"]["cached_tokens"], "cached_pre": e["engine"]["cached_tokens"],
                             "ram_restore": e["span_id"] in restored, "tokens": tb})
            continue
        B["n_replayed"] += 1
        measured[src_key] = tb
        # signed gap of the source's choice, at every position up to and including the first divergence
        k = first_divergence(ta, tb)
        info = decision_info(ta, sc["output"]["text"])
        pair = f'{sc["engine"]["path_class"]}->{e["engine"]["path_class"]}'
        for i, (x, y) in enumerate(zip(ta[: None if k is None else k + 1], tb)):
            ga, gb = margin_of(x, x[0]), margin_of(y, x[0])
            if ga is not None and gb is not None:
                shift_rows.append({"call": f"{src_key[0]}|{src_key[1]}", "pair": pair, "pos": i, "gap": ga,
                                   "dgap": gb - ga, "at_divergence": i == k,
                                   "in_decision_prefix": info["end"] is not None and i <= info["end"]})
cmp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
sh = pd.DataFrame(shift_rows)
if len(cmp):
    B.update({
        "paths": cmp.groupby(["path_source", "path_replay"]).size().rename("n").reset_index().to_dict("records"),
        "text_changed": int((~cmp.same_text.astype(bool)).sum()),
        "decision_flipped": int(cmp.decision_flipped.fillna(False).astype(bool).sum()),
        "flips": cmp[cmp.decision_flipped.fillna(False).astype(bool) | ~cmp.same_text.astype(bool)][
            ["source_run", "source_call", "decision", "decision_replay", "margin_source", "margin_replay",
             "first_divergent_token"]].to_dict("records"),
        "d_margin_quantiles_abs": quantiles(cmp.d_margin.abs(), (0.5, 0.9, 0.99, 1.0)),
        "d_margin_sign": {"negative": int((cmp.d_margin < 0).sum()), "positive": int((cmp.d_margin > 0).sum()),
                          "zero": int((cmp.d_margin == 0).sum())},
    })
def shift_stats(s: pd.DataFrame) -> dict:
    """|gap shift| distribution of one path pair, with a call-clustered bootstrap (positions within a call are
    correlated), by source-gap bin, and the sign of the non-zero shifts."""
    nz = s[s.dgap != 0]
    calls = s.call.unique()
    rng = np.random.default_rng(0)
    groups = {c: g.dgap.abs().to_numpy() for c, g in s.groupby("call")}
    boots = {"median": [], "p90": [], "p99": [], "max": []}
    for _ in range(1000):
        x = np.concatenate([groups[c] for c in rng.choice(calls, size=len(calls))])
        for k, f in (("median", np.median), ("p90", lambda v: np.quantile(v, .9)),
                     ("p99", lambda v: np.quantile(v, .99)), ("max", np.max)):
            boots[k].append(f(x))
    out = {"n_positions": len(s), "n_calls": int(len(calls)), "share_zero": float((s.dgap == 0).mean()),
           "abs_quantiles": quantiles(s.dgap.abs(), (0.5, 0.9, 0.99, 0.999, 1.0)),
           "bootstrap_ci95": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boots.items()},
           "decision_prefix_positions": quantiles(s[s.in_decision_prefix].dgap.abs(), (0.5, 0.9, 0.99, 1.0)),
           "nonzero_share_negative": float((nz.dgap < 0).mean()) if len(nz) else None, "by_gap_bin": []}
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 3), (3, 6), (6, 1e9)]:
        g = s[(s.gap >= lo) & (s.gap < hi)]
        if len(g):
            gnz = g[g.dgap != 0]
            out["by_gap_bin"].append({"gap_from": lo, "gap_to": hi, "n": len(g),
                                      **quantiles(g.dgap.abs(), (0.5, 0.9, 1.0)),
                                      "nonzero_share_negative": float((gnz.dgap < 0).mean()) if len(gnz) else None})
    return out


if len(sh):
    B["gap_shift"] = {pair: shift_stats(g) for pair, g in sh.groupby("pair")}
    sh.to_csv(DEST / "stage1_gap_shifts.csv.gz", index=False)
if pre_rows:
    pr = pd.DataFrame(pre_rows)
    batched = pr[pr.path == "partial"]
    moved = batched[batched.cached_pre != batched.cached_source]
    single = pr[pr.path == "single_token"]
    B["presend_batched_bitwise"] = {
        "n": len(pr), "bitwise": int(pr.bitwise.sum()), "paths": pr.path.value_counts().to_dict(),
        "batched": {"n": len(batched), "bitwise": int(batched.bitwise.sum()),
                    "split_point_moved": len(moved), "split_point_moved_bitwise": int(moved.bitwise.sum()),
                    "cached_source_range": [int(batched.cached_source.min()), int(batched.cached_source.max())],
                    "cached_pre_range": [int(batched.cached_pre.min()), int(batched.cached_pre.max())]},
        # a pre-send that hit the host-RAM prompt cache is itself single-token, i.e. already the second lineage
        "single_token": {"n": len(single), "ram_restore": int(single.ram_restore.sum()),
                         "bitwise_vs_source": int(single.bitwise.sum()),
                         "bitwise_vs_measured_send": int(sum(r.tokens == measured.get(r.source)
                                                             for r in single.itertuples()))}}

# noise model check: a decision flips iff gap + shift < 0 at some position up to the end of its value. With the
# measured signed-shift distribution (leave-one-call-out for replayed calls), the expected number of flips
# should match the observed one.
if len(sh) and len(cmp):
    # 14b's resends all ran partial->single_token; a long (thinking) answer resends on other path pairs, so fall
    # back to pooling every pair there
    pooled = sh[sh.pair == "partial->single_token"]
    if not len(pooled):
        pooled = sh
    B["noise_model_pairs"] = sorted(pooled.pair.unique().tolist())
    all_d = np.sort(pooled.dgap.to_numpy())
    own = {c: np.sort(g.dgap.to_numpy()) for c, g in pooled.groupby("call")}

    def p_below(x, call=None):
        n_all, below = len(all_d), np.searchsorted(all_d, x)
        if call in own:
            n_all, below = n_all - len(own[call]), below - np.searchsorted(own[call], x)
        return below / n_all

    def p_flip_lineage(gaps_json, call=None):
        g = json.loads(gaps_json) if isinstance(gaps_json, str) else []
        return 1 - math.prod(1 - p_below(-x, call) for x in g) if g else None

    span_key = {(r.run, r.span_id): r for r in df.itertuples()}
    cal = []
    for r in cmp.itertuples():
        d = span_key.get((r.source_run, r.source_span))
        if d is None:
            continue
        cal.append({"margin": d.margin_prefix, "flipped": str(r.decision_flipped) == "True",
                    "p": p_flip_lineage(d.decision_gaps, f"{r.source_run}|{r.source_span}")})
    cal = pd.DataFrame(cal)
    df["p_flip_lineage"] = [p_flip_lineage(x) for x in df.decision_gaps]
    replayed = {(r.source_run, r.source_span) for r in cmp.itertuples()}
    low = df[df.margin_prefix < 0.5]
    bins = [0, 0.01, 0.03, 0.1, 0.5, np.inf]
    cal["bin"] = pd.cut(cal.margin, bins, right=False)
    B["noise_model_check"] = {
        "replayed_observed_flips": int(cal.flipped.sum()), "replayed_expected_flips_loo": float(cal.p.sum()),
        "by_margin_bin": [{"margin_from": float(b.left), "margin_to": float(b.right), "n": len(g),
                           "observed": int(g.flipped.sum()), "expected": float(g.p.sum())}
                          for b, g in cal.groupby("bin", observed=True)],
        "all_decisions_expected_flips": float(df.p_flip_lineage.sum()),
        "decisions_margin_below_0.5": len(low),
        "decisions_margin_below_0.5_replayed": int(sum((r.run, r.span_id) in replayed for r in low.itertuples())),
        "max_abs_shift": float(np.abs(all_d).max()),
        "decisions_margin_above_max_shift": int((df.margin_prefix > np.abs(all_d).max()).sum())}
    df.to_csv(DEST / "stage1_decisions.csv", index=False)
report["B_two_lineages"] = B

# ------------------------------------------------------------------ C: transport
C = {}
tr = {}
for rd in run_dirs(OUT / "1c"):
    events = read_run_events(rd)
    header = next(e for e in events if e["type"] == "run_header")
    tr[header["condition"]["engine"]["transport"]] = (
        [e for e in events if e["type"] == "llm_call"], BlobStore(rd.parent / "_blobs"))
    audits[rd.name] = (next((e for e in events if e["type"] == "run_end"), {}).get("engine_audit") or {})
if {"ollama", "openai_v1"} <= tr.keys():
    (ca, ba), (cb, bb) = tr["ollama"], tr["openai_v1"]
    steps = []
    for x, y in zip(ca, cb):
        ta, tb = ba.get(x["token_logprobs"]), bb.get(y["token_logprobs"])
        d_any = [abs(lp - dict(q[2]).get(c, lp)) for p, q in zip(ta, tb) for c, lp in p[2]
                 if c in dict(q[2])] if [t[0] for t in ta] == [t[0] for t in tb] else []
        steps.append({"step": x["step"], "prompt_tokens": [x["engine"]["prompt_tokens"], y["engine"]["prompt_tokens"]],
                      "paths": [x["engine"]["path_class"], y["engine"]["path_class"]],
                      "same_text": x["output"]["text"] == y["output"]["text"],
                      "same_tokens": [t[0] for t in ta] == [t[0] for t in tb], "bitwise": ta == tb,
                      "max_abs_d_any": max(d_any, default=None)})
    st = pd.DataFrame(steps)
    C = {"n_steps": len(st), "same_prompt_tokens": int(st.prompt_tokens.map(lambda p: p[0] == p[1]).sum()),
         "same_text": int(st.same_text.sum()), "same_tokens": int(st.same_tokens.sum()),
         "bitwise": int(st.bitwise.sum()), "max_abs_d_any": st.max_abs_d_any.max(),
         "paths": Counter(tuple(p) for p in st.paths).most_common()}
    # P2 again: each cold call against the same input in the natural (cached, batched) 1a run
    nat_runs = set(df[(df.study == "clark") & (df["mode"] == "shortlist")].run)
    nat = {c["input_sha256"]: (c, b) for (run, _), (c, b) in src_calls.items() if run in nat_runs}
    pairs = [(x, nat[x["input_sha256"]]) for x in ca if x["input_sha256"] in nat]
    C["cold_vs_natural_1a"] = {
        "n": len(pairs), "bitwise": int(sum(ba.get(x["token_logprobs"]) == b.get(c["token_logprobs"])
                                            for x, (c, b) in pairs)),
        "natural_paths": Counter(c["engine"]["path_class"] for _, (c, _b) in pairs).most_common()}
report["C_transport"] = C

# ------------------------------------------------------------------ E: back-to-back repeats (1e, optional)
E = []
nat_by_mode = {m: {c["input_sha256"]: (c, b) for (run, _), (c, b) in src_calls.items()
                   if run in set(df[(df.study == "clark") & (df["mode"] == m)].run)} for m in ("full", "shortlist")}
prev = {}
for rd in sorted(run_dirs(OUT / "1e"), key=lambda r: r.name.rsplit("-", 2)[1]):
    events = read_run_events(rd)
    header = next(e for e in events if e["type"] == "run_header")
    mode = header["condition"]["method"]["mode"]
    bs = BlobStore(rd.parent / "_blobs")
    audits[rd.name] = (next((e for e in events if e["type"] == "run_end"), {}).get("engine_audit") or {})
    restored = set(audits[rd.name].get("ram_restores") or [])
    calls = [e for e in events if e["type"] == "llm_call"]
    toks = {c["input_sha256"]: bs.get(c["token_logprobs"]) for c in calls}
    vs_nat = [toks[c["input_sha256"]] == b.get(n["token_logprobs"])
              for c in calls for n, b in [nat_by_mode[mode].get(c["input_sha256"], (None, None))] if n]
    same_text_nat = [c["output"]["text"] == n["output"]["text"]
                     for c in calls for n, _b in [nat_by_mode[mode].get(c["input_sha256"], (None, None))] if n]

    def target(text):
        try:
            return json.loads(text).get("target")
        except (TypeError, ValueError, AttributeError):
            return None
    changed = [c["step"] for c in calls for n, _b in [nat_by_mode[mode].get(c["input_sha256"], (None, None))]
               if n and target(c["output"]["text"]) != target(n["output"]["text"])]
    p = prev.get(mode)
    E.append({"run": rd.name, "mode": mode, "n_calls": len(calls),
              "paths": Counter(c["engine"]["path_class"] for c in calls).most_common(),
              "ram_restores": len(restored),
              "ram_restored_steps": [c["step"] for c in calls if c["span_id"] in restored],
              "bitwise_vs_1a": int(sum(vs_nat)), "same_text_vs_1a": int(sum(same_text_nat)),
              "decisions_changed_vs_1a": changed,
              "bitwise_vs_previous_repeat": None if p is None else int(sum(toks[k] == p.get(k) for k in toks))})
    prev[mode] = toks
report["E_back_to_back"] = E

# ------------------------------------------------------------------ D: audits and validation
agg = Counter()
for name, s in audits.items():
    for k in ("foreign_tasks", "runner_starts", "other_requests"):
        agg[k] += s.get(k) or 0
    for k in ("unmatched_calls", "truncated", "ram_restores", "sampler_mismatches", "cache_disagreements"):
        agg[k] += len(s.get(k) or [])
    agg["unavailable"] += "unavailable" in s
report["D_audits"] = {"n_runs": len(audits), **agg}
vs = OUT / "validate_summary.json"
report["D_validate"] = json.loads(vs.read_text()) if vs.exists() else "not yet written"

JSON_OUT.write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str)[:12000])

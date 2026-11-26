"""One readable HTML page per run: `gatekeeper report <run_dir>` (Sprint 11).

Built only from the files a run already wrote (summary.json, the contract result, the cleaning
log, quality.json, anomalies, drift, decisions.json, impact). Rules for the page:
  - self-contained: inline CSS and inline SVG, no scripts, fonts, images or links that load
    anything, so it opens offline and can be mailed as one file
  - plain words first: the overall status as one sentence a non-technical reader understands
  - no raw personal data: it only shows what the reports hold, and those already mask PII
    values (contracts, drift) or never use PII columns (anomalies); every text is HTML-escaped
  - prints well (@media print: no backgrounds, no page breaks inside tables)
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

STATUS_WORDS = {
    "passed": ("ok", "This file passed every check that was run. It can be used."),
    "stopped at a gate": ("stop", "This file was STOPPED: it broke a rule that must not be broken. "
                                  "Do not use it until the problem below is fixed."),
    "error": ("error", "The run did not finish because a step failed. The results below are "
                       "incomplete."),
}
STEP_WORDS = {
    "profile": "Describe the file (types, missing values, duplicates)",
    "validate": "Check the file against its contract (the agreed rules)",
    "clean": "Clean the file (missing codes, duplicates, formats, filling gaps)",
    "mask_pii": "Find and protect personal data",
    "quality": "Score data quality before and after cleaning",
    "anomalies": "List the most unusual rows and why",
    "drift": "Compare with a reference file: did the data change?",
    "impact": "Measure whether cleaning helps a prediction model",
}
CSS = """
:root{--ink:#1d2330;--muted:#5b6475;--line:#d9dde5;--ok:#1b7f4b;--warn:#a86200;
--bad:#b3261e;--bg:#ffffff;--soft:#f4f6f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}
.meta{color:var(--muted);font-size:13px}
.banner{border-radius:8px;padding:14px 16px;margin:16px 0;font-size:16px;border:1px solid}
.banner.ok{background:#e9f6ef;border-color:var(--ok)}.banner.stop{background:#fdeceb;
border-color:var(--bad)}.banner.error{background:#fff4e5;border-color:var(--warn)}
table{width:100%;border-collapse:collapse;margin:8px 0;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--soft);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.passed,.pass,.stable{background:#e9f6ef;color:var(--ok)}.failed,.fail,.major,.error{
background:#fdeceb;color:var(--bad)}.warn,.moderate{background:#fff4e5;color:var(--warn)}
.skipped,.not_run{background:var(--soft);color:var(--muted)}
.note{background:var(--soft);border-left:4px solid var(--warn);padding:8px 12px;margin:8px 0}
.wrap{overflow-x:auto}svg text{font:12px sans-serif;fill:var(--ink)}
@media print{body{font-size:12px}.banner,.pill,th{background:none!important}
table,tr{page-break-inside:avoid}h2{page-break-after:avoid}main{max-width:none}}
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find(run_dir: Path, suffix: str) -> dict | None:
    hits = sorted(run_dir.glob(f"*{suffix}"))
    return _load(hits[0]) if hits else None


def _table(head: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(
        f"<td class='num'>{_e(c)}</td>" if i in numeric else f"<td>{c}</td>"
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f"<div class='wrap'><table><thead><tr>{th}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def _pill(status: str) -> str:
    return f"<span class='pill {_e(status)}'>{_e(status.replace('_', ' '))}</span>"


def _bars(items: list[tuple[str, float]], max_value: float, fmt: str = "{:.2f}") -> str:
    """A horizontal bar chart as inline SVG."""
    if not items:
        return ""
    h, label_w, bar_w = 22, 170, 520
    rows = []
    for i, (label, v) in enumerate(items):
        w = 0 if max_value <= 0 else max(1, bar_w * min(v, max_value) / max_value)
        y = i * h
        rows.append(f"<text x='0' y='{y + 15}'>{_e(label[:26])}</text>"
                    f"<rect x='{label_w}' y='{y + 4}' width='{w:.1f}' height='14' fill='#5b7bd5'/>"
                    f"<text x='{label_w + w + 6:.1f}' y='{y + 15}'>{_e(fmt.format(v))}</text>")
    view = f"0 0 {label_w + bar_w + 70} {h * len(items)}"
    return f"<svg role='img' width='100%' viewBox='{view}'>{''.join(rows)}</svg>"


def _decisions(dec: dict | None) -> str:
    if not dec or not dec.get("decisions"):
        return ("<p>No value needs a person's decision: every code found is handled by its "
                "default.</p>")
    rows = []
    for d in dec["decisions"]:
        rate = ("-" if d.get("label_rate_in") is None else
                f"{d['label_rate_in']:.1%} vs {d['label_rate_rest']:.1%}")
        rows.append([_e(d["column"]), _e(d["value"]), _e(d["kind"].replace("_", " ")), d["rows"],
                     _e(rate), _e(d["current_default"]), _e(d["recommended"] or "a person decides"),
                     _e(d.get("configured") or "not set yet")])
    intro = ("<p>The gatekeeper is <b>not sure</b> about these values. Each one needs a person "
             "to say what it means. The evidence is how often the target is positive in those "
             "rows compared with the other rows.</p>")
    if dec.get("target") is None:
        intro += "<p class='note'>No target was given, so only counts are shown.</p>"
    head = ["column", "value", "question", "rows", "target rate: these rows vs the rest",
            "today", "recommended", "confirmed in config"]
    return intro + _table(head, rows, {3})


def build_report(run_dir: str | Path) -> str:
    """The HTML page (a string) for one run folder."""
    run_dir = Path(run_dir)
    summary = _load(run_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"{run_dir}: no summary.json (is this a run folder?)")
    steps = {s["step"]: s for s in summary["steps"]}
    tone, sentence = STATUS_WORDS.get(summary["status"], ("error", summary["status"]))
    parts = [f"<h1>Data Gatekeeper report: {_e(summary['name'])}</h1>",
             f"<div class='meta'>File {_e(Path(summary['input']).name)} · run started "
             f"{_e(summary['started_at'])} · {_e(summary['total_seconds'])} s</div>",
             f"<div class='banner {tone}'><b>{_e(sentence)}</b></div>"]
    stopped = [s for s in summary["steps"] if s["status"] in ("failed", "error")]
    if stopped:
        s = stopped[0]
        parts.append(f"<p>It stopped at <b>{_e(STEP_WORDS.get(s['step'], s['step']))}</b>"
                     + (f": {_e(s['error'])}" if s.get("error") else ".") + "</p>")

    parts.append("<h2>What was done</h2>")
    parts.append(_table(["step", "what it means", "result", "seconds"], [
        [_e(s["step"]), _e(STEP_WORDS.get(s["step"], "")), _pill(s["status"]),
         s.get("seconds", "-")] for s in summary["steps"]], {3}))

    contract = _find(run_dir, "_contract.json")
    if contract:
        c = contract["counts"]
        parts.append("<h2>Contract: the agreed rules</h2>")
        parts.append(f"<p>{c['passed']} of {c['rules']} rules passed, {c['failed_error']} broke a "
                     f"must-have rule, {c['failed_warn']} gave a warning.</p>")
        bad = [r for r in contract["results"] if r["status"] in ("fail", "warn")]
        if bad:
            parts.append(_table(["rule", "column", "result", "what we saw", "what was expected"], [
                [_e(r["rule"]), _e(r["column"] or "whole table"), _pill(r["status"]),
                 _e(r["observed"]), _e(r["expected"])] for r in bad]))

    decisions = _load(run_dir / "decisions.json")
    parts.append("<h2>Decisions for a person</h2>")
    parts.append(_decisions(decisions))

    quality = _load(run_dir / "quality.json")
    if quality:
        parts.append("<h2>Data quality score (0-100)</h2>")
        if "before" in quality:
            parts.append(_bars([("before cleaning", quality["before"]),
                                ("after cleaning", quality["after"])], 100))
            parts.append(f"<p>Change: {quality['delta']:+.2f} points.</p>")
        else:
            parts.append(_bars([("this file", quality["score"])], 100))
        parts.append("<p class='note'><b>A higher quality score does not mean a better "
                     "model.</b> The score counts what cleaning changed, not whether those "
                     "changes were right for your purpose. On one census dataset the score went "
                     "up to 100 while the prediction model got worse, because two codes that "
                     "carried meaning were removed. Check the decisions above.</p>")

    anomalies = _find(run_dir, "_anomalies.json")
    if anomalies:
        method = next(iter(anomalies["methods"]))
        m = anomalies["methods"][method]
        parts.append("<h2>Most unusual rows</h2>")
        parts.append(f"<p>{m['flagged']} of {anomalies['rows']} rows "
                     f"({m['flagged_share']:.1%}) look unusual ({_e(method)}). Unusual is not "
                     "the same as wrong: a person should look at them. Rows are numbered from 0 "
                     "(the first data row).</p>")
        rows = []
        for t in m["top"][:5]:
            why = "; ".join(f"{r['column']} = {r['value']} (typical {r['median']})"
                            for r in t["reasons"])
            rows.append([t["rank"], t["row"], _e(why)])
        parts.append(_table(["rank", "row", "why it stands out"], rows, {0, 1}))

    drift = _find(run_dir, "_drift.json")
    if drift:
        parts.append("<h2>Did the data change? (drift)</h2>")
        cnt = drift["counts"]
        parts.append(f"<p>Compared with {_e(drift['reference']['name'])}: {cnt['major']} "
                     f"columns changed a lot, {cnt['moderate']} a little, {cnt['stable']} stayed "
                     "the same.</p>")
        top = [(c["column"], c["psi"]) for c in drift["columns"] if c["psi"] is not None][:8]
        major = drift["thresholds"]["psi_major"]
        parts.append(_bars(top, max(major * 2, max((v for _, v in top), default=0)), "{:.3f}"))
        parts.append(f"<p class='meta'>PSI: below {drift['thresholds']['psi_moderate']} = stable, "
                     f"above {major} = a big change.</p>")

    impact = steps.get("impact")
    if impact and impact.get("result"):
        parts.append("<h2>Does cleaning help a prediction model?</h2>")
        parts.append(_table(["model", "ROC-AUC with cleaning minus without (95% interval)"], [
            [_e(k.replace(" ROC-AUC B-A", "")), _e(v)] for k, v in impact["result"].items()]))
        parts.append("<p class='meta'>A positive number means cleaning helped; an interval that "
                     "includes 0 means no clear difference.</p>")

    body = "\n".join(parts)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Gatekeeper report: {_e(summary['name'])}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>\n")


def write_report(run_dir: str | Path) -> Path:
    """Write run_dir/report.html and return its path."""
    run_dir = Path(run_dir)
    path = run_dir / "report.html"
    path.write_text(build_report(run_dir), encoding="utf-8")
    return path

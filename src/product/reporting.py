"""Dependency-free HTML views for GFS Studio product reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping


def _value(value: Any, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def render_match_html(report: Mapping[str, Any]) -> str:
    fixture, result = report["fixture"], report["result"]
    layers = report.get("layers") or {}
    psychology = layers.get("psychology") or {}
    world_model = layers.get("world_model") or {}
    cognition = layers.get("cognition") or {}
    evidence = report.get("evidence_snapshot") or {}
    timeline = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in report.get("timeline") or []
    ) or "<li>No major timeline events recorded.</li>"
    raw_json = html.escape(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(fixture['home'])} vs {html.escape(fixture['away'])} · GFS Studio</title>
<style>
:root{{--ink:#e9f1ff;--muted:#9cacbf;--panel:#111b2c;--line:#26344a;--accent:#67e8b5;--warn:#ffc66d}}
*{{box-sizing:border-box}}body{{margin:0;background:#08111f;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:36px 22px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:24px}}
h1{{font-size:clamp(28px,6vw,58px);margin:0;letter-spacing:-.04em}}h2{{margin:0 0 14px;font-size:18px}}.mode{{color:var(--accent);text-transform:uppercase;letter-spacing:.12em}}
.score{{font-size:clamp(44px,9vw,92px);font-weight:800;white-space:nowrap}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:24px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.metric{{font-size:28px;font-weight:700;margin-top:5px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}ul{{margin:0;padding-left:20px}}details{{margin-top:24px}}pre{{overflow:auto;background:#050b14;padding:18px;border-radius:12px;color:#b9c7da}}
.ok{{color:var(--accent)}}.off{{color:var(--warn)}}@media(max-width:650px){{header,.split{{display:block}}.score{{margin-top:20px}}}}
</style></head><body><main>
<header><div><div class="mode">{html.escape(report['studio']['mode'])} mode · GFS Studio</div><h1>{html.escape(fixture['home'])}<br>{html.escape(fixture['away'])}</h1></div><div class="score">{_value(result['score']['home'],0)}–{_value(result['score']['away'],0)}</div></header>
<section class="grid">
<article class="card"><div class="label">Expected goals</div><div class="metric">{_value(result['xg']['home'])} – {_value(result['xg']['away'])}</div></article>
<article class="card"><div class="label">Possession</div><div class="metric">{100*result['possession']['home']:.0f}% – {100*result['possession']['away']:.0f}%</div></article>
<article class="card"><div class="label">Passes</div><div class="metric">{_value(result['passes']['home'],0)} – {_value(result['passes']['away'],0)}</div></article>
<article class="card"><div class="label">Shots</div><div class="metric">{_value(result['shots']['home'],0)} – {_value(result['shots']['away'],0)}</div></article>
</section>
<section class="split">
<article class="card"><h2>System layers</h2><p>World model: <b class="{'ok' if world_model.get('enabled') else 'off'}">{'active' if world_model.get('enabled') else 'stable fallback'}</b><br>Cognition: <b class="{'ok' if cognition.get('enabled') else 'off'}">{'active' if cognition.get('enabled') else 'off'}</b><br>Shot planning: <b class="off">{html.escape(str(evidence.get('shot_fallback') or 'learned head'))}</b></p></article>
<article class="card"><h2>Match psychology</h2><p>Crowd field: <b>{_value(psychology.get('crowd_field'))}</b><br>Coach stress: <b>{_value((psychology.get('coach_stress') or {{}}).get('home'))} / {_value((psychology.get('coach_stress') or {{}}).get('away'))}</b><br>Tactical drift: <b>{_value((psychology.get('tactical_drift') or {{}}).get('home'))} / {_value((psychology.get('tactical_drift') or {{}}).get('away'))}</b></p></article>
</section>
<section class="card" style="margin-top:14px"><h2>Timeline</h2><ul>{timeline}</ul></section>
<details><summary>Complete auditable report</summary><pre>{raw_json}</pre></details>
</main></body></html>"""


def write_match_html(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_match_html(report), encoding="utf-8")
    return target

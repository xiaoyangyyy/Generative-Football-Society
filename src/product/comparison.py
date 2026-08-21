"""Auditable shared-seed tactical contrasts for GFS Studio."""

from __future__ import annotations

import html
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping

from src.product.match_plan import PLAYABLE_TACTICS


class PairingError(ValueError):
    """Two reports cannot form the requested structural pair."""


METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("score_home", ("result", "score", "home")),
    ("score_away", ("result", "score", "away")),
    ("xg_home", ("result", "xg", "home")),
    ("xg_away", ("result", "xg", "away")),
    ("possession_home", ("result", "possession", "home")),
    ("possession_away", ("result", "possession", "away")),
    ("passes_home", ("result", "passes", "home")),
    ("passes_away", ("result", "passes", "away")),
    ("shots_home", ("result", "shots", "home")),
    ("shots_away", ("result", "shots", "away")),
    ("crowd_field", ("layers", "psychology", "crowd_field")),
    ("coach_stress_home", ("layers", "psychology", "coach_stress", "home")),
    ("coach_stress_away", ("layers", "psychology", "coach_stress", "away")),
    ("tactical_drift_home", ("layers", "psychology", "tactical_drift", "home")),
    ("tactical_drift_away", ("layers", "psychology", "tactical_drift", "away")),
    ("wm_action_opportunities", ("layers", "world_model", "action_adoption", "opportunities")),
    ("wm_influenced_opportunities", ("layers", "world_model", "action_adoption", "influenced_opportunities")),
    ("wm_counterfactual_changes", ("layers", "world_model", "action_adoption", "counterfactual_action_changes")),
    ("wm_expected_changes", ("layers", "world_model", "action_adoption", "expected_counterfactual_action_changes")),
    ("wm_mean_probability_shift", ("layers", "world_model", "action_adoption", "mean_recommended_probability_shift")),
)
MAX_PAIRED_REPLAY_EVENTS_PER_SIDE = 240
MAX_PAIRED_REPLAY_FRAMES = 240


def _nested(report: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_text(value: Any, maximum: int = 100) -> str:
    text = str(value or "").strip()
    return "".join(char for char in text if ord(char) >= 32)[:maximum]


def _replay_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    return [min(1.0, max(0.0, x)), min(1.0, max(0.0, y))]


def _bounded_replay_side(
    report: Mapping[str, Any], *, home: str, away: str,
) -> dict[str, Any]:
    replay = report.get("replay") or {}
    if not isinstance(replay, Mapping) or not replay.get("available"):
        return {"available": False, "reason": "replay_unavailable", "events": []}
    raw_events = replay.get("events") or []
    if not isinstance(raw_events, (list, tuple)):
        return {
            "available": False, "reason": "invalid_replay_event_collection",
            "events": [],
        }
    events = []
    invalid_rows = 0
    for raw in raw_events[:MAX_PAIRED_REPLAY_EVENTS_PER_SIDE]:
        if not isinstance(raw, Mapping):
            invalid_rows += 1
            continue
        event_type = _safe_text(raw.get("type"), 20).lower()
        team = _safe_text(raw.get("team"))
        start, end = _replay_point(raw.get("start")), _replay_point(raw.get("end"))
        t_sec = _finite(raw.get("t_sec"))
        if (
            event_type not in {"pass", "shot"}
            or team not in {home, away}
            or start is None or end is None or t_sec is None
        ):
            invalid_rows += 1
            continue
        t_sec = min(8_000.0, max(0.0, t_sec))
        wm_link = raw.get("world_model_link") or {}
        if not isinstance(wm_link, Mapping):
            wm_link = {}
        wm_linked = bool(wm_link) or bool(raw.get("wm_linked"))
        wm_changed = bool(wm_link.get("policy_changed_action")) or bool(
            raw.get("wm_changed")
        )
        events.append({
            "source_index": len(events), "t_sec": t_sec,
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "type": event_type, "team": team,
            "actor": _safe_text(raw.get("actor")),
            "target": _safe_text(raw.get("target")),
            "kind": _safe_text(raw.get("kind"), 40),
            "outcome": _safe_text(raw.get("outcome"), 40).upper(),
            "start": start, "end": end,
            "wm_linked": wm_linked, "wm_changed": wm_changed,
        })
    events.sort(key=lambda event: (event["t_sec"], event["source_index"]))
    for index, event in enumerate(events):
        event["event_index"] = index
    return {
        "available": bool(events),
        "reason": "available" if events else "no_valid_replay_events",
        "events": events, "invalid_rows": invalid_rows,
        "input_truncated": len(raw_events) > (
            MAX_PAIRED_REPLAY_EVENTS_PER_SIDE
        ),
    }


def _evenly_select(values: list[float], count: int) -> list[float]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    positions = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[position] for position in sorted(positions)]


def _select_shared_timepoints(
    baseline_events: list[dict[str, Any]],
    treatment_events: list[dict[str, Any]],
) -> tuple[list[float], bool]:
    all_events = baseline_events + treatment_events
    all_times = sorted({float(event["t_sec"]) for event in all_events})
    if len(all_times) <= MAX_PAIRED_REPLAY_FRAMES:
        return all_times, False
    priority_times = sorted({
        float(event["t_sec"]) for event in all_events
        if event["type"] == "shot"
        or event["outcome"] in {"GOAL", "SAVED", "INTERCEPTED"}
        or event["wm_changed"]
    })
    if len(priority_times) >= MAX_PAIRED_REPLAY_FRAMES:
        return _evenly_select(priority_times, MAX_PAIRED_REPLAY_FRAMES), True
    priority_set = set(priority_times)
    remaining = [time for time in all_times if time not in priority_set]
    selected = priority_times + _evenly_select(
        remaining, MAX_PAIRED_REPLAY_FRAMES - len(priority_times),
    )
    return sorted(set(selected)), True


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "actions": len(events),
        "passes": sum(event["type"] == "pass" for event in events),
        "shots": sum(event["type"] == "shot" for event in events),
        "goals": sum(event["outcome"] == "GOAL" for event in events),
        "wm_changed": sum(bool(event["wm_changed"]) for event in events),
    }


def build_synchronized_paired_replay(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any],
    *, home: str, away: str,
) -> dict[str, Any]:
    """Build a bounded clock-aligned view without claiming event correspondence."""
    baseline_side = _bounded_replay_side(baseline, home=home, away=away)
    treatment_side = _bounded_replay_side(treatment, home=home, away=away)
    if not baseline_side["available"] or not treatment_side["available"]:
        missing = []
        if not baseline_side["available"]:
            missing.append("baseline")
        if not treatment_side["available"]:
            missing.append("treatment")
        return {
            "schema_version": 1, "available": False,
            "reason": "replay_unavailable:" + ",".join(missing),
            "baseline": baseline_side, "treatment": treatment_side,
            "frames": [], "event_correspondence_authorized": False,
        }
    baseline_events = baseline_side["events"]
    treatment_events = treatment_side["events"]
    timepoints, frames_truncated = _select_shared_timepoints(
        baseline_events, treatment_events,
    )
    baseline_times = [event["t_sec"] for event in baseline_events]
    treatment_times = [event["t_sec"] for event in treatment_events]
    frames = []
    for index, t_sec in enumerate(timepoints):
        frames.append({
            "frame_index": index, "t_sec": t_sec,
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "baseline_visible_count": bisect_right(baseline_times, t_sec),
            "treatment_visible_count": bisect_right(treatment_times, t_sec),
            "baseline_current_indices": [
                event["event_index"] for event in baseline_events
                if event["t_sec"] == t_sec
            ],
            "treatment_current_indices": [
                event["event_index"] for event in treatment_events
                if event["t_sec"] == t_sec
            ],
        })
    baseline_counts = _event_counts(baseline_events)
    treatment_counts = _event_counts(treatment_events)
    return {
        "schema_version": 1, "available": True,
        "reason": "shared_match_clock_alignment",
        "alignment_rule": "shared match clock only; no cross-match event matching",
        "event_correspondence_authorized": False,
        "causal_claim_authorized": False,
        "baseline": baseline_side, "treatment": treatment_side,
        "frames": frames, "frames_truncated": frames_truncated,
        "summary": {
            metric: {
                "baseline": baseline_counts[metric],
                "treatment": treatment_counts[metric],
                "delta": treatment_counts[metric] - baseline_counts[metric],
            }
            for metric in baseline_counts
        },
    }


def _checkpoint_identity(report: Mapping[str, Any]) -> str:
    runtime = _nested(report, ("layers", "world_model", "runtime")) or {}
    if isinstance(runtime, Mapping) and runtime.get("checkpoint_signature"):
        return str(runtime["checkpoint_signature"])
    evidence = report.get("evidence_snapshot") or {}
    return str(
        evidence.get("world_model_checkpoint_sha256") or "not_applicable"
    )


def build_paired_comparison(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a same-seed contrast without overstating a single pair."""
    baseline_id = str(baseline.get("match_id") or "")
    treatment_id = str(treatment.get("match_id") or "")
    if not baseline_id or not treatment_id or baseline_id == treatment_id:
        raise PairingError("paired reports require two distinct match identities")
    baseline_fixture = baseline.get("fixture") or {}
    treatment_fixture = treatment.get("fixture") or {}
    ordered_fixture = (
        str(baseline_fixture.get("home")),
        str(baseline_fixture.get("away")),
    )
    if ordered_fixture != (
        str(treatment_fixture.get("home")),
        str(treatment_fixture.get("away")),
    ):
        raise PairingError("paired reports require the same ordered fixture")
    baseline_seed = baseline_fixture.get("seed")
    treatment_seed = treatment_fixture.get("seed")
    if baseline_seed != treatment_seed or not isinstance(baseline_seed, int):
        raise PairingError("paired reports require the same deterministic seed")
    baseline_plan = baseline.get("match_plan") or {}
    treatment_plan = treatment.get("match_plan") or {}
    if treatment_plan.get("paired_baseline_match_id") != baseline_id:
        raise PairingError("treatment does not identify the supplied baseline")
    if not treatment_plan.get("reuse_last_seed"):
        raise PairingError("treatment did not declare shared-seed reuse")

    changed_sides = [
        side for side in ("home", "away")
        if baseline_plan.get(f"{side}_tactic")
        != treatment_plan.get(f"{side}_tactic")
    ]
    if not changed_sides:
        raise PairingError("paired reports contain no tactical intervention change")
    scope = (
        "single_side_tactical_intervention"
        if len(changed_sides) == 1 else "joint_tactical_intervention"
    )
    same_mode = (
        (baseline.get("studio") or {}).get("mode")
        == (treatment.get("studio") or {}).get("mode")
    )
    mode = str((treatment.get("studio") or {}).get("mode") or "unknown")
    checks = {
        "same_ordered_fixture": True,
        "same_seed": True,
        "same_fast_configuration": (
            baseline_fixture.get("fast") == treatment_fixture.get("fast")
        ),
        "same_studio_mode": same_mode,
        "physics_score_path_both": (
            baseline_plan.get("score_path") == "physics_official"
            and treatment_plan.get("score_path") == "physics_official"
        ),
        "integrity_accepted_both": (
            bool((baseline.get("integrity") or {}).get("accepted"))
            and bool((treatment.get("integrity") or {}).get("accepted"))
        ),
        "same_world_model_identity": (
            _checkpoint_identity(baseline) == _checkpoint_identity(treatment)
        ),
        "provider_determinism_controlled": mode != "cognitive",
    }
    eligible = all(checks.values())
    metrics: dict[str, dict[str, float | None]] = {}
    for metric, path in METRICS:
        before = _finite(_nested(baseline, path))
        after = _finite(_nested(treatment, path))
        metrics[metric] = {
            "baseline": before,
            "treatment": after,
            "delta": (
                after - before if before is not None and after is not None else None
            ),
        }
    paired_replay = build_synchronized_paired_replay(
        baseline, treatment, home=ordered_fixture[0], away=ordered_fixture[1],
    )
    return {
        "schema_version": 1,
        "comparison_id": f"{treatment_id}-paired-vs-{baseline_id}",
        "fixture": {
            "home": ordered_fixture[0], "away": ordered_fixture[1],
            "seed": baseline_seed,
        },
        "baseline": {"match_id": baseline_id, "plan": dict(baseline_plan)},
        "treatment": {"match_id": treatment_id, "plan": dict(treatment_plan)},
        "intervention": {
            "scope": scope,
            "changed_sides": changed_sides,
            "baseline_tactics": {
                side: baseline_plan.get(f"{side}_tactic")
                for side in ("home", "away")
            },
            "treatment_tactics": {
                side: treatment_plan.get(f"{side}_tactic")
                for side in ("home", "away")
            },
        },
        "eligibility": {
            "eligible_for_tactical_attribution": eligible,
            "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
        },
        "metrics": metrics,
        "paired_replay": paired_replay,
        "claim_boundary": (
            "paired deterministic contrast for this fixture and seed only; "
            "not a population effect, significance test, or general performance claim"
        ),
    }


def _display_tactic(value: Any) -> str:
    key = str(value or "unknown")
    return str((PLAYABLE_TACTICS.get(key) or {}).get("label") or key)


def _paired_pitch_svg(
    events: list[dict[str, Any]], *, home: str, label: str,
) -> str:
    trajectories = []
    for event in events:
        x1, y1 = 40 + 920 * event["start"][0], 40 + 560 * event["start"][1]
        x2, y2 = 40 + 920 * event["end"][0], 40 + 560 * event["end"][1]
        team_class = "pair-home" if event["team"] == home else "pair-away"
        classes = [
            "pair-event", f'pair-{event["type"]}', team_class,
            "pair-wm-linked" if event["wm_linked"] else "",
            "pair-wm-changed" if event["wm_changed"] else "",
        ]
        title = (
            f'{event["clock"]} · {event["team"]} · {event["actor"]} · '
            f'{event["kind"]} · {event["outcome"]}'
        )
        trajectories.append(
            f'<g class="{" ".join(filter(None, classes))}">'
            f'<title>{html.escape(title)}</title>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" />'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" '
            f'r="{5 if event["type"] == "shot" else 2.5}" /></g>'
        )
    return f"""<svg viewBox="0 0 1000 640" role="img" aria-label="{html.escape(label, quote=True)}">
<rect class="pair-pitch" x="20" y="20" width="960" height="600" rx="8"/><line class="pair-marking" x1="500" y1="20" x2="500" y2="620"/><circle class="pair-marking" cx="500" cy="320" r="78"/><circle class="pair-spot" cx="500" cy="320" r="4"/><rect class="pair-marking" x="20" y="155" width="150" height="330"/><rect class="pair-marking" x="830" y="155" width="150" height="330"/><rect class="pair-marking" x="20" y="235" width="55" height="170"/><rect class="pair-marking" x="925" y="235" width="55" height="170"/>{''.join(trajectories)}</svg>"""


def _current_action_text(
    events: list[dict[str, Any]], indices: list[int],
) -> str:
    selected = [
        events[index] for index in indices[:3]
        if isinstance(index, int) and 0 <= index < len(events)
    ]
    if not selected:
        return '<span class="muted">该时点无新保留动作</span>'
    items = []
    for event in selected:
        badge = (
            ' <span class="pair-badge changed">WM changed</span>'
            if event["wm_changed"] else
            ' <span class="pair-badge">WM-linked</span>'
            if event["wm_linked"] else ""
        )
        items.append(
            f'<span><b>{html.escape(event["team"])}</b> · '
            f'{html.escape(event["actor"])} → {html.escape(event["target"])} · '
            f'{html.escape(event["kind"])} · '
            f'{html.escape(event["outcome"])}{badge}</span>'
        )
    if len(indices) > len(selected):
        items.append(
            f'<span class="muted">另有 {len(indices) - len(selected)} 个同刻动作</span>'
        )
    return "".join(items)


def _paired_replay_panel(comparison: Mapping[str, Any]) -> str:
    fixture = comparison.get("fixture") or {}
    home, away = _safe_text(fixture.get("home")), _safe_text(fixture.get("away"))
    stored = comparison.get("paired_replay") or {}
    if not isinstance(stored, Mapping):
        stored = {}
    baseline_stored = stored.get("baseline") or {}
    treatment_stored = stored.get("treatment") or {}
    replay = build_synchronized_paired_replay(
        {"replay": baseline_stored if isinstance(baseline_stored, Mapping) else {}},
        {"replay": treatment_stored if isinstance(treatment_stored, Mapping) else {}},
        home=home, away=away,
    )
    if not replay.get("available"):
        return f"""<section class="card paired-replay" data-testid="paired-replay-panel">
<h2>同步动作复盘</h2><p class="muted">双场播放器不可用：{html.escape(str(replay.get('reason') or 'unknown'))}。指标比较与资格检查仍然有效。</p></section>"""
    baseline_events = replay["baseline"]["events"]
    treatment_events = replay["treatment"]["events"]
    frames = replay["frames"][:MAX_PAIRED_REPLAY_FRAMES]
    summary = replay.get("summary") or {}
    inputs = [
        '<input type="radio" name="pair-step" id="pair-step-overview" checked>'
    ]
    markers = [
        '<label for="pair-step-overview" title="显示两场完整轨迹">概览</label>'
    ]
    navigation = [
        '<div class="pair-frame pair-frame-overview"><div><span class="eyebrow">同步时间轴</span><strong>两场完整动作地图</strong><span class="muted">按共享比赛时钟对齐，不代表跨场动作对应。</span></div><label class="pair-button" for="pair-step-0">开始复盘</label></div>'
    ]
    rules = [
        '#pair-step-overview:checked~.pair-navigation .pair-frame-overview{display:flex}',
        '#pair-step-overview:checked~.pair-timeline label[for="pair-step-overview"]{border-color:var(--ok);color:var(--ok);background:#10281f}',
    ]
    for frame_index, frame in enumerate(frames):
        step_id = f"pair-step-{frame_index}"
        inputs.append(f'<input type="radio" name="pair-step" id="{step_id}">')
        markers.append(
            f'<label for="{step_id}" title="共享比赛时钟 {html.escape(frame["clock"], quote=True)}">'
            f'{frame_index + 1}<span>{html.escape(frame["clock"])}</span></label>'
        )
        previous_id = (
            "pair-step-overview" if frame_index == 0
            else f"pair-step-{frame_index - 1}"
        )
        next_id = (
            "pair-step-overview" if frame_index + 1 == len(frames)
            else f"pair-step-{frame_index + 1}"
        )
        next_label = "返回概览" if frame_index + 1 == len(frames) else "下一时点"
        navigation.append(
            f'<div class="pair-frame pair-frame-{frame_index}">'
            f'<label class="pair-button" for="{previous_id}">上一时点</label>'
            f'<div class="pair-frame-center"><span class="eyebrow">时点 '
            f'{frame_index + 1} / {len(frames)} · {html.escape(frame["clock"])}</span>'
            f'<div class="pair-current"><div><b>基线场</b>{_current_action_text(baseline_events, frame["baseline_current_indices"])}</div>'
            f'<div><b>处理场</b>{_current_action_text(treatment_events, frame["treatment_current_indices"])}</div></div></div>'
            f'<label class="pair-button" for="{next_id}">{next_label}</label></div>'
        )
        rules.extend([
            f'#{step_id}:checked~.pair-navigation .pair-frame-{frame_index}{{display:flex}}',
            f'#{step_id}:checked~.pair-timeline label[for="{step_id}"]{{border-color:var(--ok);color:var(--ok);background:#10281f}}',
            f'#{step_id}:focus-visible~.pair-timeline label[for="{step_id}"]{{outline:3px solid var(--warn);outline-offset:2px}}',
        ])
        for side_name, events, visible_key, current_key in (
            ("baseline", baseline_events, "baseline_visible_count", "baseline_current_indices"),
            ("treatment", treatment_events, "treatment_visible_count", "treatment_current_indices"),
        ):
            visible = max(0, min(len(events), int(frame[visible_key])))
            rules.extend([
                f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event line{{opacity:.12}}',
                f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event circle{{opacity:.2}}',
            ])
            if visible < len(events):
                rules.append(
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} '
                    f'.pair-event:nth-of-type(n+{visible + 1}){{display:none}}'
                )
            for current_index in frame[current_key]:
                nth = int(current_index) + 1
                rules.extend([
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}){{display:inline;filter:drop-shadow(0 0 7px #fff)}}',
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}) line{{opacity:1;stroke-width:8}}',
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}) circle{{opacity:1}}',
                ])
    summary_cards = "".join(
        f'<div><span class="eyebrow">{html.escape(metric)}</span>'
        f'<strong>{html.escape(str((values or {}).get("baseline")))} → '
        f'{html.escape(str((values or {}).get("treatment")))}</strong>'
        f'<span>Δ {html.escape(str((values or {}).get("delta")))}</span></div>'
        for metric, values in summary.items()
    )
    truncation_note = (
        "共享时点超过上限，已优先保留射门、关键结果与 WM-changed 时点。"
        if replay.get("frames_truncated") else "全部共享动作时点均已保留。"
    )
    return f"""<section class="card paired-replay" data-testid="paired-replay-panel">
<div class="pair-panel-head"><div><span class="eyebrow">Shared-clock descriptive replay</span><h2>同步动作复盘</h2></div><span class="pair-badge">{len(frames)} 时点</span></div>
<p class="muted">两场只按比赛时钟同步，不进行动作一一配对，也不新增因果资格。{truncation_note}</p>
<div class="pair-summary">{summary_cards}</div>
<fieldset class="pair-controls"><legend>双场轨迹筛选</legend>
<input type="radio" name="pair-filter" id="pair-filter-all" checked><label for="pair-filter-all">全部</label>
<input type="radio" name="pair-filter" id="pair-filter-shots"><label for="pair-filter-shots">仅射门</label>
<input type="radio" name="pair-filter" id="pair-filter-wm"><label for="pair-filter-wm">WM changed</label>
{''.join(inputs)}
<div class="pair-timeline" role="group" aria-label="跳转到共享比赛时点">{''.join(markers)}</div>
<div class="pair-navigation" aria-live="polite">{''.join(navigation)}</div>
<div class="pair-pitches"><article class="pair-side pair-side-baseline"><h3>基线场</h3>{_paired_pitch_svg(baseline_events, home=home, label='基线场动作轨迹')}</article><article class="pair-side pair-side-treatment"><h3>处理场</h3>{_paired_pitch_svg(treatment_events, home=home, label='处理场动作轨迹')}</article></div>
</fieldset><style>{''.join(rules)}</style></section>"""


def render_paired_comparison_html(comparison: Mapping[str, Any]) -> str:
    fixture = comparison.get("fixture") or {}
    eligibility = comparison.get("eligibility") or {}
    intervention = comparison.get("intervention") or {}
    eligible = bool(eligibility.get("eligible_for_tactical_attribution"))
    state = "配对资格通过" if eligible else "仅描述性比较"
    before = intervention.get("baseline_tactics") or {}
    after = intervention.get("treatment_tactics") or {}
    metric_rows = []
    for name, values in (comparison.get("metrics") or {}).items():
        values = values or {}
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(values.get('baseline')))}</td>"
            f"<td>{html.escape(str(values.get('treatment')))}</td>"
            f"<td>{html.escape(str(values.get('delta')))}</td></tr>"
        )
    check_rows = "".join(
        f"<li class=\"{'ok' if passed else 'warn'}\">"
        f"{html.escape(str(name))}: {'通过' if passed else '未通过'}</li>"
        for name, passed in (eligibility.get("checks") or {}).items()
    )
    reports = comparison.get("reports") or {}
    report_links = []
    for label, key in (("打开基线场", "baseline"), ("打开处理场", "treatment")):
        source = reports.get(key)
        if source:
            filename = Path(str(source)).with_suffix(".html").name
            report_links.append(
                f'<a href="{html.escape(filename, quote=True)}">{label}</a>'
            )
    navigation = " · ".join(report_links)
    replay_panel = _paired_replay_panel(comparison)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(fixture.get('home')))} vs {html.escape(str(fixture.get('away')))} · 战术配对比较</title>
<style>:root{{--bg:#07111e;--panel:#111d2e;--line:#2a3a51;--ink:#edf4ff;--muted:#a8b6c9;--ok:#65e6b4;--warn:#ffc36a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px}}h1{{font-size:clamp(28px,6vw,54px);margin:.2em 0}}.eyebrow,.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line)}}.scroll{{overflow:auto}}code{{color:var(--ok)}}.pair-panel-head{{display:flex;justify-content:space-between;gap:12px;align-items:start}}.pair-panel-head h2{{margin:.2em 0}}.pair-badge{{display:inline-block;border:1px solid #42516a;border-radius:999px;padding:2px 8px;color:#cbd7e8;font-size:12px}}.pair-badge.changed{{border-color:var(--ok);color:var(--ok)}}.pair-summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:14px 0}}.pair-summary>div{{display:grid;background:#0a1424;border:1px solid var(--line);border-radius:9px;padding:10px}}.pair-summary strong{{font-size:18px}}.pair-controls{{display:flex;flex-wrap:wrap;gap:7px;border:0;padding:0;margin:0}}.pair-controls legend{{width:100%;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.pair-controls>input{{position:absolute;opacity:0;pointer-events:none}}.pair-controls>label{{border:1px solid #42516a;border-radius:999px;padding:6px 10px;cursor:pointer}}.pair-controls>input:focus-visible+label{{outline:3px solid var(--warn);outline-offset:2px}}.pair-controls>input:checked+label{{border-color:var(--ok);color:var(--ok);background:#10281f}}.pair-timeline{{display:flex;gap:5px;overflow:auto;width:100%;padding:12px 2px 5px}}.pair-timeline label{{flex:0 0 auto;min-width:42px;border:1px solid #42516a;border-radius:8px;padding:4px 7px;text-align:center;cursor:pointer;font-size:12px}}.pair-timeline label span{{display:block;color:var(--muted);font-size:10px}}.pair-navigation{{width:100%;margin-top:8px}}.pair-frame{{display:none;align-items:center;justify-content:space-between;gap:12px;background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:10px}}.pair-frame>div{{display:grid;gap:3px;text-align:center;min-width:0}}.pair-frame-center{{flex:1}}.pair-current{{display:grid;grid-template-columns:1fr 1fr;gap:8px;text-align:left}}.pair-current>div{{display:grid;gap:2px;background:#0d192a;border-radius:8px;padding:8px}}.pair-current>div>span{{display:block}}.pair-button{{border:1px solid #42516a;border-radius:8px;padding:7px 10px;cursor:pointer;color:var(--ok);white-space:nowrap}}.pair-pitches{{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;margin-top:12px}}.pair-side{{background:#071c19;border:1px solid #285448;border-radius:12px;padding:10px}}.pair-side h3{{margin:0 0 6px}}.pair-side svg{{display:block;width:100%;height:auto}}.pair-pitch{{fill:#0c392d;stroke:#b8d8cd;stroke-width:3}}.pair-marking{{fill:none;stroke:#b8d8cd;stroke-width:3}}.pair-spot{{fill:#b8d8cd}}.pair-event line{{stroke-width:4;stroke-linecap:round;opacity:.42}}.pair-event circle{{opacity:.75}}.pair-home line,.pair-home circle{{stroke:#65e6b4;fill:#65e6b4}}.pair-away line,.pair-away circle{{stroke:#ff9f7a;fill:#ff9f7a}}.pair-shot line{{stroke-width:7;opacity:.85}}.pair-wm-changed line{{filter:drop-shadow(0 0 5px #fff);opacity:1}}#pair-filter-shots:checked~.pair-pitches .pair-event:not(.pair-shot),#pair-filter-wm:checked~.pair-pitches .pair-event:not(.pair-wm-changed){{display:none}}@media(max-width:760px){{th,td{{padding:7px;font-size:12px}}.pair-pitches,.pair-current{{grid-template-columns:1fr}}.pair-frame{{flex-wrap:wrap}}.pair-frame-center{{order:-1;width:100%;flex-basis:100%}}.pair-button{{flex:1;text-align:center}}.pair-panel-head{{display:block}}}}</style></head><body><main>
<div class="eyebrow">GFS Tactical Lab · shared-seed contrast</div><h1>{html.escape(str(fixture.get('home')))} vs {html.escape(str(fixture.get('away')))}</h1><p>seed <code>{html.escape(str(fixture.get('seed')))}</code> · <strong class="{'ok' if eligible else 'warn'}">{state}</strong></p><p>{navigation}</p>
<section class="grid"><article class="card"><h2>基线战术</h2><p>主队：{html.escape(_display_tactic(before.get('home')))}<br>客队：{html.escape(_display_tactic(before.get('away')))}</p></article><article class="card"><h2>处理战术</h2><p>主队：{html.escape(_display_tactic(after.get('home')))}<br>客队：{html.escape(_display_tactic(after.get('away')))}</p></article><article class="card"><h2>干预范围</h2><p>{html.escape(str(intervention.get('scope')))}</p></article></section>
<section class="card"><h2>配对资格检查</h2><ul>{check_rows}</ul></section>
{replay_panel}
<section class="card"><h2>处理场减去基线场</h2><div class="scroll"><table><thead><tr><th>指标</th><th>基线</th><th>处理</th><th>差值</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div></section>
<section class="card"><h2>推断边界</h2><p class="muted">{html.escape(str(comparison.get('claim_boundary') or ''))}</p></section>
</main></body></html>"""


def write_paired_comparison_html(
    path: str | Path, comparison: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_paired_comparison_html(comparison), encoding="utf-8")
    return target

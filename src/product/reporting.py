"""Dependency-free HTML views for GFS Studio product reports."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.product.match_plan import NATIVE_TACTIC, PLAYABLE_TACTICS


def _value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "&mdash;"
    return html.escape(str(value))


def _percent(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite percentage")
        return f"{100.0 * number:.{digits}f}%"
    except (TypeError, ValueError):
        return "&mdash;"


def _signed(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return f"{number:+.{digits}f}"
    except (TypeError, ValueError):
        return "&mdash;"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dominant_model_action(record: Mapping[str, Any]) -> str:
    adjustments = record.get("model_adjustments") or {}
    if not isinstance(adjustments, Mapping):
        return str(record.get("recommended_action") or "unresolved")
    finite = []
    for action, value in adjustments.items():
        try:
            finite.append((abs(float(value)), str(action)))
        except (TypeError, ValueError):
            continue
    if finite and max(finite)[0] > 0.0:
        return max(finite)[1]
    return str(record.get("recommended_action") or "unresolved")


def _quality_gate_text(record: Mapping[str, Any], action: str) -> tuple[str, str]:
    gates = record.get("quality_gates") or {}
    gate = gates.get(action) or {} if isinstance(gates, Mapping) else {}
    if not isinstance(gate, Mapping):
        gate = {}
    if gate.get("open"):
        confidence = _percent(gate.get("confidence"))
        certainty = _percent(gate.get("certainty"))
        return "open", f"confidence {confidence}; certainty {certainty}"
    reason = str(gate.get("reason") or "no action-specific gate evidence")
    return "closed", reason


def _quality_gate_details(record: Mapping[str, Any]) -> str:
    gates = record.get("quality_gates") or {}
    if not isinstance(gates, Mapping):
        gates = {}
    items = []
    for action in sorted(gates, key=str):
        gate = gates.get(action) or {}
        if not isinstance(gate, Mapping):
            gate = {}
        state = "open" if gate.get("open") else "closed"
        reason = str(gate.get("reason") or "validated evidence admitted")
        items.append(
            f"<li><b>{html.escape(str(action))}</b>: {state}; "
            f"{html.escape(reason)}; confidence {_percent(gate.get('confidence'))}; "
            f"certainty {_percent(gate.get('certainty'))}</li>"
        )
    if not items:
        return "<p class=\"subtle\">No action-specific gate record.</p>"
    return f"<ul class=\"gate-list\">{''.join(items)}</ul>"


def _action_link_label(status: str) -> str:
    return {
        "direct_runtime_identity_match": "direct runtime identity match",
        "no_ball_trajectory_by_design": "no trajectory for hold",
        "legacy_record_without_runtime_identity": "legacy record: no identity key",
        "ambiguous_runtime_identity_collision": "ambiguous identity collision",
        "ambiguous_action_record_identity_collision": "duplicate decision identity",
        "action_event_may_be_truncated": "action may be outside retained log",
        "action_event_missing": "matching action event missing",
        "runtime_identity_action_mismatch": "identity/action mismatch",
        "runtime_identity_team_mismatch": "identity/team mismatch",
        "runtime_identity_timestamp_mismatch": "identity/timestamp mismatch",
        "unresolved_or_unsupported_action": "unresolved or unsupported action",
        "no_link_record": "no linkage record",
    }.get(status, status.replace("_", " "))


def _action_influence_panel(
    world_model: Mapping[str, Any], replay: Mapping[str, Any] | None = None,
) -> str:
    configured = bool(world_model.get("configured"))
    enabled = bool(world_model.get("enabled"))
    adoption = world_model.get("action_adoption") or {}
    mechanism = world_model.get("action_adoption_mechanism") or {}
    if not isinstance(adoption, Mapping):
        adoption = {}
    if not isinstance(mechanism, Mapping):
        mechanism = {}
    linkage = (replay or {}).get("world_model_action_links") or {}
    if not isinstance(linkage, Mapping):
        linkage = {}
    links_by_identity = {
        str(link.get("opportunity_id")): link
        for link in list(linkage.get("links") or [])[:96]
        if isinstance(link, Mapping) and link.get("opportunity_id")
    }
    if not configured:
        return """<section class="card wm-panel" data-testid="wm-action-panel">
<h2>World-model action influence</h2>
<p class="notice">Inactive in stable mode. Stable matches use the released simulator without research-policy influence.</p>
</section>"""
    if not enabled:
        return """<section class="card wm-panel" data-testid="wm-action-panel">
<h2>World-model action influence</h2>
<p class="notice warn">Configured but not observed at runtime. This match cannot provide world-model action evidence.</p>
</section>"""

    opportunities = max(0, int(_finite_float(adoption.get("opportunities"))))
    influenced = max(0, int(_finite_float(
        adoption.get("influenced_opportunities")
    )))
    eligible = max(0, int(_finite_float(
        adoption.get("attribution_eligible_opportunities")
    )))
    changed = max(0, int(_finite_float(
        adoption.get("counterfactual_action_changes")
    )))
    expected = max(0.0, _finite_float(
        adoption.get("expected_counterfactual_action_changes")
    ))
    mean_shift = max(0.0, _finite_float(
        adoption.get("mean_recommended_probability_shift")
    ))
    mechanism_state = str(mechanism.get("result_status") or mechanism.get(
        "execution_state", "not registered",
    ))
    budget = max(0, int(_finite_float(mechanism.get("fixed_run_budget"))))
    runs = max(0, int(_finite_float(mechanism.get("runs_executed"))))
    protocol_line = f"Mechanism protocol: <b>{html.escape(mechanism_state)}</b>"
    if budget:
        protocol_line += f" &middot; {runs}/{budget} fixed runs"
    if not adoption.get("available", False):
        return f"""<section class="card wm-panel" data-testid="wm-action-panel">
<h2>World-model action influence</h2>
<p class="notice">The model loaded, but no direct action-policy opportunity was registered.</p>
<p class="evidence">{protocol_line}</p></section>"""

    records = [
        row for row in list(adoption.get("records") or [])
        if isinstance(row, Mapping)
    ]
    records.sort(key=lambda row: (
        bool(row.get("policy_changed_action")),
        _finite_float(row.get("total_variation_distance")),
    ), reverse=True)
    rows = []
    changed_count = 0
    probability_count = 0
    gate_closed_count = 0
    for record in records:
        base_action = html.escape(str(
            record.get("counterfactual_baseline_action") or "unresolved"
        ))
        actual_action = html.escape(str(record.get("actual_action") or "unresolved"))
        team = html.escape(str(record.get("team_id") or "unknown"))
        minute = _finite_float(record.get("t_sec")) / 60.0
        signal_action = _dominant_model_action(record)
        base_probability = record.get("base_probability") or {}
        adjusted_probability = record.get("adjusted_probability") or {}
        adjustments = record.get("model_adjustments") or {}
        base_signal = (
            base_probability.get(signal_action)
            if isinstance(base_probability, Mapping) else None
        )
        adjusted_signal = (
            adjusted_probability.get(signal_action)
            if isinstance(adjusted_probability, Mapping) else None
        )
        model_adjustment = (
            adjustments.get(signal_action)
            if isinstance(adjustments, Mapping) else None
        )
        gate_state, gate_detail = _quality_gate_text(record, signal_action)
        gate_details = _quality_gate_details(record)
        sampling_uniform = record.get("sampling_uniform")
        draw = _value(sampling_uniform, 3)
        changed_mark = "changed" if record.get("policy_changed_action") else "same"
        attribution = "eligible" if record.get("attribution_eligible") else "not isolated"
        probability_changed = _finite_float(
            record.get("total_variation_distance")
        ) > 1e-12
        changed_count += int(bool(record.get("policy_changed_action")))
        probability_count += int(probability_changed)
        gate_closed_count += int(gate_state == "closed")
        row_classes = [
            "decision-changed" if record.get("policy_changed_action") else "",
            "decision-probability" if probability_changed else "",
            "decision-gate-closed" if gate_state == "closed" else "",
        ]
        action_link = links_by_identity.get(str(record.get("opportunity_id") or ""))
        link_status = (
            str(action_link.get("status") or "unavailable")
            if isinstance(action_link, Mapping) else "no_link_record"
        )
        if link_status == "direct_runtime_identity_match":
            observed_evidence = (
                f'<a href="#action-replay-panel">direct trajectory '
                f'#{int(_finite_float(action_link.get("event_index"))) + 1}</a>'
            )
        elif link_status == "no_ball_trajectory_by_design":
            observed_evidence = _action_link_label(link_status)
        else:
            observed_evidence = html.escape(_action_link_label(link_status))
        rows.append(
            f"<tr class=\"{' '.join(filter(None, row_classes))}\">"
            f"<td>{minute:.1f}'</td><td>{team}</td>"
            f"<td><b>{html.escape(signal_action)}</b> {_signed(model_adjustment)}<br>"
            f"<span class=\"subtle\">{html.escape(gate_state)}: "
            f"{html.escape(gate_detail)}</span>"
            f"<details><summary>all quality gates</summary>{gate_details}</details></td>"
            f"<td>{_percent(base_signal)} &rarr; {_percent(adjusted_signal)}"
            f"<br><span class=\"subtle\">u={draw}</span></td>"
            f"<td>{base_action} &rarr; {actual_action}</td>"
            f"<td><span class=\"tag {changed_mark}\">{changed_mark}</span> "
            f"<span class=\"tag\">{attribution}</span></td>"
            f"<td><span class=\"subtle\">{observed_evidence}</span></td></tr>"
        )
    decision_table = (
        "<div class=\"wm-explorer\"><fieldset class=\"wm-filters\"><legend>Decision filter</legend>"
        f"<input type=\"radio\" name=\"wm-filter\" id=\"wm-filter-all\" checked><label for=\"wm-filter-all\">All ({len(rows)})</label>"
        f"<input type=\"radio\" name=\"wm-filter\" id=\"wm-filter-changed\"><label for=\"wm-filter-changed\">Changed action ({changed_count})</label>"
        f"<input type=\"radio\" name=\"wm-filter\" id=\"wm-filter-probability\"><label for=\"wm-filter-probability\">Probability shifted ({probability_count})</label>"
        f"<input type=\"radio\" name=\"wm-filter\" id=\"wm-filter-closed\"><label for=\"wm-filter-closed\">Dominant gate closed ({gate_closed_count})</label>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Time</th><th>Team</th>"
        "<th>Gate &amp; model signal</th><th>Probability &amp; shared draw</th>"
        "<th>Baseline &rarr; actual</th><th>Audit</th><th>Observed action</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table></div></fieldset></div>"
        if rows else "<p class=\"notice\">No retained decision samples.</p>"
    )
    return f"""<section class="card wm-panel" data-testid="wm-action-panel">
<div class="panel-head"><div><div class="label">Research-mode explanation</div><h2>World-model action influence</h2></div><span class="tag">quality gated</span></div>
<div class="wm-grid">
  <div><span class="label">Opportunities</span><strong>{opportunities}</strong></div>
  <div><span class="label">Non-zero influence</span><strong>{influenced}</strong></div>
  <div><span class="label">Attribution eligible</span><strong>{eligible}</strong></div>
  <div><span class="label">Counterfactual changes</span><strong>{changed}</strong></div>
  <div><span class="label">Expected changes</span><strong>{expected:.2f}</strong></div>
  <div><span class="label">Mean probability shift</span><strong>{_percent(mean_shift, 2)}</strong></div>
</div>
<p class="evidence">{protocol_line}. Expected changes are probability mass, not additional observed outcomes. This panel does not authorize product or academic promotion.</p>
<p class="evidence">Observed trajectories are linked only by the runtime opportunity identity plus matching action, team and timestamp. Time proximity alone is never treated as a direct match.</p>
<ol class="causal-chain"><li>Quality gate decides whether validated model evidence may enter the policy.</li><li>The bounded model signal adjusts action utility and therefore sampling probability.</li><li>The same stored random draw is applied to baseline and adjusted distributions.</li><li>A different sampled action is a local counterfactual change; only isolated records are attribution eligible.</li></ol>
{decision_table}</section>"""


def _match_plan_panel(report: Mapping[str, Any]) -> str:
    fixture = report.get("fixture") or {}
    plan = report.get("match_plan") or {
        "experience": "observational",
        "home_tactic": NATIVE_TACTIC,
        "away_tactic": NATIVE_TACTIC,
        "score_path": "macro_replay",
        "reuse_last_seed": False,
    }
    experience = str(plan.get("experience") or "observational")
    home_tactic = str(plan.get("home_tactic") or NATIVE_TACTIC)
    away_tactic = str(plan.get("away_tactic") or NATIVE_TACTIC)
    world_model_policy = str(
        plan.get("world_model_policy") or "mode_default"
    )
    branch_at_sec = _optional_finite_float(
        plan.get("world_model_branch_at_sec")
    )
    simulation_clock = report.get("simulation_clock") or {}

    def tactic_label(value: str) -> str:
        metadata = PLAYABLE_TACTICS.get(value) or {}
        return html.escape(str(metadata.get("label") or value))

    score_path = str(plan.get("score_path") or "not reported")
    score_label = {
        "physics_official": "物理射门与扑救产生正式比分",
        "macro_replay": "稳定宏观比分回放",
    }.get(score_path, score_path)
    seed = fixture.get("seed")
    paired = bool(plan.get("reuse_last_seed"))
    paired_baseline = plan.get("paired_baseline_match_id")
    comparison_dashboard = (
        (report.get("artifacts") or {}).get("paired_comparison_dashboard")
    )
    comparison_link = ""
    if comparison_dashboard:
        comparison_name = Path(str(comparison_dashboard)).name
        comparison_link = (
            f'<p><a href="{html.escape(comparison_name, quote=True)}">'
            "打开同种子战术配对比较</a></p>"
        )
    if experience == "world_model_lab":
        inference = (
            "本场属于同种子世界模型策略分叉；只有与绑定的另一世界比较，"
            "且全部资格检查通过，才能归因于模拟器内动作策略开关。"
        )
        experience_label = "世界模型因果分叉"
        policy_label = {
            "predict_only": "基线：仅预测，不进入动作策略",
            "action_policy": "干预：质量门控动作策略",
        }.get(world_model_policy, world_model_policy)
        policy_card = (
            '<div><span class="label">世界模型策略</span><strong>'
            f'{html.escape(policy_label)}</strong></div>'
        )
        if branch_at_sec is not None:
            branch_minute = branch_at_sec / 60.0
            clock_contract = str(
                simulation_clock.get("contract") or "not reported"
            )
            policy_card += (
                '<div><span class="label">Branch minute</span><strong>'
                f'{branch_minute:g}</strong></div>'
                '<div><span class="label">Clock contract</span><strong>'
                f'{html.escape(clock_contract)}</strong></div>'
            )
    elif experience == "tactical_lab":
        inference = (
            f"已复用同对阵随机条件（基线 {paired_baseline or '未记录'}）；"
            "需与对应基线报告配对比较，才可讨论战术归因。"
            if paired else
            "本场是单次战术运行，只支持描述性复盘，不支持战术因果归因。"
        )
        experience_label = "战术实验室"
        policy_card = ""
    else:
        inference = "原生观赛不施加玩家战术干预。"
        experience_label = "原生观赛"
        policy_card = ""
    return f"""<section class="card plan-panel" data-testid="match-plan-panel">
<div class="panel-head"><div><div class="label">Reproducible match plan</div><h2>比赛方案与推断边界</h2></div><span class="tag">seed {_value(seed, 0)}</span></div>
<div class="wm-grid">
  <div><span class="label">体验</span><strong>{html.escape(experience_label)}</strong></div>
  <div><span class="label">主队战术</span><strong>{tactic_label(home_tactic)}</strong></div>
  <div><span class="label">客队战术</span><strong>{tactic_label(away_tactic)}</strong></div>
  <div><span class="label">正式比分路径</span><strong>{html.escape(score_label)}</strong></div>
  {policy_card}
</div>
<p class="evidence">{html.escape(inference)}</p>{comparison_link}</section>"""


def _replay_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


def _replay_sequence_controls(
    events: list[dict[str, Any]],
) -> tuple[str, str, str, str]:
    """Build a bounded, script-free sequence navigator for replay actions."""
    inputs = [
        '<input type="radio" name="replay-step" '
        'id="replay-step-overview" checked>'
    ]
    markers = [
        '<label for="replay-step-overview" title="Show complete trajectory">'
        'Overview</label>'
    ]
    frames = [
        '<div class="sequence-frame sequence-frame-overview">'
        '<div><span class="label">Sequence navigator</span>'
        '<strong>Complete action map</strong><span>Choose a timestamp or start '
        'from the first retained action.</span></div>'
        '<label class="sequence-button" for="replay-step-0">Start sequence</label>'
        '</div>'
    ]
    rules = [
        '#replay-step-overview:checked~.replay-navigation '
        '.sequence-frame-overview{display:flex}',
        '#replay-step-overview:checked~.replay-timeline '
        'label[for="replay-step-overview"]{border-color:var(--accent);'
        'color:var(--accent);background:#10281f}',
    ]
    total = len(events)
    for index, event in enumerate(events):
        step_id = f"replay-step-{index}"
        inputs.append(
            f'<input type="radio" name="replay-step" id="{step_id}">'
        )
        title = (
            f'{event["clock"]} · {event["team"]} · {event["actor"]} · '
            f'{event["kind"]} · {event["outcome"]}'
        )
        markers.append(
            f'<label for="{step_id}" title="{html.escape(title, quote=True)}">'
            f'{index + 1}<span>{html.escape(event["clock"])}</span></label>'
        )
        previous_id = "replay-step-overview" if index == 0 else (
            f"replay-step-{index - 1}"
        )
        next_id = (
            "replay-step-overview" if index + 1 == total
            else f"replay-step-{index + 1}"
        )
        next_label = "Return to overview" if index + 1 == total else "Next action"
        wm_badge = (
            '<span class="tag changed">WM changed + observed</span>'
            if event["world_model_link"].get("policy_changed_action") else
            '<span class="tag">WM-linked</span>'
            if event["world_model_link"] else ""
        )
        frames.append(
            f'<div class="sequence-frame sequence-frame-{index}">'
            f'<label class="sequence-button" for="{previous_id}">Previous</label>'
            f'<div><span class="label">Action {index + 1} / {total} · '
            f'{html.escape(event["clock"])}</span>'
            f'<strong>{html.escape(event["team"])} · '
            f'{html.escape(event["kind"])}</strong>'
            f'<span>{html.escape(event["actor"])} → '
            f'{html.escape(event["target"])} · '
            f'{html.escape(event["outcome"])}</span>{wm_badge}</div>'
            f'<label class="sequence-button" for="{next_id}">{next_label}</label>'
            '</div>'
        )
        nth = index + 1
        rules.extend([
            f'#{step_id}:checked~.replay-navigation '
            f'.sequence-frame-{index}{{display:flex}}',
            f'#{step_id}:checked~.replay-timeline label[for="{step_id}"]'
            '{border-color:var(--accent);color:var(--accent);background:#10281f}',
            f'#{step_id}:focus-visible~.replay-timeline label[for="{step_id}"]'
            '{outline:3px solid var(--warn);outline-offset:2px}',
            f'#{step_id}:checked~.replay-stage .replay-event line'
            '{opacity:.13}',
            f'#{step_id}:checked~.replay-stage .replay-event circle'
            '{opacity:.22}',
            f'#{step_id}:checked~.replay-stage '
            f'.replay-event:nth-of-type(n+{nth + 1}){{display:none}}',
            f'#{step_id}:checked~.replay-stage '
            f'.replay-event:nth-of-type({nth}){{display:inline;filter:drop-shadow(0 0 7px #fff)}}',
            f'#{step_id}:checked~.replay-stage '
            f'.replay-event:nth-of-type({nth}) line{{opacity:1;stroke-width:8}}',
            f'#{step_id}:checked~.replay-stage '
            f'.replay-event:nth-of-type({nth}) circle{{opacity:1}}',
        ])
    timeline = (
        '<div class="replay-timeline" role="group" '
        'aria-label="Jump to retained action">' + "".join(markers) + '</div>'
    )
    navigation = (
        '<div class="replay-navigation" aria-live="polite">'
        + "".join(frames) + '</div>'
    )
    return "".join(inputs), timeline, navigation, "<style>" + "".join(rules) + "</style>"


def _action_replay_panel(report: Mapping[str, Any]) -> str:
    replay = report.get("replay") or {}
    if not isinstance(replay, Mapping) or not replay.get("available"):
        reason = (
            str(replay.get("reason") or "ball_log_unavailable")
            if isinstance(replay, Mapping) else "ball_log_unavailable"
        )
        return f"""<section class="card replay-panel" data-testid="action-replay-panel">
<h2>Ball-action trajectory replay</h2>
<p class="notice">Unavailable for this match ({html.escape(reason)}). This does not affect score integrity.</p>
</section>"""

    fixture = report.get("fixture") or {}
    home, away = str(fixture.get("home") or "Home"), str(
        fixture.get("away") or "Away"
    )
    normalized = []
    for raw in list(replay.get("events") or [])[:240]:
        if not isinstance(raw, Mapping):
            continue
        start, end = _replay_point(raw.get("start")), _replay_point(raw.get("end"))
        event_type = str(raw.get("type") or "")
        if start is None or end is None or event_type not in {
            "pass", "shot", "cross",
        }:
            continue
        t_sec = min(8_000.0, max(0.0, _finite_float(raw.get("t_sec"))))
        normalized.append({
            "type": event_type,
            "team": str(raw.get("team") or "")[:100],
            "actor": str(raw.get("actor") or "")[:100],
            "target": str(raw.get("target") or "")[:100],
            "kind": str(raw.get("kind") or event_type)[:40],
            "outcome": str(raw.get("outcome") or "UNKNOWN")[:40].upper(),
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "t_sec": t_sec, "start": start, "end": end,
            "world_model_link": (
                dict(raw.get("world_model_link"))
                if isinstance(raw.get("world_model_link"), Mapping) else {}
            ),
        })
    if not normalized:
        return """<section class="card replay-panel" data-testid="action-replay-panel">
<h2>Ball-action trajectory replay</h2><p class="notice">No renderable ball actions were retained.</p></section>"""

    for sequence_index, event in enumerate(normalized):
        event["sequence_index"] = sequence_index

    lines = []
    for event in normalized:
        x1, y1 = 40 + 920 * event["start"][0], 40 + 560 * event["start"][1]
        x2, y2 = 40 + 920 * event["end"][0], 40 + 560 * event["end"][1]
        side = "home" if event["team"] == home else "away" if event["team"] == away else "neutral"
        important = event["outcome"] in {"GOAL", "SAVED", "INTERCEPTED"}
        wm_linked = bool(event["world_model_link"])
        wm_changed = bool(event["world_model_link"].get("policy_changed_action"))
        title = (
            f'{event["clock"]} {event["team"]}: {event["actor"]} '
            f'{event["kind"]} to {event["target"]} — {event["outcome"]}'
        )
        lines.append(
            f'<g class="replay-event replay-{event["type"]} replay-{side} '
            f'{"replay-important" if important else ""} '
            f'{"replay-wm-linked" if wm_linked else ""} '
            f'{"replay-wm-changed" if wm_changed else ""}"><title>{html.escape(title)}</title>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{5 if important else 2.4}" /></g>'
        )

    priority = [
        event for event in normalized
        if event["type"] == "shot" or event["outcome"] in {"GOAL", "SAVED", "INTERCEPTED"}
    ]
    priority_ids = {id(event) for event in priority}
    key_events = (priority + [
        event for event in normalized if id(event) not in priority_ids
    ])[:32]
    key_events.sort(key=lambda event: event["t_sec"])
    rows = "".join(
        f'<tr class="replay-{event["type"]} replay-'
        f'{"home" if event["team"] == home else "away" if event["team"] == away else "neutral"} '
        f'{"replay-wm-linked" if event["world_model_link"] else ""} '
        f'{"replay-wm-changed" if event["world_model_link"].get("policy_changed_action") else ""}">'
        f'<td>{html.escape(event["clock"])}</td><td>{html.escape(event["team"])}</td>'
        f'<td>{html.escape(event["actor"])}</td><td>{html.escape(event["kind"])}</td>'
        f'<td>{html.escape(event["outcome"])}</td>'
        f'<td>{"direct identity" if event["world_model_link"] else "—"}</td></tr>'
        for event in key_events
    )
    retained = max(0, int(_finite_float(replay.get("retained_events"))))
    source = max(retained, int(_finite_float(replay.get("source_events"))))
    warnings = []
    if replay.get("retention_truncated") or replay.get("source_limit_reached"):
        warnings.append("display is bounded and sampled")
    if replay.get("logger_truncated"):
        warnings.append("source logger reached its event cap")
    if _finite_float(replay.get("invalid_rows")):
        warnings.append("malformed source rows were ignored")
    warning = "; ".join(warnings) or "all retained actions shown"
    linkage = replay.get("world_model_action_links") or {}
    if not isinstance(linkage, Mapping):
        linkage = {}
    direct_links = max(0, int(_finite_float(linkage.get("directly_observed"))))
    changed_links = max(0, int(_finite_float(
        linkage.get("counterfactual_changed_and_observed")
    )))
    missing_links = max(0, int(_finite_float(
        linkage.get("unresolved_or_missing")
    )))
    linkage_available = bool(linkage.get("available"))
    if linkage_available:
        linkage_summary = f"""<div class="wm-grid replay-link-metrics">
  <div><span class="label">Direct WM links</span><strong>{direct_links}</strong></div>
  <div><span class="label">Changed &amp; observed</span><strong>{changed_links}</strong></div>
  <div><span class="label">Unresolved / missing</span><strong>{missing_links}</strong></div>
</div>
<p class="evidence">A direct link requires the exact runtime opportunity identity, action, team and timestamp. It proves record correspondence, not match-level causal effect.</p>"""
        wm_filters = f"""<input type="radio" name="replay-filter" id="replay-wm"><label for="replay-wm">WM-linked ({direct_links})</label>
<input type="radio" name="replay-filter" id="replay-wm-changed"><label for="replay-wm-changed">WM changed + observed ({changed_links})</label>"""
    else:
        link_reason = str(linkage.get("reason") or "not active for this match")
        linkage_summary = (
            '<p class="notice">World-model trajectory linkage unavailable: '
            f'{html.escape(_action_link_label(link_reason))}.</p>'
        )
        wm_filters = ""
    sequence_inputs, timeline_controls, sequence_navigation, sequence_styles = (
        _replay_sequence_controls(normalized)
    )
    manager_annotations = []
    for annotation in list(replay.get("manager_annotations") or [])[:10]:
        if not isinstance(annotation, Mapping):
            continue
        action = annotation.get("tactic") or "planned substitution"
        manager_annotations.append(
            "<li>"
            f"{_value(annotation.get('minute'))}' · "
            f"{html.escape(str(annotation.get('team') or 'unknown'))} · "
            f"{html.escape(str(annotation.get('condition') or 'always'))} at "
            f"{html.escape(str(annotation.get('score_for', '?')))}-"
            f"{html.escape(str(annotation.get('score_against', '?')))} · "
            f"{html.escape(str(action))} · "
            f"{html.escape(str(annotation.get('status') or 'unknown'))}</li>"
        )
    manager_timeline = (
        "<details class=\"manager-replay-annotations\"><summary>"
        f"In-match manager events ({len(manager_annotations)})</summary><ol>"
        + "".join(manager_annotations) + "</ol></details>"
        if manager_annotations else ""
    )
    return f"""<section id="action-replay-panel" class="card replay-panel" data-testid="action-replay-panel">
<div class="panel-head"><div><div class="label">Inspectable match evidence</div><h2>Ball-action trajectory replay</h2></div><span class="tag">{retained}/{source} actions</span></div>
<p class="evidence">Pass, cross and shot trajectories only—not video or full-player tracking. Coordinates are normalized, bounded and clipped defensively; {html.escape(warning)}.</p>
{linkage_summary}
{manager_timeline}
<fieldset class="replay-filters"><legend>Trajectory filter</legend>
<input type="radio" name="replay-filter" id="replay-all" checked><label for="replay-all">All</label>
<input type="radio" name="replay-filter" id="replay-passes"><label for="replay-passes">Passes</label>
<input type="radio" name="replay-filter" id="replay-crosses"><label for="replay-crosses">Crosses</label>
<input type="radio" name="replay-filter" id="replay-shots"><label for="replay-shots">Shots</label>
<input type="radio" name="replay-filter" id="replay-home"><label for="replay-home">{html.escape(home)}</label>
<input type="radio" name="replay-filter" id="replay-away"><label for="replay-away">{html.escape(away)}</label>
{wm_filters}
{sequence_inputs}
{timeline_controls}
{sequence_navigation}
<div class="replay-stage"><svg viewBox="0 0 1000 640" role="img" aria-label="Ball-action trajectories on a normalized football pitch">
<rect class="pitch" x="20" y="20" width="960" height="600" rx="8"/><line class="marking" x1="500" y1="20" x2="500" y2="620"/><circle class="marking" cx="500" cy="320" r="78"/><circle class="spot" cx="500" cy="320" r="4"/><rect class="marking" x="20" y="155" width="150" height="330"/><rect class="marking" x="830" y="155" width="150" height="330"/><rect class="marking" x="20" y="235" width="55" height="170"/><rect class="marking" x="925" y="235" width="55" height="170"/>{''.join(lines)}</svg></div>
<div class="table-wrap replay-table"><table><thead><tr><th>Time</th><th>Team</th><th>Actor</th><th>Action</th><th>Outcome</th><th>WM record</th></tr></thead><tbody>{rows}</tbody></table></div>
</fieldset>{sequence_styles}</section>"""


def _management_panel(report: Mapping[str, Any]) -> str:
    layers = report.get("layers") or {}
    management = layers.get("management") or {}
    decision = management.get("decision") or {}
    if not isinstance(decision, Mapping) or not decision.get("team"):
        return ""
    fixture = report.get("fixture") or {}
    team = str(decision.get("team"))
    side = "home" if team == fixture.get("home") else "away" if team == fixture.get("away") else ""
    effects = management.get("effects") or {}
    effect = effects.get(side) if side and isinstance(effects, Mapping) else {}
    effect = effect if isinstance(effect, Mapping) else {}
    lineup = effect.get("lineup") or decision.get("lineup") or {}
    lineup = lineup if isinstance(lineup, Mapping) else {}
    starter_names = list(lineup.get("starter_names") or lineup.get("starters") or [])[:11]
    bench_names = list(lineup.get("bench_names") or lineup.get("bench") or [])[:12]
    starters = "".join(
        f"<li>{html.escape(str(name))}</li>" for name in starter_names
    ) or "<li>Player-level lineup unavailable for this roster.</li>"
    bench = "".join(
        f"<li>{html.escape(str(name))}</li>" for name in bench_names
    ) or "<li>No frozen bench reported.</li>"
    base_status = effect.get("base_status")
    effective_status = effect.get("effective_status")
    status_text = (
        f"{_value(base_status)} &rarr; {_value(effective_status)}"
        if base_status is not None or effective_status is not None else "not reported"
    )
    fatigue = (
        _value(effect.get("fatigue_load_multiplier"))
        if effect.get("fatigue_load_multiplier") is not None else "not reported"
    )
    source = str(lineup.get("source") or "team-level fallback")
    in_match = management.get("in_match") or {}
    runtime = in_match.get(side) if side and isinstance(in_match, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    outcome_rows = []
    for outcome in list(runtime.get("outcomes") or [])[:5]:
        if not isinstance(outcome, Mapping):
            continue
        requested = outcome.get("requested_tactic") or "substitution"
        score = f"{outcome.get('score_for', '?')}-{outcome.get('score_against', '?')}"
        outcome_rows.append(
            "<tr>"
            f"<td>{_value(outcome.get('scheduled_minute'))}'</td>"
            f"<td>{html.escape(str(outcome.get('condition') or 'always'))}</td>"
            f"<td>{html.escape(score)}</td>"
            f"<td>{html.escape(str(requested))}</td>"
            f"<td>{html.escape(str(outcome.get('status') or 'unknown'))}: "
            f"{html.escape(str(outcome.get('reason') or ''))}</td></tr>"
        )
    in_match_table = (
        "<h3>In-match instruction outcomes</h3><div class=\"table-wrap\"><table>"
        "<thead><tr><th>Minute</th><th>Condition</th><th>Score</th>"
        "<th>Request</th><th>Outcome</th></tr></thead><tbody>"
        + "".join(outcome_rows) + "</tbody></table></div>"
        if outcome_rows else
        "<p class=\"subtle\">No in-match instruction was evaluated.</p>"
    )
    tactical_execution = management.get("tactical_execution") or {}
    tactical_binding = (
        tactical_execution.get(side)
        if side and isinstance(tactical_execution, Mapping) else None
    )
    tactical_block = (
        "<p class=\"subtle\">This report predates direct tactical-vector "
        "binding evidence.</p>"
    )
    if isinstance(tactical_binding, Mapping):
        vector = tactical_binding.get("initial_vector") or {}
        vector = vector if isinstance(vector, Mapping) else {}
        changed = tactical_binding.get("changed_controls") or []
        changed_count = len(changed) if isinstance(changed, list) else 0
        binding_label = (
            "native team vector"
            if tactical_binding.get("binding_kind") == "native_team_vector"
            else "locked tactical preset"
        )
        tactical_block = (
            "<h3>Direct tactical runtime binding</h3>"
            "<div class=\"wm-grid\">"
            f"<div><span class=\"label\">Binding</span><strong>{html.escape(binding_label)}</strong></div>"
            f"<div><span class=\"label\">Pressing</span><strong>{_percent(vector.get('pressing_intensity'), 0)}</strong></div>"
            f"<div><span class=\"label\">Line height</span><strong>{_percent(vector.get('line_height'), 0)}</strong></div>"
            f"<div><span class=\"label\">Verticality</span><strong>{_percent(vector.get('verticality'), 0)}</strong></div>"
            f"<div><span class=\"label\">Changed dimensions</span><strong>{changed_count}/22</strong></div>"
            "</div>"
        )
    return f"""<section class="card manager-panel" data-testid="manager-decision-panel">
<div class="panel-head"><div><div class="label">Season manager</div><h2>{html.escape(team)} match decision</h2></div><span class="tag">{html.escape(source)} lineup</span></div>
<div class="wm-grid"><div><span class="label">Tactic</span><strong>{html.escape(str(decision.get('tactic') or 'team_identity'))}</strong></div><div><span class="label">Rotation</span><strong>{html.escape(str(decision.get('rotation') or 'balanced'))}</strong></div><div><span class="label">Effective status</span><strong>{status_text}</strong></div><div><span class="label">Fatigue load</span><strong>{fatigue}</strong></div></div>
<div class="lineup-columns"><div><h3>Starting XI</h3><ol>{starters}</ol></div><div><h3>Bench</h3><ul>{bench}</ul></div></div>
{tactical_block}
{in_match_table}
<p class="evidence">The frozen lineup and tactical vector are direct engine inputs for one simulated fixture. They do not establish real-world performance, score causality or tactical effectiveness.</p>
</section>"""


def render_match_html(report: Mapping[str, Any]) -> str:
    fixture, result = report["fixture"], report["result"]
    layers = report.get("layers") or {}
    psychology = layers.get("psychology") or {}
    world_model = layers.get("world_model") or {}
    cognition = layers.get("cognition") or {}
    integrity = report.get("integrity") or {}
    timeline = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in report.get("timeline") or []
    ) or "<li>No major timeline events recorded.</li>"
    raw_json = html.escape(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    integrity_state = str(integrity.get("state", "unknown"))
    shot_source = str(world_model.get("shot_probability_source", "not reported"))
    replay = report.get("replay") or {}
    action_panel = _action_influence_panel(
        world_model, replay if isinstance(replay, Mapping) else {},
    )
    plan_panel = _match_plan_panel(report)
    management_panel = _management_panel(report)
    replay_panel = _action_replay_panel(report)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(fixture['home'])} vs {html.escape(fixture['away'])} &middot; GFS Studio</title>
<style>
:root{{--ink:#e9f1ff;--muted:#9cacbf;--panel:#111b2c;--line:#26344a;--accent:#67e8b5;--warn:#ffc66d}}
*{{box-sizing:border-box}}body{{margin:0;background:#08111f;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:36px 22px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:24px}}
h1{{font-size:clamp(28px,6vw,58px);margin:0;letter-spacing:-.04em}}h2{{margin:0 0 14px;font-size:18px}}.mode{{color:var(--accent);text-transform:uppercase;letter-spacing:.12em}}
.score{{font-size:clamp(44px,9vw,92px);font-weight:800;white-space:nowrap}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:24px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.metric{{font-size:28px;font-weight:700;margin-top:5px}}
.manager-panel{{margin-top:14px}}.lineup-columns{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.lineup-columns>div{{background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:12px}}.lineup-columns h3{{margin:0 0 8px;font-size:14px}}.lineup-columns ol,.lineup-columns ul{{columns:2;column-gap:24px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}ul{{margin:0;padding-left:20px}}details{{margin-top:24px}}pre{{overflow:auto;background:#050b14;padding:18px;border-radius:12px;color:#b9c7da}}
.ok{{color:var(--accent)}}.off,.warn{{color:var(--warn)}}.wm-panel{{margin-top:14px}}.panel-head{{display:flex;justify-content:space-between;gap:12px;align-items:start}}
.wm-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}}.wm-grid>div{{background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:12px}}.wm-grid strong{{display:block;font-size:24px;margin-top:4px}}
.notice,.evidence{{color:var(--muted)}}.evidence{{border-left:3px solid var(--line);padding-left:12px}}.tag{{display:inline-block;border:1px solid #42516a;border-radius:999px;padding:2px 8px;color:#c6d2e3;font-size:12px}}.tag.changed{{border-color:var(--accent);color:var(--accent)}}
.causal-chain{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;list-style-position:inside;padding:0;margin:16px 0}}.causal-chain li{{background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:10px;color:#c6d2e3}}.subtle{{color:var(--muted);font-size:11px}}
.wm-filters{{display:flex;flex-wrap:wrap;gap:7px;border:0;padding:0;margin:18px 0}}.wm-filters legend{{width:100%;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.wm-filters input{{position:absolute;opacity:0;pointer-events:none}}.wm-filters label{{border:1px solid #42516a;border-radius:999px;padding:6px 10px;cursor:pointer}}.wm-filters input:focus-visible+label{{outline:3px solid var(--warn);outline-offset:2px}}.wm-filters input:checked+label{{border-color:var(--accent);color:var(--accent);background:#10281f}}
#wm-filter-changed:checked~.table-wrap tbody tr:not(.decision-changed),#wm-filter-probability:checked~.table-wrap tbody tr:not(.decision-probability),#wm-filter-closed:checked~.table-wrap tbody tr:not(.decision-gate-closed){{display:none}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}}th{{color:var(--muted);font-weight:600}}td details{{margin-top:5px}}.gate-list{{white-space:normal;min-width:260px;color:var(--muted)}}
.replay-panel{{margin-top:14px;scroll-margin-top:16px}}.replay-filters{{display:flex;flex-wrap:wrap;gap:7px;border:0;padding:0;margin:14px 0 0}}.replay-filters legend{{width:100%;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.replay-filters>input{{position:absolute;opacity:0;pointer-events:none}}.replay-filters>label{{border:1px solid #42516a;border-radius:999px;padding:6px 10px;cursor:pointer}}.replay-filters>input:focus-visible+label{{outline:3px solid var(--warn);outline-offset:2px}}.replay-filters>input:checked+label{{border-color:var(--accent);color:var(--accent);background:#10281f}}.replay-stage{{width:100%;margin-top:12px;background:#071c19;border:1px solid #285448;border-radius:12px;overflow:hidden}}.replay-stage svg{{display:block;width:100%;height:auto}}.pitch{{fill:#0c392d;stroke:#b8d8cd;stroke-width:3}}.marking{{fill:none;stroke:#b8d8cd;stroke-width:3}}.spot{{fill:#b8d8cd}}.replay-event line{{stroke-width:4;stroke-linecap:round;opacity:.43}}.replay-event circle{{stroke:none;opacity:.78}}.replay-home line,.replay-home circle{{stroke:#67e8b5;fill:#67e8b5}}.replay-away line,.replay-away circle{{stroke:#ff9f7a;fill:#ff9f7a}}.replay-neutral line,.replay-neutral circle{{stroke:#d6deeb;fill:#d6deeb}}.replay-shot line{{stroke-width:7;opacity:.84}}.replay-important circle{{stroke:#fff;stroke-width:2;opacity:1}}.replay-wm-linked line{{opacity:.95;filter:drop-shadow(0 0 5px #fff)}}.replay-wm-changed circle{{stroke:#fff;stroke-width:4;r:7px}}.replay-table{{margin-top:12px}}#replay-passes:checked~.replay-stage .replay-event:not(.replay-pass),#replay-shots:checked~.replay-stage .replay-event:not(.replay-shot),#replay-home:checked~.replay-stage .replay-event:not(.replay-home),#replay-away:checked~.replay-stage .replay-event:not(.replay-away),#replay-wm:checked~.replay-stage .replay-event:not(.replay-wm-linked),#replay-wm-changed:checked~.replay-stage .replay-event:not(.replay-wm-changed),#replay-passes:checked~.replay-table tr:not(.replay-pass),#replay-shots:checked~.replay-table tr:not(.replay-shot),#replay-home:checked~.replay-table tr:not(.replay-home),#replay-away:checked~.replay-table tr:not(.replay-away),#replay-wm:checked~.replay-table tr:not(.replay-wm-linked),#replay-wm-changed:checked~.replay-table tr:not(.replay-wm-changed){{display:none}}
.replay-cross line{{stroke-dasharray:10 5;stroke-width:6}}#replay-crosses:checked~.replay-stage .replay-event:not(.replay-cross),#replay-crosses:checked~.replay-table tr:not(.replay-cross){{display:none}}
.replay-timeline{{display:flex;gap:5px;overflow:auto;width:100%;padding:12px 2px 5px;scrollbar-color:#42516a transparent}}.replay-timeline label{{flex:0 0 auto;min-width:42px;border:1px solid #42516a;border-radius:8px;padding:4px 7px;text-align:center;cursor:pointer;color:#c6d2e3;font-size:12px}}.replay-timeline label span{{display:block;color:var(--muted);font-size:10px}}.replay-navigation{{width:100%;margin-top:8px}}.sequence-frame{{display:none;align-items:center;justify-content:space-between;gap:12px;background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:10px}}.sequence-frame>div{{display:grid;gap:2px;text-align:center;min-width:0}}.sequence-frame strong,.sequence-frame span{{overflow:hidden;text-overflow:ellipsis}}.sequence-button{{border:1px solid #42516a;border-radius:8px;padding:7px 10px;cursor:pointer;color:var(--accent);white-space:nowrap}}.sequence-frame .tag{{justify-self:center;margin-top:3px}}.replay-event{{transition:opacity .12s ease}}
@media(max-width:650px){{header,.split,.lineup-columns{{display:block}}.lineup-columns>div+div{{margin-top:10px}}.lineup-columns ol,.lineup-columns ul{{columns:1}}.score{{margin-top:20px}}.panel-head{{display:block}}.sequence-frame{{flex-wrap:wrap}}.sequence-frame>div{{order:-1;width:100%}}.sequence-button{{flex:1;text-align:center}}}}
</style></head><body><main>
<header><div><div class="mode">{html.escape(report['studio']['mode'])} mode &middot; GFS Studio</div><h1>{html.escape(fixture['home'])}<br>{html.escape(fixture['away'])}</h1></div><div class="score">{_value(result['score']['home'],0)}&ndash;{_value(result['score']['away'],0)}</div></header>
<section class="grid">
<article class="card"><div class="label">Expected goals</div><div class="metric">{_value(result['xg']['home'])} &ndash; {_value(result['xg']['away'])}</div></article>
<article class="card"><div class="label">Possession</div><div class="metric">{_percent(result['possession']['home'],0)} &ndash; {_percent(result['possession']['away'],0)}</div></article>
<article class="card"><div class="label">Passes</div><div class="metric">{_value(result['passes']['home'],0)} &ndash; {_value(result['passes']['away'],0)}</div></article>
<article class="card"><div class="label">Shots</div><div class="metric">{_value(result['shots']['home'],0)} &ndash; {_value(result['shots']['away'],0)}</div></article>
</section>
<section class="split">
<article class="card"><h2>System layers</h2><p>Integrity: <b class="{'ok' if integrity.get('accepted') else 'off'}">{html.escape(integrity_state)}</b><br>World model: <b class="{'ok' if world_model.get('enabled') else 'off'}">{'active' if world_model.get('enabled') else 'stable fallback'}</b><br>Shot probability: <b>{html.escape(shot_source)}</b><br>Cognition: <b class="{'ok' if cognition.get('enabled') else 'off'}">{'active' if cognition.get('enabled') else 'off'}</b></p></article>
    <article class="card"><h2>Match psychology</h2><p>Crowd field: <b>{_value(psychology.get('crowd_field'))}</b><br>Coach stress: <b>{_value((psychology.get('coach_stress') or {}).get('home'))} / {_value((psychology.get('coach_stress') or {}).get('away'))}</b><br>Tactical drift: <b>{_value((psychology.get('tactical_drift') or {}).get('home'))} / {_value((psychology.get('tactical_drift') or {}).get('away'))}</b></p></article>
</section>
{plan_panel}
{management_panel}
{action_panel}
{replay_panel}
<section class="card" style="margin-top:14px"><h2>Timeline</h2><ul>{timeline}</ul></section>
<details><summary>Complete auditable report</summary><pre>{raw_json}</pre></details>
</main></body></html>"""


def write_match_html(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_match_html(report), encoding="utf-8")
    return target

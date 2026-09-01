"""Recoverable fixed-budget world-model branch-time sensitivity workbench."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.product.manager_future import validate_manager_future_context_shape
from src.product.match_plan import WorldModelForkSetPlan


FUTURE_SET_SCENARIO_STATUSES = {
    "descriptive_only_ineligible",
    "no_realized_action_divergence",
    "action_divergence_without_local_attribution",
    "local_action_divergence_with_descriptive_future_difference",
    "local_action_divergence_without_measured_future_difference",
}


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _paths(workspace: Any, set_id: str) -> dict[str, Path]:
    root = Path(workspace.output_root) / "fork_sets" / set_id
    return {
        "root": root,
        "lease": root / "fork_set.lock",
        "protocol": root / "protocol.json",
        "progress": root / "progress.json",
        "result": root / "result.json",
        "dashboard": root / "index.html",
    }


def _resolve_comparison(workspace: Any, value: Any) -> Path:
    workspace_root = Path(workspace.root).resolve()
    matches_root = (Path(workspace.output_root) / "matches").resolve()
    candidate = Path(str(value or ""))
    resolved = (candidate if candidate.is_absolute() else workspace_root / candidate).resolve()
    try:
        resolved.relative_to(matches_root)
    except ValueError as exc:
        raise ValueError("fork-set comparison must stay inside Studio matches") from exc
    if not resolved.name.endswith(".comparison.json") or not resolved.is_file():
        raise ValueError("fork-set comparison artifact is unavailable")
    return resolved


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def validate_fork_set_scenario_evidence(
    scenarios: Any, branch_times_sec: Any,
) -> list[dict[str, Any]]:
    """Replay the bounded per-timepoint evidence exposed to product surfaces."""
    if (
        not isinstance(branch_times_sec, (list, tuple))
        or not 2 <= len(branch_times_sec) <= 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 5400
            or (
                index > 0
                and float(value) <= float(branch_times_sec[index - 1])
            )
            for index, value in enumerate(branch_times_sec)
        )
    ):
        raise ValueError("future-set scenario branch contract is invalid")
    expected_times = [float(value) for value in branch_times_sec]
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != len(expected_times)
    ):
        raise ValueError("future-set scenario evidence budget is invalid")
    required = {
        "schema_version", "branch_at_sec", "branch_minute",
        "future_status", "eligible", "anchor_verified",
        "branch_state_identity",
        "changed_actions", "locally_attributable_changes",
        "descriptive_future_difference_count",
        "simulator_local_action_attribution",
        "outcome_causality_authorized", "real_football_causality_authorized",
        "scenario_identity",
    }
    normalized = []
    for index, raw in enumerate(scenarios):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("future-set scenario evidence fields are invalid")
        branch = raw.get("branch_at_sec")
        minute = raw.get("branch_minute")
        state_identity = raw.get("branch_state_identity")
        if (
            raw.get("schema_version") != 1
            or isinstance(branch, bool)
            or not isinstance(branch, (int, float))
            or not math.isfinite(float(branch))
            or float(branch) != expected_times[index]
            or isinstance(minute, bool)
            or not isinstance(minute, (int, float))
            or not math.isclose(
                float(minute), float(branch) / 60.0,
                rel_tol=0.0, abs_tol=1e-9,
            )
            or raw.get("future_status") not in FUTURE_SET_SCENARIO_STATUSES
            or not isinstance(raw.get("eligible"), bool)
            or not isinstance(raw.get("anchor_verified"), bool)
            or (
                raw.get("eligible") is True
                and raw.get("anchor_verified") is not True
            )
            or (
                state_identity is not None
                and re.fullmatch(r"[0-9a-f]{64}", str(state_identity)) is None
            )
            or (
                raw.get("anchor_verified") is True
                and state_identity is None
            )
            or not isinstance(
                raw.get("simulator_local_action_attribution"), bool,
            )
            or raw.get("outcome_causality_authorized") is not False
            or raw.get("real_football_causality_authorized") is not False
        ):
            raise ValueError("future-set scenario evidence values are invalid")
        for field in (
            "changed_actions", "locally_attributable_changes",
            "descriptive_future_difference_count",
        ):
            value = raw.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 100_000
            ):
                raise ValueError("future-set scenario evidence counts are invalid")
        changed = int(raw["changed_actions"])
        local = int(raw["locally_attributable_changes"])
        differences = int(raw["descriptive_future_difference_count"])
        local_authorized = raw["simulator_local_action_attribution"]
        status = raw["future_status"]
        semantic_status = (
            "descriptive_only_ineligible"
            if not raw["eligible"] else
            "no_realized_action_divergence"
            if changed == 0 else
            "action_divergence_without_local_attribution"
            if not local_authorized else
            "local_action_divergence_with_descriptive_future_difference"
            if differences > 0 else
            "local_action_divergence_without_measured_future_difference"
        )
        if (
            local > changed
            or (local_authorized and local == 0)
            or (not raw["eligible"] and local_authorized)
            or status != semantic_status
        ):
            raise ValueError("future-set scenario evidence semantics are invalid")
        frozen = dict(raw)
        observed = frozen.pop("scenario_identity")
        if (
            not isinstance(observed, str)
            or observed != _identity(frozen)
        ):
            raise ValueError("future-set scenario evidence identity mismatch")
        normalized.append(dict(raw))
    return normalized


def summarize_fork_set_scenario_evidence(
    scenarios: Any, branch_times_sec: Any,
) -> dict[str, Any]:
    """Rebuild aggregate facts from the bounded scenario evidence."""
    validated = validate_fork_set_scenario_evidence(
        scenarios, branch_times_sec,
    )
    signatures = {
        (
            row["future_status"], row["changed_actions"],
            row["locally_attributable_changes"],
            row["descriptive_future_difference_count"],
        )
        for row in validated
    }
    return {
        "eligible_scenarios": sum(
            row["eligible"] for row in validated
        ),
        "verified_anchor_scenarios": sum(
            row["anchor_verified"] for row in validated
        ),
        "action_divergence_scenarios": sum(
            row["changed_actions"] > 0 for row in validated
        ),
        "local_attribution_scenarios": sum(
            row["simulator_local_action_attribution"]
            for row in validated
        ),
        "descriptive_future_difference_scenarios": sum(
            row["descriptive_future_difference_count"] > 0
            for row in validated
        ),
        "timing_sensitivity_observed": len(signatures) > 1,
        "status_counts": dict(sorted(Counter(
            row["future_status"] for row in validated
        ).items())),
    }


def project_fork_set_scenario_evidence(
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a privacy-safe, non-ranked product projection from official rows."""
    plan = result.get("plan")
    rows = result.get("rows")
    branch_times = (
        plan.get("branch_times_sec")
        if isinstance(plan, Mapping) else None
    )
    if (
        not isinstance(rows, list)
        or not isinstance(branch_times, list)
        or len(rows) != len(branch_times)
    ):
        raise ValueError("future-set scenario source is invalid")
    scenarios = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("future-set scenario source row is invalid")
        raw_branch = row.get("branch_at_sec")
        raw_minute = row.get("branch_minute")
        if (
            isinstance(raw_branch, bool)
            or not isinstance(raw_branch, (int, float))
            or not math.isfinite(float(raw_branch))
            or isinstance(raw_minute, bool)
            or not isinstance(raw_minute, (int, float))
            or not math.isfinite(float(raw_minute))
        ):
            raise ValueError("future-set scenario source time is invalid")
        raw_state_identity = str(row.get("branch_state_identity") or "")
        state_identity = (
            raw_state_identity
            if re.fullmatch(r"[0-9a-f]{64}", raw_state_identity)
            else None
        )
        payload = {
            "schema_version": 1,
            "branch_at_sec": float(raw_branch),
            "branch_minute": float(raw_minute),
            "future_status": str(row.get("future_status") or ""),
            "eligible": row.get("eligible") is True,
            "anchor_verified": row.get("branch_anchor_verified") is True,
            "branch_state_identity": state_identity,
            "changed_actions": _bounded_count(row.get("changed_actions")),
            "locally_attributable_changes": _bounded_count(
                row.get("locally_attributable_changes")
            ),
            "descriptive_future_difference_count": _bounded_count(
                row.get("descriptive_future_difference_count")
            ),
            "simulator_local_action_attribution": (
                row.get("simulator_local_action_attribution") is True
            ),
            "outcome_causality_authorized": False,
            "real_football_causality_authorized": False,
        }
        scenarios.append({
            **payload, "scenario_identity": _identity(payload),
        })
    return validate_fork_set_scenario_evidence(scenarios, branch_times)


def _row(plan: WorldModelForkSetPlan, comparison: Mapping[str, Any]) -> dict[str, Any]:
    fixture = comparison.get("fixture") or {}
    intervention = comparison.get("intervention") or {}
    anchor = intervention.get("branch_anchor") or {}
    eligibility = comparison.get("eligibility") or {}
    propagation = comparison.get("policy_propagation") or {}
    summary = propagation.get("summary") or {}
    future = comparison.get("counterfactual_future_summary") or {}
    future_summary = future.get("summary") or {}
    authority = future.get("claim_authority") or {}
    branch = intervention.get("branch_at_sec")
    if isinstance(branch, bool) or not isinstance(branch, (int, float)):
        raise ValueError("comparison branch time is invalid")
    if branch not in plan.branch_times_sec:
        raise ValueError("comparison branch time is outside frozen fork set")
    if (
        fixture.get("seed") != plan.seed
        or intervention.get("scope") != "world_model_action_policy"
        or intervention.get("changed_sides") != []
        or intervention.get("baseline_policy") != "predict_only"
        or intervention.get("treatment_policy") != "action_policy"
        or intervention.get("baseline_tactics") != {
            "home": plan.home_tactic, "away": plan.away_tactic,
        }
        or intervention.get("treatment_tactics") != {
            "home": plan.home_tactic, "away": plan.away_tactic,
        }
    ):
        raise ValueError("comparison is not the frozen world-model policy contrast")
    changed = _bounded_count(summary.get("valid_changed_decisions"))
    local = _bounded_count(summary.get("locally_attributable_changes"))
    differences = _bounded_count(
        future_summary.get("descriptive_outcome_difference_count")
    )
    eligible = bool(
        propagation.get("available") is True
        and future.get("available") is True
        and anchor.get("verified") is True
        and eligibility.get(
            "eligible_for_world_model_policy_attribution"
        ) is True
        and re.fullmatch(
            r"[0-9a-f]{64}", str(anchor.get("state_identity") or "")
        ) is not None
    )
    local_authorized = bool(
        eligible and local > 0
        and authority.get("simulator_local_action_attribution") is True
    )
    status = (
        "descriptive_only_ineligible" if not eligible else
        "no_realized_action_divergence" if changed == 0 else
        "action_divergence_without_local_attribution"
        if not local_authorized else
        "local_action_divergence_with_descriptive_future_difference"
        if differences > 0 else
        "local_action_divergence_without_measured_future_difference"
    )
    return {
        "branch_at_sec": float(branch),
        "branch_minute": float(branch) / 60.0,
        "eligible": eligible,
        "branch_anchor_verified": anchor.get("verified") is True,
        "branch_state_identity": str(anchor.get("state_identity") or "")[:64],
        "future_status": status,
        "changed_actions": changed,
        "locally_attributable_changes": local,
        "descriptive_future_difference_count": differences,
        "simulator_local_action_attribution": local_authorized,
        "outcome_causality": False,
        "real_football_causality": False,
    }


def aggregate_fork_set(
    plan: WorldModelForkSetPlan,
    comparisons: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(comparisons) != len(plan.branch_times_sec):
        raise ValueError("fork-set aggregation requires the complete fixed budget")
    rows = [_row(plan, comparison) for comparison in comparisons]
    if [row["branch_at_sec"] for row in rows] != list(plan.branch_times_sec):
        raise ValueError("fork-set comparisons do not match frozen branch order")
    status_counts = Counter(row["future_status"] for row in rows)
    signatures = {
        (
            row["future_status"], row["changed_actions"],
            row["locally_attributable_changes"],
            row["descriptive_future_difference_count"],
        )
        for row in rows
    }
    return {
        "rows": rows,
        "aggregate": {
            "eligible_scenarios": sum(row["eligible"] for row in rows),
            "verified_anchor_scenarios": sum(
                row["branch_anchor_verified"] for row in rows
            ),
            "action_divergence_scenarios": sum(
                row["changed_actions"] > 0 for row in rows
            ),
            "local_attribution_scenarios": sum(
                row["simulator_local_action_attribution"] for row in rows
            ),
            "descriptive_future_difference_scenarios": sum(
                row["descriptive_future_difference_count"] > 0 for row in rows
            ),
            "timing_sensitivity_observed": len(signatures) > 1,
            "status_counts": dict(sorted(status_counts.items())),
            "ranking_performed": False,
            "best_branch_time": None,
        },
        "claim_authority": {
            "descriptive_simulator_timing_sensitivity": True,
            "best_time_recommendation": False,
            "match_outcome_causality": False,
            "population_inference": False,
            "real_football_causality": False,
            "promotion_authorized": False,
        },
    }


def render_fork_set_html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr><td>{:.1f}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        "<td><a href='{}'>查看双世界证据</a></td></tr>".format(
            float(row.get("branch_minute", 0.0)),
            "已验证" if row.get("branch_anchor_verified") else "未验证",
            html.escape(str(row.get("future_status") or "unknown")),
            int(row.get("changed_actions", 0)),
            int(row.get("descriptive_future_difference_count", 0)),
            html.escape(str((row.get("evidence") or {}).get("comparison_dashboard") or "#"), quote=True),
        )
        for row in result.get("rows") or []
    )
    aggregate = result.get("aggregate") or {}
    fixture = result.get("fixture") or {}
    boundary = html.escape(str(result.get("claim_boundary") or ""))
    source = result.get("source_context") or {}
    manager_binding = (
        "<section><h2>经理决策绑定</h2><p>赛季 {} · 轮次 {} · 决策上下文 {}</p>"
        "<p>本报告复用该冻结决策的正式对阵、seed 与双方实际战术；"
        "若赛季 revision 改变，任务会失败关闭。</p></section>"
    ).format(
        html.escape(str(source.get("season_id") or "")),
        html.escape(str((source.get("fixture") or {}).get("matchday") or "")),
        html.escape(str(source.get("context_identity") or "")[:16]),
    ) if source.get("kind") == "manager_prematch_world_model_future_set" else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GFS 多时点未来分叉</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{max-width:1100px;margin:auto;padding:32px 20px;background:#07111e;color:#edf4ff;font:15px/1.6 system-ui}}section{{background:#111d2e;border:1px solid #2a3a51;border-radius:14px;padding:18px;margin:16px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #2a3a51}}a{{color:#65e6b4}}.muted{{color:#a8b6c9}}a:focus-visible{{outline:3px solid #ffc36a}}</style></head><body><p class="muted">Football Causal World Lab</p><h1>{html.escape(str(fixture.get('home') or ''))} vs {html.escape(str(fixture.get('away') or ''))} · 多时点未来分叉</h1>{manager_binding}<section><h2>固定预算已完成</h2><p>{int(result.get('scenarios_completed', 0))} / {int(result.get('fixed_scenario_budget', 0))} 个预注册时点；动作分叉 {int(aggregate.get('action_divergence_scenarios', 0))} 个，局部归因 {int(aggregate.get('local_attribution_scenarios', 0))} 个，未来描述差异 {int(aggregate.get('descriptive_future_difference_scenarios', 0))} 个。</p><p>时间敏感性：{'已观察到' if aggregate.get('timing_sensitivity_observed') else '在已测指标中未观察到'}。系统没有挑选最佳时点。</p></section><section><h2>同一控制条件下的未来集</h2><table><thead><tr><th>分叉分钟</th><th>前缀锚点</th><th>证据状态</th><th>动作变化</th><th>后续描述差异</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>结论边界</h2><p>{boundary}</p><p>这是模拟器内时间敏感性描述，不授权赛果因果、总体推断、现实足球因果或产品/论文晋级。</p></section></body></html>"""


def _validate_completed_result(
    plan: WorldModelForkSetPlan, set_id: str, result: Mapping[str, Any],
    *, home: str, away: str, source_context: Mapping[str, Any] | None,
) -> None:
    rows = result.get("rows") or []
    aggregate = result.get("aggregate") or {}
    authority = result.get("claim_authority") or {}
    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("set_id") != set_id
        or result.get("fixture") != {"home": home, "away": away}
        or result.get("source_context") != (
            dict(source_context) if source_context is not None else None
        )
        or result.get("plan") != plan.as_dict()
        or result.get("fixed_scenario_budget") != len(plan.branch_times_sec)
        or result.get("scenarios_completed") != len(plan.branch_times_sec)
        or result.get("interim_ranking_disclosed") is not False
        or not isinstance(rows, list)
        or len(rows) != len(plan.branch_times_sec)
        or not isinstance(aggregate, Mapping)
        or aggregate.get("ranking_performed") is not False
        or aggregate.get("best_branch_time") is not None
        or not isinstance(authority, Mapping)
        or authority.get("best_time_recommendation") is not False
        or authority.get("match_outcome_causality") is not False
        or authority.get("population_inference") is not False
        or authority.get("real_football_causality") is not False
        or authority.get("promotion_authorized") is not False
    ):
        raise ValueError("completed fork-set result identity is invalid")
    if [row.get("branch_at_sec") for row in rows if isinstance(row, Mapping)] != list(
        plan.branch_times_sec
    ):
        raise ValueError("completed fork-set branch identity is invalid")
    if any(
        not isinstance(row, Mapping)
        or row.get("outcome_causality") is not False
        or row.get("real_football_causality") is not False
        for row in rows
    ):
        raise ValueError("completed fork-set claim boundary is invalid")
    scenarios = project_fork_set_scenario_evidence(result)
    rebuilt = summarize_fork_set_scenario_evidence(
        scenarios, plan.branch_times_sec,
    )
    if any(aggregate.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("completed fork-set scenario aggregate mismatch")


def execute_world_model_fork_set(
    workspace: Any, plan: WorldModelForkSetPlan, *, set_id: str,
    home: str, away: str, fast: bool,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume all preregistered branch times without interim ranking."""
    if getattr(getattr(workspace, "config", None), "mode", None) != "research":
        raise ValueError("world-model fork sets require research mode")
    if not set_id.isalnum() or len(set_id) > 64:
        raise ValueError("invalid fork-set identity")
    if source_context is not None:
        source_context = validate_manager_future_context_shape(source_context)
        if (
            source_context["fixture"]["home"] != home
            or source_context["fixture"]["away"] != away
            or source_context["match_seed"] != plan.seed
            or source_context["fast"] is not fast
            or source_context["home_tactic"] != plan.home_tactic
            or source_context["away_tactic"] != plan.away_tactic
        ):
            raise ValueError("future-set source context does not match controls")
    from src.infrastructure import FileLease
    from src.product.workspace import _atomic_json

    paths = _paths(workspace, set_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": 1,
        "set_id": set_id,
        "fixture": {"home": home, "away": away},
        "fast": fast,
        "plan": plan.as_dict(),
        "source_context": (
            dict(source_context) if source_context is not None else None
        ),
    }
    with FileLease(paths["lease"], timeout=5.0):
        if paths["protocol"].is_file():
            if json.loads(paths["protocol"].read_text(encoding="utf-8")) != frozen:
                raise ValueError("fork-set identity belongs to a different frozen plan")
        else:
            _atomic_json(paths["protocol"], frozen)
        if paths["result"].is_file():
            result = json.loads(paths["result"].read_text(encoding="utf-8"))
            if not isinstance(result, Mapping):
                raise ValueError("completed fork-set result must be an object")
            _validate_completed_result(
                plan, set_id, result, home=home, away=away,
                source_context=source_context,
            )
            if not paths["dashboard"].is_file():
                paths["dashboard"].write_text(render_fork_set_html(result), encoding="utf-8")
            return {**result, "result_path": str(paths["result"]), "dashboard_path": str(paths["dashboard"])}
        progress = {
            "schema_version": 1,
            "set_id": set_id,
            "state": "running",
            "fixed_scenario_budget": len(plan.branch_times_sec),
            "scenarios_completed": 0,
            "completed": [],
            "interim_ranking_disclosed": False,
            "analysis": None,
        }
        if paths["progress"].is_file():
            loaded = json.loads(paths["progress"].read_text(encoding="utf-8"))
            if loaded.get("set_id") != set_id:
                raise ValueError("invalid fork-set progress identity")
            progress["completed"] = list(loaded.get("completed") or [])
        observed = [
            item.get("branch_at_sec") if isinstance(item, Mapping) else None
            for item in progress["completed"]
        ]
        if observed != list(plan.branch_times_sec[:len(observed)]):
            raise ValueError("fork-set progress is not a valid frozen prefix")
        progress["scenarios_completed"] = len(observed)
        _atomic_json(paths["progress"], progress)
        try:
            for index, branch in enumerate(plan.branch_times_sec):
                if index < len(observed):
                    continue
                fork = plan.fork_plan(branch)
                baseline, treatment = workspace.run_paired_matches(
                    home, away, fast=fast,
                    baseline_plan=fork.baseline_plan(),
                    treatment_plan=fork.treatment_plan(), seed=plan.seed,
                    transaction_id=f"{set_id}-{index}",
                )
                resolved = _resolve_comparison(workspace, treatment.get("comparison_path"))
                comparison = json.loads(resolved.read_text(encoding="utf-8"))
                if (comparison.get("fixture") or {}).get("home") != home or (
                    comparison.get("fixture") or {}
                ).get("away") != away:
                    raise ValueError("fork-set comparison fixture identity mismatch")
                _row(plan, comparison)
                entry = {
                    "branch_at_sec": branch,
                    "baseline_match_id": baseline["match_id"],
                    "treatment_match_id": treatment["match_id"],
                    "comparison": resolved.relative_to(Path(workspace.root).resolve()).as_posix(),
                }
                progress["completed"].append(entry)
                progress["scenarios_completed"] = len(progress["completed"])
                _atomic_json(paths["progress"], progress)
            comparisons = []
            evidence = {}
            for entry in progress["completed"]:
                comparison_path = _resolve_comparison(workspace, entry["comparison"])
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                if (comparison.get("fixture") or {}).get("home") != home or (
                    comparison.get("fixture") or {}
                ).get("away") != away:
                    raise ValueError("fork-set comparison fixture identity mismatch")
                comparisons.append(comparison)
                dashboard = comparison_path.with_suffix(".html")
                if not dashboard.is_file():
                    raise ValueError("fork-set comparison dashboard is unavailable")
                evidence[float(entry["branch_at_sec"])] = {
                    "comparison_dashboard": Path(os.path.relpath(
                        dashboard, paths["root"],
                    )).as_posix()
                }
            analysis = aggregate_fork_set(plan, comparisons)
            for row in analysis["rows"]:
                row["evidence"] = evidence[row["branch_at_sec"]]
            result = {
                "schema_version": 1,
                "set_id": set_id,
                "status": "complete",
                "fixture": {"home": home, "away": away},
                "fast": fast,
                "plan": plan.as_dict(),
                "source_context": (
                    dict(source_context) if source_context is not None else None
                ),
                "fixed_scenario_budget": len(plan.branch_times_sec),
                "scenarios_completed": len(plan.branch_times_sec),
                "interim_ranking_disclosed": False,
                **analysis,
                "claim_boundary": plan.as_dict()["claim_boundary"],
            }
            _atomic_json(paths["result"], result)
            paths["dashboard"].write_text(render_fork_set_html(result), encoding="utf-8")
            progress.update({"state": "complete", "analysis": None})
            _atomic_json(paths["progress"], progress)
            return {**result, "result_path": str(paths["result"]), "dashboard_path": str(paths["dashboard"])}
        except BaseException as exc:
            progress.update({
                "state": "failed", "error_type": type(exc).__name__,
                "analysis": None, "interim_ranking_disclosed": False,
            })
            _atomic_json(paths["progress"], progress)
            raise

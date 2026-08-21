"""Fixed-budget, shared-seed tactical studies without optional stopping."""

from __future__ import annotations

import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.product.match_plan import MatchPlan


PRIMARY_METRIC = "focus_xg_difference"
SECONDARY_METRICS = (
    "focus_goal_difference",
    "focus_possession",
    "focus_passes",
    "focus_shots",
    "wm_expected_changes",
)
ALL_STUDY_METRICS = (PRIMARY_METRIC, *SECONDARY_METRICS)
MIN_PAIRS = 4
MAX_PAIRS = 32
ALPHA = 0.05
DEFAULT_SESOI = 0.05
_T_CRITICAL_975 = {
    3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
    28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040,
}


@dataclass(frozen=True)
class TacticalStudyPlan:
    study_id: str
    home: str
    away: str
    baseline_home_tactic: str
    baseline_away_tactic: str
    treatment_home_tactic: str
    treatment_away_tactic: str
    seeds: tuple[int, ...]
    fast: bool = True
    smallest_effect_size: float = DEFAULT_SESOI

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{2,63}", self.study_id
        ):
            raise ValueError("study_id must be a stable lowercase slug")
        if (
            not isinstance(self.home, str) or not isinstance(self.away, str)
            or not self.home.strip() or not self.away.strip()
        ):
            raise ValueError("study fixture teams must not be empty")
        if self.home.casefold() == self.away.casefold():
            raise ValueError("study fixture teams must differ")
        if not isinstance(self.fast, bool):
            raise ValueError("fast must be boolean")
        normalized_seeds = tuple(self.seeds)
        if not MIN_PAIRS <= len(normalized_seeds) <= MAX_PAIRS:
            raise ValueError(f"study requires {MIN_PAIRS} to {MAX_PAIRS} seed pairs")
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("study seeds must be unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed <= 2**31 - 1
            for seed in normalized_seeds
        ):
            raise ValueError("study seeds must be 32-bit non-negative integers")
        object.__setattr__(self, "seeds", normalized_seeds)
        try:
            smallest_effect = float(self.smallest_effect_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "smallest_effect_size must be finite and positive"
            ) from exc
        if not math.isfinite(smallest_effect) or smallest_effect <= 0:
            raise ValueError("smallest_effect_size must be finite and positive")
        baseline = MatchPlan(
            experience="tactical_lab",
            home_tactic=self.baseline_home_tactic,
            away_tactic=self.baseline_away_tactic,
        )
        treatment = MatchPlan(
            experience="tactical_lab",
            home_tactic=self.treatment_home_tactic,
            away_tactic=self.treatment_away_tactic,
            reuse_last_seed=True,
        )
        changed = [
            side for side in ("home", "away")
            if getattr(baseline, f"{side}_tactic")
            != getattr(treatment, f"{side}_tactic")
        ]
        if len(changed) != 1:
            raise ValueError("fixed tactical study requires exactly one changed side")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TacticalStudyPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("tactical study plan must be an object")
        fixture = payload.get("fixture") or {}
        baseline = payload.get("baseline") or {}
        treatment = payload.get("treatment") or {}
        analysis = payload.get("analysis_plan") or {}
        return cls(
            study_id=payload.get("study_id"),
            home=fixture.get("home"), away=fixture.get("away"),
            baseline_home_tactic=baseline.get("home_tactic"),
            baseline_away_tactic=baseline.get("away_tactic"),
            treatment_home_tactic=treatment.get("home_tactic"),
            treatment_away_tactic=treatment.get("away_tactic"),
            seeds=tuple(payload.get("seeds") or ()),
            fast=payload.get("fast", True),
            smallest_effect_size=analysis.get(
                "smallest_effect_size", DEFAULT_SESOI,
            ),
        )

    @property
    def focus_side(self) -> str:
        return "home" if (
            self.baseline_home_tactic != self.treatment_home_tactic
        ) else "away"

    def baseline_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="tactical_lab",
            home_tactic=self.baseline_home_tactic,
            away_tactic=self.baseline_away_tactic,
        )

    def treatment_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="tactical_lab",
            home_tactic=self.treatment_home_tactic,
            away_tactic=self.treatment_away_tactic,
            reuse_last_seed=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study_id": self.study_id,
            "fixture": {"home": self.home, "away": self.away},
            "baseline": self.baseline_plan().as_dict(),
            "treatment": self.treatment_plan().as_dict(),
            "focus_side": self.focus_side,
            "seeds": list(self.seeds),
            "fixed_pair_budget": len(self.seeds),
            "fast": self.fast,
            "analysis_plan": {
                "primary_metric": PRIMARY_METRIC,
                "secondary_metrics": list(SECONDARY_METRICS),
                "alpha": ALPHA,
                "smallest_effect_size": float(self.smallest_effect_size),
                "missing_data": "fail_metric_closed_no_complete_case_substitution",
                "interim_analysis": "withheld_until_fixed_budget_complete",
                "decision_rule": "two_sided_t_interval_and_preregistered_sesoi",
            },
            "claim_boundary": (
                "mean paired contrast over this fixed fixture and seed set only; "
                "not a cross-fixture population effect or promotion authorization"
            ),
        }


def _metric_value(comparison: Mapping[str, Any], metric: str, field: str) -> float:
    value = ((comparison.get("metrics") or {}).get(metric) or {}).get(field)
    if isinstance(value, bool):
        raise ValueError(f"invalid comparison metric {metric}.{field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing comparison metric {metric}.{field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"nonfinite comparison metric {metric}.{field}")
    return number


def _validate_comparison(
    plan: TacticalStudyPlan, comparison: Mapping[str, Any],
) -> int:
    fixture = comparison.get("fixture") or {}
    if (fixture.get("home"), fixture.get("away")) != (plan.home, plan.away):
        raise ValueError("comparison fixture does not match frozen study plan")
    seed = fixture.get("seed")
    if seed not in plan.seeds:
        raise ValueError("comparison seed is outside frozen study plan")
    eligibility = comparison.get("eligibility") or {}
    if not eligibility.get("eligible_for_tactical_attribution"):
        raise ValueError("ineligible comparison cannot enter tactical study")
    intervention = comparison.get("intervention") or {}
    if intervention.get("scope") != "single_side_tactical_intervention":
        raise ValueError("study accepts only single-side tactical interventions")
    if intervention.get("changed_sides") != [plan.focus_side]:
        raise ValueError("comparison changed side does not match study focus")
    expected_baseline = {
        "home": plan.baseline_home_tactic,
        "away": plan.baseline_away_tactic,
    }
    expected_treatment = {
        "home": plan.treatment_home_tactic,
        "away": plan.treatment_away_tactic,
    }
    if intervention.get("baseline_tactics") != expected_baseline:
        raise ValueError("comparison baseline tactics do not match frozen plan")
    if intervention.get("treatment_tactics") != expected_treatment:
        raise ValueError("comparison treatment tactics do not match frozen plan")
    return int(seed)


def _focused_effects(
    plan: TacticalStudyPlan, comparison: Mapping[str, Any],
) -> dict[str, float]:
    focus, other = ("home", "away") if plan.focus_side == "home" else ("away", "home")
    score_focus = _metric_value(comparison, f"score_{focus}", "delta")
    score_other = _metric_value(comparison, f"score_{other}", "delta")
    xg_focus = _metric_value(comparison, f"xg_{focus}", "delta")
    xg_other = _metric_value(comparison, f"xg_{other}", "delta")
    return {
        "focus_xg_difference": xg_focus - xg_other,
        "focus_goal_difference": score_focus - score_other,
        "focus_possession": _metric_value(
            comparison, f"possession_{focus}", "delta"
        ),
        "focus_passes": _metric_value(comparison, f"passes_{focus}", "delta"),
        "focus_shots": _metric_value(comparison, f"shots_{focus}", "delta"),
        "wm_expected_changes": _metric_value(
            comparison, "wm_expected_changes", "delta"
        ),
    }


def _sign_test(values: list[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    effective = positive + negative
    if effective == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(effective, index)
            for index in range(min(positive, negative) + 1)
        ) / (2 ** effective)
        p_value = min(1.0, 2.0 * tail)
    return {
        "positive": positive, "negative": negative,
        "ties": len(values) - effective,
        "two_sided_exact_p": p_value,
    }


def _summary(values: list[float], *, inferential: bool) -> dict[str, Any]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    supporting = sum(
        value > 0 if mean > 0 else value < 0 if mean < 0 else value == 0
        for value in values
    )
    result: dict[str, Any] = {
        "n": len(values), "mean": mean, "sample_sd": sd,
        "minimum": min(values), "maximum": max(values),
        "direction_consistency": supporting / len(values),
    }
    if inferential:
        standard_error = sd / math.sqrt(len(values))
        critical = _T_CRITICAL_975[len(values) - 1]
        result.update({
            "standard_error": standard_error,
            "confidence_level": 1.0 - ALPHA,
            "ci_low": mean - critical * standard_error,
            "ci_high": mean + critical * standard_error,
            "sign_test": _sign_test(values),
        })
    return result


def analyze_fixed_tactical_study(
    plan: TacticalStudyPlan,
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Withhold all effects until every frozen seed has one eligible comparison."""
    by_seed: dict[int, Mapping[str, Any]] = {}
    for comparison in comparisons:
        seed = _validate_comparison(plan, comparison)
        if seed in by_seed:
            raise ValueError("duplicate comparison seed in tactical study")
        by_seed[seed] = comparison
    missing = [seed for seed in plan.seeds if seed not in by_seed]
    if missing:
        return {
            "schema_version": 1,
            "study_id": plan.study_id,
            "status": "incomplete_analysis_withheld",
            "fixed_pair_budget": len(plan.seeds),
            "pairs_completed": len(by_seed),
            "missing_seeds": missing,
            "interim_effects_disclosed": False,
            "analysis": None,
        }
    rows = []
    for seed in plan.seeds:
        rows.append({"seed": seed, **_focused_effects(plan, by_seed[seed])})
    summaries = {
        metric: _summary(
            [row[metric] for row in rows], inferential=metric == PRIMARY_METRIC,
        )
        for metric in ALL_STUDY_METRICS
    }
    primary = summaries[PRIMARY_METRIC]
    primary_mean = float(primary["mean"])
    for row in rows:
        value = float(row[PRIMARY_METRIC])
        row["primary_direction"] = (
            "positive" if value > 0 else "negative" if value < 0 else "tie"
        )
        row["supports_aggregate_direction"] = (
            value > 0 if primary_mean > 0
            else value < 0 if primary_mean < 0
            else value == 0
        )
    sesoi = float(plan.smallest_effect_size)
    if primary["mean"] >= sesoi and primary["ci_low"] > 0:
        decision = "beneficial_on_fixed_seed_set"
    elif primary["mean"] <= -sesoi and primary["ci_high"] < 0:
        decision = "harmful_on_fixed_seed_set"
    else:
        decision = "inconclusive_on_fixed_seed_set"
    return {
        "schema_version": 1,
        "study_id": plan.study_id,
        "status": "complete",
        "fixed_pair_budget": len(plan.seeds),
        "pairs_completed": len(rows),
        "interim_effects_disclosed": False,
        "plan": plan.as_dict(),
        "rows": rows,
        "analysis": {
            "primary_metric": PRIMARY_METRIC,
            "primary_decision": decision,
            "summaries": summaries,
            "secondary_metrics_are_descriptive": True,
            "promotion_authorized": False,
        },
        "claim_boundary": plan.as_dict()["claim_boundary"],
    }


def _finite_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _study_number(value: Any, *, signed: bool = False, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "&mdash;"
    if not math.isfinite(number):
        return "&mdash;"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _study_evidence_href(value: Any) -> str | None:
    raw = str(value or "")
    if not raw or "\\" in raw:
        return None
    path = Path(raw)
    parts = path.parts
    filename = path.name
    if (
        path.is_absolute() or ".." in parts
        or len(parts) < 2 or parts[-2] != "matches"
        or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.html", filename)
    ):
        return None
    return f"../../matches/{filename}"


def _study_evidence_links(row: Mapping[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return '<span class="muted">证据链接不可用</span>'
    links = []
    for label, key in (
        ("基线", "baseline_dashboard"),
        ("处理", "treatment_dashboard"),
        ("配对", "comparison_dashboard"),
    ):
        href = _study_evidence_href(evidence.get(key))
        if href:
            links.append(
                f'<a href="{html.escape(href, quote=True)}">{label}</a>'
            )
    return " · ".join(links) if links else (
        '<span class="muted">证据链接不可用</span>'
    )


def render_tactical_study_html(result: Mapping[str, Any]) -> str:
    status = str(result.get("status") or "unknown")
    if status != "complete":
        completed = max(0, int(_finite_or_zero(result.get("pairs_completed"))))
        budget = max(0, int(_finite_or_zero(result.get("fixed_pair_budget"))))
        body = (
            f"<h1>战术研究尚未完成</h1><p>{completed}/{budget} 个配对已完成。</p>"
            "<p>固定预算完成前禁止披露中期效果。</p>"
        )
    else:
        analysis = result.get("analysis") or {}
        if not isinstance(analysis, Mapping):
            analysis = {}
        summaries = analysis.get("summaries") or {}
        if not isinstance(summaries, Mapping):
            summaries = {}
        primary = summaries.get(PRIMARY_METRIC) or {}
        if not isinstance(primary, Mapping):
            primary = {}
        rendered_rows = []
        for row in result.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            supports = bool(row.get("supports_aggregate_direction"))
            direction = "支持总体方向" if supports else "反向或持平"
            rendered_rows.append(
                f'<tr class="{"supports" if supports else "opposes"}">'
                f"<td>{html.escape(str(row.get('seed') or '—'))}</td>"
                f"<td>{_study_number(row.get(PRIMARY_METRIC), signed=True)}</td>"
                f"<td>{_study_number(row.get('focus_goal_difference'), signed=True)}</td>"
                f"<td>{_study_number(row.get('wm_expected_changes'), signed=True)}</td>"
                f"<td><span class=\"tag\">{direction}</span></td>"
                f"<td>{_study_evidence_links(row)}</td></tr>"
            )
        rows = "".join(rendered_rows) or (
            '<tr><td colspan="6">固定预算结果缺少逐种子明细。</td></tr>'
        )
        consistency = 100.0 * _finite_or_zero(
            primary.get("direction_consistency")
        )
        body = f"""<h1>固定预算多种子战术研究</h1>
<p><strong>{html.escape(str(analysis.get('primary_decision') or 'unknown'))}</strong></p>
<section><h2>主要指标：焦点侧 xG 差变化</h2><p>均值 {_study_number(primary.get('mean'), signed=True)} · 95% t 区间 [{_study_number(primary.get('ci_low'), signed=True)}, {_study_number(primary.get('ci_high'), signed=True)}] · 方向一致性 {consistency:.1f}%</p></section>
<section><h2>固定 seed 证据下钻</h2><div class="scroll"><table><thead><tr><th>Seed</th><th>xG 差变化</th><th>净胜球变化</th><th>世界模型期望动作变化</th><th>方向</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></div><p class="muted">基线与处理比赛用于逐场复盘；配对页核验共享 seed、干预范围和归因资格。</p></section>
<section><h2>推断边界</h2><p>{html.escape(str(result.get('claim_boundary') or ''))}</p><p>次要指标仅作描述；本报告不授权产品或论文推广。</p></section>"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GFS Tactical Study</title><style>:root{{--bg:#07111e;--panel:#111d2e;--line:#2a3a51;--ink:#edf4ff;--muted:#a8b6c9;--accent:#65e6b4;--warn:#ffc36a}}*{{box-sizing:border-box}}body{{max-width:1100px;margin:auto;padding:32px 20px;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui}}section{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line);white-space:nowrap}}th,.muted{{color:var(--muted)}}a{{color:var(--accent)}}.tag{{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:3px 8px}}tr.opposes .tag{{color:var(--warn);border-color:var(--warn)}}tr.supports .tag{{color:var(--accent);border-color:var(--accent)}}a:focus-visible{{outline:3px solid var(--warn);outline-offset:3px}}</style></head><body>{body}</body></html>"""


def write_tactical_study_html(path: str | Path, result: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_tactical_study_html(result), encoding="utf-8")
    return target


def _study_paths(workspace: Any, study_id: str) -> dict[str, Path]:
    root = Path(workspace.output_root) / "studies" / study_id
    return {
        "root": root,
        "lease": root / "study.lock",
        "protocol": root / "protocol.json",
        "progress": root / "progress.json",
        "result": root / "result.json",
        "dashboard": root / "index.html",
    }


def _validate_completed_result_identity(
    plan: TacticalStudyPlan, result: Mapping[str, Any],
) -> None:
    analysis = result.get("analysis") or {}
    rows = result.get("rows") or []
    if (
        result.get("schema_version") != 1
        or result.get("study_id") != plan.study_id
        or result.get("status") != "complete"
        or result.get("plan") != plan.as_dict()
        or result.get("fixed_pair_budget") != len(plan.seeds)
        or result.get("pairs_completed") != len(plan.seeds)
        or result.get("interim_effects_disclosed") is not False
        or not isinstance(analysis, Mapping)
        or analysis.get("primary_metric") != PRIMARY_METRIC
        or analysis.get("promotion_authorized") is not False
        or not isinstance(rows, list)
    ):
        raise ValueError("completed tactical study result identity is invalid")
    observed_seeds = [
        row.get("seed") if isinstance(row, Mapping) else None for row in rows
    ]
    if observed_seeds != list(plan.seeds):
        raise ValueError("completed tactical study result seed set is invalid")


def _resolve_study_comparison(workspace: Any, value: Any) -> Path:
    workspace_root = Path(workspace.root).resolve()
    matches_root = (Path(workspace.output_root) / "matches").resolve()
    candidate = Path(str(value or ""))
    resolved = (
        candidate if candidate.is_absolute() else workspace_root / candidate
    ).resolve()
    try:
        resolved.relative_to(matches_root)
    except ValueError as exc:
        raise ValueError(
            "tactical study comparison must stay inside Studio matches"
        ) from exc
    if not resolved.name.endswith(".comparison.json") or not resolved.is_file():
        raise ValueError("tactical study comparison artifact is unavailable")
    return resolved


def _resolve_match_report_html(workspace: Any, value: Any) -> Path:
    workspace_root = Path(workspace.root).resolve()
    matches_root = (Path(workspace.output_root) / "matches").resolve()
    report = Path(str(value or ""))
    resolved_report = (
        report if report.is_absolute() else workspace_root / report
    ).resolve()
    try:
        resolved_report.relative_to(matches_root)
    except ValueError as exc:
        raise ValueError("study match report must stay inside Studio matches") from exc
    if resolved_report.suffix.lower() != ".json":
        raise ValueError("study match report must be JSON")
    dashboard = resolved_report.with_suffix(".html")
    if not resolved_report.is_file() or not dashboard.is_file():
        raise ValueError("study match evidence is unavailable")
    return dashboard


def _final_pair_evidence(
    workspace: Any, comparison_path: Path, comparison: Mapping[str, Any],
) -> dict[str, str]:
    workspace_root = Path(workspace.root).resolve()
    expected_dashboard = comparison_path.with_suffix(".html").resolve()
    declared_dashboard = Path(str(comparison.get("dashboard") or ""))
    declared_dashboard = (
        declared_dashboard if declared_dashboard.is_absolute()
        else workspace_root / declared_dashboard
    ).resolve()
    if declared_dashboard != expected_dashboard or not expected_dashboard.is_file():
        raise ValueError("study comparison dashboard identity mismatch")
    reports = comparison.get("reports") or {}
    if not isinstance(reports, Mapping):
        raise ValueError("study comparison reports are invalid")
    baseline = _resolve_match_report_html(workspace, reports.get("baseline"))
    treatment = _resolve_match_report_html(workspace, reports.get("treatment"))
    return {
        "baseline_dashboard": baseline.relative_to(workspace_root).as_posix(),
        "treatment_dashboard": treatment.relative_to(workspace_root).as_posix(),
        "comparison_dashboard": expected_dashboard.relative_to(
            workspace_root
        ).as_posix(),
    }


def execute_tactical_study(
    workspace: Any, plan: TacticalStudyPlan,
) -> dict[str, Any]:
    """Execute or resume a frozen study; never publish effects before completion."""
    if getattr(getattr(workspace, "config", None), "mode", None) != "research":
        raise ValueError("inferential tactical study requires research mode")
    from src.infrastructure import FileLease
    from src.product.workspace import _atomic_json

    paths = _study_paths(workspace, plan.study_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with FileLease(paths["lease"], timeout=5.0):
        frozen = plan.as_dict()
        if paths["protocol"].is_file():
            existing = json.loads(paths["protocol"].read_text(encoding="utf-8"))
            if existing != frozen:
                raise ValueError("study_id already belongs to a different frozen plan")
        else:
            _atomic_json(paths["protocol"], frozen)
        if paths["result"].is_file():
            result = json.loads(paths["result"].read_text(encoding="utf-8"))
            if not isinstance(result, Mapping):
                raise ValueError("completed tactical study result must be an object")
            _validate_completed_result_identity(plan, result)
            if not paths["dashboard"].is_file():
                write_tactical_study_html(paths["dashboard"], result)
            return {
                **result,
                "result_path": str(paths["result"]),
                "dashboard_path": str(paths["dashboard"]),
            }
        progress = {
            "schema_version": 1,
            "study_id": plan.study_id,
            "state": "running",
            "fixed_pair_budget": len(plan.seeds),
            "pairs_completed": 0,
            "completed": [],
            "interim_effects_disclosed": False,
            "analysis": None,
        }
        if paths["progress"].is_file():
            loaded = json.loads(paths["progress"].read_text(encoding="utf-8"))
            if loaded.get("study_id") != plan.study_id:
                raise ValueError("invalid tactical study progress identity")
            progress["completed"] = list(loaded.get("completed") or [])
            progress["pairs_completed"] = len(progress["completed"])
        completed_seeds = [
            item.get("seed") if isinstance(item, Mapping) else None
            for item in progress["completed"]
        ]
        if (
            len(completed_seeds) != len(set(completed_seeds))
            or any(seed not in plan.seeds for seed in completed_seeds)
        ):
            raise ValueError("tactical study progress contains invalid seeds")
        completed_by_seed = {
            int(item["seed"]): item for item in progress["completed"]
        }
        _atomic_json(paths["progress"], progress)
        try:
            for seed in plan.seeds:
                if seed in completed_by_seed:
                    continue
                baseline, treatment = workspace.run_paired_matches(
                    plan.home, plan.away, fast=plan.fast,
                    baseline_plan=plan.baseline_plan(),
                    treatment_plan=plan.treatment_plan(), seed=seed,
                )
                resolved = _resolve_study_comparison(
                    workspace, treatment.get("comparison_path")
                )
                comparison = json.loads(resolved.read_text(encoding="utf-8"))
                validation = analyze_fixed_tactical_study(plan, [comparison])
                if validation["status"] != "incomplete_analysis_withheld":
                    raise RuntimeError("single study pair unexpectedly disclosed analysis")
                entry = {
                    "seed": seed,
                    "baseline_match_id": baseline["match_id"],
                    "treatment_match_id": treatment["match_id"],
                    "comparison": resolved.relative_to(
                        Path(workspace.root).resolve()
                    ).as_posix(),
                }
                progress["completed"].append(entry)
                completed_by_seed[seed] = entry
                progress["pairs_completed"] = len(progress["completed"])
                _atomic_json(paths["progress"], progress)
            comparisons = []
            evidence_by_seed: dict[int, dict[str, str]] = {}
            for seed in plan.seeds:
                relative = completed_by_seed[seed]["comparison"]
                candidate = _resolve_study_comparison(workspace, relative)
                comparison = json.loads(candidate.read_text(encoding="utf-8"))
                comparisons.append(comparison)
                evidence_by_seed[seed] = _final_pair_evidence(
                    workspace, candidate, comparison,
                )
            result = analyze_fixed_tactical_study(plan, comparisons)
            for row in result.get("rows") or []:
                row["evidence"] = evidence_by_seed[int(row["seed"])]
            result["artifacts"] = {
                "protocol": paths["protocol"].relative_to(
                    Path(workspace.root)
                ).as_posix(),
                "progress": paths["progress"].relative_to(
                    Path(workspace.root)
                ).as_posix(),
                "dashboard": paths["dashboard"].relative_to(
                    Path(workspace.root)
                ).as_posix(),
            }
            _atomic_json(paths["result"], result)
            write_tactical_study_html(paths["dashboard"], result)
            progress.update({"state": "complete", "analysis": None})
            _atomic_json(paths["progress"], progress)
            return {
                **result,
                "result_path": str(paths["result"]),
                "dashboard_path": str(paths["dashboard"]),
            }
        except BaseException as exc:
            progress.update({
                "state": "failed", "error_type": type(exc).__name__,
                "analysis": None, "interim_effects_disclosed": False,
            })
            _atomic_json(paths["progress"], progress)
            raise

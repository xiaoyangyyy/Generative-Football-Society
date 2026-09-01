import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path
from urllib.request import urlopen
from wsgiref.util import setup_testing_defaults

import pytest

from src.cli import build_parser, cmd_studio_web
from src.product.web import ProductWebApp, _is_loopback_host, create_product_web_server
from src.product.web_security import WebAccessPolicy
from src.product.tasks import BackgroundMatchWorker
from src.product.tactical_study import TacticalStudyPlan
from scripts.build_excellence_evidence_kit import PROTOCOLS, SECRET_PATTERN


ROOT = Path(__file__).resolve().parents[1]


def _copy_evidence_kit_inputs(target):
    for relative in PROTOCOLS.values():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    guide = target / "docs/EXCELLENCE_EVIDENCE_KIT.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_bytes((ROOT / "docs/EXCELLENCE_EVIDENCE_KIT.md").read_bytes())


def _request(
    app, method="GET", path="/", payload=None, *, csrf=None, host=None,
    idempotency_key=None, forwarded_proto=None, cookie=None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ = {}
    setup_testing_defaults(environ)
    environ.update({
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    })
    if payload is not None:
        environ["CONTENT_TYPE"] = "application/json"
    if csrf is not None:
        environ["HTTP_X_GFS_CSRF"] = csrf
    if host is not None:
        environ["HTTP_HOST"] = host
    if idempotency_key is not None:
        environ["HTTP_IDEMPOTENCY_KEY"] = idempotency_key
    if forwarded_proto is not None:
        environ["HTTP_X_FORWARDED_PROTO"] = forwarded_proto
    if cookie is not None:
        environ["HTTP_COOKIE"] = cookie
    observed = {}

    def start_response(status, headers):
        observed["status"] = status
        observed["headers"] = dict(headers)

    response = b"".join(app(environ, start_response))
    observed["body"] = response
    if observed["headers"]["Content-Type"].startswith("application/json"):
        observed["json"] = json.loads(response)
    return observed


def test_root_is_accessible_and_hardened(tmp_path):
    response = _request(ProductWebApp(tmp_path))
    document = response["body"].decode("utf-8")
    assert response["status"].startswith("200")
    assert "Content-Security-Policy" in response["headers"]
    assert response["headers"]["X-Frame-Options"] == "DENY"
    capabilities = _request(
        ProductWebApp(tmp_path), path="/api/v1/studio"
    )["json"]["match_capabilities"]
    assert capabilities["simulation_clock"] == {
        "contract": "authoritative_tick_v2",
        "legacy_contract": "legacy_affective_offset_v1",
        "studio_requires_authoritative": True,
    }
    assert capabilities["paired_match"] == {
        "modes": ["research", "cognitive"],
        "seed": "explicit_shared_seed",
        "intervention": "exactly_one_tactical_side",
        "score_path": "physics_official",
        "cognitive_claim_boundary": "descriptive_only",
    }
    assert capabilities["world_model_fork"] == {
        "modes": ["research"],
        "seed": "explicit_shared_seed",
        "intervention": "predict_only_to_action_policy",
            "branch_time_seconds": {"min": 0, "max": 5400, "default": 2700},
            "branch_execution": "identity_bound_deterministic_replay",
            "resume_capability": "deterministic_replay_only",
            "clock_contract": "authoritative_tick_v2",
            "result_contract": "evidence_graded_counterfactual_future_v1",
            "fixed_controls": [
                "fixture", "tactics", "fast_configuration", "checkpoint",
                "clock_contract",
            ],
            "score_path": "physics_official",
            "evidence_chain": [
                "isolated_policy_assignment",
                "identical_pre_intervention_branch_anchor",
                "realized_action_change",
                "direct_runtime_identity",
                "descriptive_downstream_windows",
                "unified_counterfactual_future_summary",
            ],
            "downstream_causal_attribution": False,
            "claim_boundary": "single_fixture_seed_simulator_contrast_only",
        }
    assert '<a class="skip" href="#main">' in document
    assert 'id="main"' in document
    assert 'aria-live="polite"' in document
    assert 'id="workflow" class="status" role="status"' in document
    assert 'id="manager-world-model-advice"' in document
    assert 'id="request-manager-advice"' in document
    assert 'id="adopt-manager-advice"' in document
    assert 'id="manager-future-set"' in document
    assert 'id="manager-intervention-workspace"' in document
    assert 'id="request-manager-future-set"' in document
    assert "/api/v1/seasons/world-model-future-set" in document
    assert "/api/v1/seasons/world-model-future-review" in document
    assert "function renderManagerFutureSets(" in document
    assert "function renderManagerInterventionWorkspace(" in document
    assert "function renderManagerWorldNavigator(" in document
    assert "已完成的足球世界章节" in document
    assert "章节完整不代表赛果改善" in document
    assert "经理反事实干预五步流程" in document
    assert "不排名时点、不预测比分" in document
    assert "从已验证进度恢复未来生成" in document
    assert "function requestManagerFutureExperiment(" in document
    assert "function reviewManagerFutureEvidence(" in document
    assert "renderManagerDecisionLedgerWithoutFutureReviews" in document
    assert "renderManagerDecisionLedgerWithoutFutureScenarioEvidence" in document
    assert "renderManagerFutureSetsWithoutScenarioEvidence" in document
    assert "function appendFutureMechanismExamples(" in document
    assert "renderManagerFutureSetsWithoutMechanismExamples" in document
    assert "renderManagerDecisionLedgerWithoutMechanismExamples" in document
    assert "已封存的具体动作采用链" in document
    assert "scenario.scenario_identity" in document
    assert "已封存的分叉机制链" in document
    assert "row.world_model_future_reviews" in document
    assert "keep_after_review" in document
    assert "revise_after_review" in document
    assert document.count("const renderLibraryWithoutForkSets=") == 1
    assert "前缀锚点" in document
    assert "确定性重放（非进程快照）" in document
    assert "反事实未来：" in document
    assert "不授予赛果因果" in document
    assert "/api/v1/seasons/decision-advice" in document
    assert "payload.advice_adoption=" in document
    assert "建议、经理选择和赛果分别取证" in document
    assert 'id="workflow-action" type="button" hidden' in document
    assert 'id="workspace-nav" class="workspace-nav"' in document
    assert document.count('data-workspace-view=') == 4
    assert 'data-workspace-area="career"' in document
    assert document.count('data-workspace-area="lab"') == 4
    assert document.count('data-workspace-area="evidence"') == 3
    assert document.count('data-workspace-area="operations"') == 2
    assert "function applyWorkspaceArea(" in document
    assert "function workflowWorkspaceArea(" in document
    assert "function renderWorkspaceAreas(" in document
    assert "sessionStorage.getItem('gfs-workspace-area')" in document
    assert "sessionStorage.setItem('gfs-workspace-area',resolved)" in document
    assert ".workspace-area-hidden { display:none!important }" in document
    assert "function renderUnifiedWorkflow(data)" in document
    assert "start_season:seasonForm" in document
    assert "freeze_player_promises:playerPromiseForm" in document
    assert "submit_manager_decision:managerDecisionForm" in document
    assert "advance_season_matchday:playMatchday" in document
    assert "workflowAction._target=targets[action.id]||null" in document
    assert "workflowAction._area=workflowAreaByAction[action]" in document
    assert "applyWorkspaceArea(workflowAction._area,{remember:true,configured:true})" in document
    assert "if(focusable)focusable.focus()" in document
    assert "prefers-reduced-motion" in document
    assert 'id="logout-button" type="button" hidden' in document
    assert 'aria-labelledby="recovery-title"' in document
    assert 'id="backup-list"' in document and 'role="list"' in document
    assert 'id="restore-form" hidden' in document
    assert 'aria-labelledby="release-title"' in document
    assert 'id="release-summary"' in document
    assert 'id="release-gates"' in document and 'role="list"' in document
    assert "renderRelease(data)" in document
    assert 'aria-labelledby="action-adoption-title"' in document
    assert 'id="action-adoption-metrics"' in document
    assert 'id="manager-advisor-protocol-evidence"' in document
    assert "renderActionAdoption(" in document
    assert "renderActionAdoptionWithoutActionSignals" in document
    assert "adoption.action_signal_breakdown||{}" in document
    assert "mean_applied_authority" in document
    assert "renderActionAdoptionWithoutCurrentCodeEvidence" in document
    assert "mechanism.result_identity_verified" in document
    assert "outcome.result_identity_verified" in document
    assert "renderActionAdoptionWithoutManagerProtocol" in document
    assert "manager_advisor_adoption" in document
    assert 'name="experience"' in document
    assert 'id="tactical-options"' in document
    assert 'name="reuse_last_seed" type="checkbox" disabled' in document
    assert "reuseSeed.disabled=!lab" in document
    assert "configureMatchPlan(data.match_capabilities" in document
    assert "战术实验使用物理比分" in document
    assert "打开双世界/战术配对比较" in document
    assert "分叉传播与对比" in document
    assert "下游窗口仅描述" in document
    assert "s?.last_match?.comparison_url" in document
    assert "action_adoption_mechanism" in document
    assert "promotion_authorized" in document
    assert "单场运行观测不等于机制证明" in document
    assert 'id="study-panel"' in document
    assert 'id="study-form"' in document
    assert 'name="pair_budget"' in document
    assert "configureTacticalStudy(data.match_capabilities" in document
    assert "中期效应保持隐藏" in document
    assert "'/api/v1/tactical-studies'" in document
    assert "task.result?.study_url" in document
    assert 'id="season-panel"' in document
    assert 'id="manager-product-journey"' in document
    assert 'class="manager-product-journey"' in document
    assert "function renderManagerProductJourney(" in document
    assert "renderUnifiedWorkflowWithoutManagerJourney" in document
    assert "item.dataset.status=status" in document
    assert "item.setAttribute('aria-current','step')" in document
    assert 'id="matchday-command-center"' in document
    assert 'id="manager-world-navigator"' in document
    assert 'id="manager-world-navigator-action"' in document
    assert 'id="manager-world-navigator-secondary"' in document
    assert 'id="manager-world-navigator-gaps"' in document
    assert 'id="manager-world-influence-path"' in document
    assert 'id="manager-world-gap-diagnostics"' in document
    assert 'id="manager-world-gap-list"' in document
    assert 'id="manager-world-action-adoption-metrics"' in document
    assert 'id="manager-world-action-adoption-diagnostics"' in document
    assert 'id="manager-world-action-adoption-states"' in document
    assert 'id="manager-world-action-propagation-boundary"' in document
    assert 'id="manager-world-trajectory"' in document
    assert 'id="manager-world-trajectory-list"' in document
    assert 'id="manager-world-trajectory-boundary"' in document
    assert 'id="manager-world-future-continuity-boundary"' in document
    assert "managerWorldInfluencePath.replaceChildren()" in document
    assert "managerWorldGapList.replaceChildren()" in document
    assert "summary.world_model_influence_path||{}" in document
    assert "reviewed_future_scenario_evidence" in document
    assert "reviewed_future_action_divergence" in document
    assert "reviewed_future_local_attribution" in document
    assert "reviewed_future_timing_sensitivity" in document
    assert "summary.continuity_gap_counts||[]" in document
    assert "summary.world_model_action_adoption_ledger" in document
    assert "function renderManagerWorldActionAdoptionLedger(" in document
    assert "function managerWorldPropagationText(" in document
    assert "renderManagerWorldActionAdoptionLedgerWithoutWorldPropagation" in document
    assert "ledger.state_counts||[]" in document
    assert "row.descriptive_world_after" in document
    assert "chapter.descriptive_world_after" in document
    assert "function renderManagerWorldTrajectory(season)" in document
    assert "function renderManagerWorldReviewedFutureContinuity(season)" in document
    assert "appendManagerWorldEvolutionThreadWithoutReviewedFutures" in document
    assert "function appendReviewedScenarioArchive(" in document
    assert "appendManagerWorldEvolutionThreadWithoutScenarioArchive" in document
    assert "renderManagerWorldNavigatorWithoutScenarioArchive" in document
    assert "reviewed_scenario_archives" in document
    assert "source_scenario_identity" in document
    assert "archive_identity" in document
    assert "不排名、不与观察比分匹配" in document
    assert "function appendReviewWorldContinuity(" in document
    assert "appendManagerWorldEvolutionThreadWithoutReviewWorldCertificate" in document
    assert "renderManagerWorldNavigatorWithoutReviewWorldCertificate" in document
    assert "renderManagerWorldNavigatorWithoutReviewWorldInfluence" in document
    assert "complete_reviewed_world_model_chains" in document
    assert "不会把赛前模拟分叉中的机会逐条匹配" in document
    assert "future.action_divergence_scenarios" in document
    assert "stage.local_attribution_scenarios" in document
    assert "trajectory_point_identity" not in document
    assert "row.fixture_id===fixtureId&&row.chapter_identity===expectedChapterIdentity" in document
    assert "turning_point_inference_authorized" not in document
    assert "按采纳状态分层只描述共现" in document
    assert "不授权状态间效果比较" in document
    assert "diagnosticRows=gapRows.concat(stateRows,trajectoryRows)" in document
    assert "至少一处局部动作改变" in document
    assert "各项是独立证据覆盖，不是递减漏斗" in document
    assert "证据断点章节" in document
    assert "未来比赛应在最后一次编辑后重新复核并冻结" in document
    assert "检查比赛是否启用世界模型并保留动作证据身份" in document
    assert "所有已完成章节均无已声明连续性断点。" in document
    assert "function visibleManagerLedgerEntries(" in document
    assert "node.dataset.fixtureId=String(row.fixture_id||'')" in document
    assert "node.dataset.chapterComplete=" in document
    assert "打开完整世界章节" in document
    assert "function managerNavigationTarget(" in document
    assert "function activateManagerNavigationTarget(" in document
    assert "function openManagerWorldChapter(" in document
    assert "function navigateManagerWorldChapter(" in document
    assert "function restoreManagerWorldChapterLocation(" in document
    assert "row.latest_fixture_id===fixtureId&&row.latest_chapter_identity===expectedChapterIdentity" in document
    assert "诊断绑定的世界章节已过期或不可用。" in document
    assert "打开最近受影响章节 · 第 " in document
    assert "open.dataset.chapterIdentity=chapterIdentity" in document
    assert "selectedRef?.seasonId===season?.season_id" in document
    assert "history.pushState({seasonId,worldChapter:fixtureId}" in document
    assert "window.addEventListener('popstate'" in document
    assert "renderManagerDecisionLedger(currentSeason);renderManagerWorldNavigator(currentSeason)" in document
    assert "navigator.secondary_actions||[]" in document
    assert "当前会话连续性断点：" in document
    assert "世界章节链接格式无效" in document
    assert "该世界章节链接属于另一个赛季" in document
    assert "!target.closest('[hidden]')" in document
    assert "for(const detail of target.querySelectorAll('details'))" in document
    assert "当前导航目标不可用，请刷新赛季状态。" in document
    assert 'id="matchday-journey"' in document
    assert 'id="matchday-briefing"' in document
    assert 'id="matchday-intelligence"' in document
    assert 'id="matchday-debrief"' in document
    assert 'id="matchday-attribution"' in document
    assert "function renderMatchdayCommand" in document
    assert "fixture.opponent_preparation" in document
    assert "对手准备审计" in document
    assert "不是学习结果或已证明的克制关系" in document
    assert "prep.selected_tactic" in document
    assert "function renderManagerIntelligence" in document
    assert "俱乐部长期支持" in document
    assert "effects.club_fatigue_load_factor" in document
    assert "direct_persisted_state" in document
    assert "单场比分仅作描述" in document
    assert "rotation_tradeoff" in document
    assert 'id="manager-decision-ledger"' in document
    assert 'id="manager-advisor-evidence-summary"' in document
    assert "function renderManagerDecisionLedger(season)" in document
    assert "renderManagerDecisionLedgerWithoutAdvisorEvidence" in document
    assert "evidence.adopted_recommendation" in document
    assert "evidence.reviewed_then_selected" in document
    assert "单场赛果不证明决策效果" in document
    assert "世界状态：比赛后疲劳" in document
    assert "伤停为模拟状态" in document
    assert "renderManagerDecisionPreviewWithoutWorldModelComparison" in document
    assert "comparison.recommended_tactic" in document
    assert "comparison.selected_tactic" in document
    assert "deltas.risk_adjusted_value" in document
    assert "authority.level==='exploratory_only'" in document
    assert "adoptManagerAdvice.textContent=" in document
    assert "command.decision_required" in document
    assert "safeFixture?.dashboard_url" in document
    assert 'id="season-form"' in document
    assert 'name="manager_objective"' in document
    assert 'name="manager_points_target"' in document
    assert 'id="season-commitment-fieldset"' in document
    assert 'name="commitment_tactic_policy"' in document
    assert 'name="commitment_rotation_policy"' in document
    assert "function renderSeasonCommitments" in document
    assert "payload.plan.manager_commitments" in document
    assert 'id="player-promise-form"' in document
    assert "function renderPlayerPromises" in document
    assert "function syncPlayerPromiseForm" in document
    assert "'/api/v1/seasons/player-promises'" in document
    assert 'name="resource_recovery"' in document
    assert 'name="resource_medical"' in document
    assert 'name="resource_sports_science"' in document
    assert 'id="club-resource-summary"' in document
    assert 'id="recruitment-fieldset"' in document
    assert 'id="recruitment-summary"' in document
    assert "function loadRecruitmentMarket" in document
    assert "function recruitmentPayload" in document
    assert "payload.plan.manager_recruitment=recruitmentPayload()" in document
    assert "'/api/v1/recruitment-markets/'" in document
    assert "function renderClubFinance" in document
    assert "data.studio?.club_finance" in document
    assert "currentRecruitmentMarket.available_budget" in document
    assert "俱乐部财政与工资" in document
    assert "财政为有界游戏积分账本" in document
    assert "function renderLeagueEcosystem" in document
    assert "data.studio?.league_ecosystem" in document
    assert "renderPlayerDevelopment(data.studio?.player_development)" in document
    assert "球员成长与生涯" in document
    assert "renderPlayerLifecycle(data.studio?.player_lifecycle)" in document
    assert "/api/v1/lifecycle-previews/" in document
    assert "合同、退役与青训" in document
    assert "renderGlobalPlayerMarket(data.studio?.player_market)" in document
    assert "/api/v1/free-agent-markets/" in document
    assert "/api/v1/scouting-reports" in document
    assert "function scoutSelectedFreeAgent" in document
    assert "renderScouting(data.studio?.scouting)" in document
    assert "function renderScoutingOutcomes" in document
    assert "renderScoutingOutcomes(data.studio?.scouting_outcomes)" in document
    assert "签约结果与球探复盘" in document
    assert 'id="sporting-director-fieldset"' in document
    assert "'/api/v1/sporting-plans/'" in document
    assert "function sportingDirectivePayload" in document
    assert "review_sporting_plan" in document
    assert "function renderSportingReview" in document
    assert "data.studio?.sporting_reviews" in document
    assert "populateSportingPlanWithoutContinuity" in document
    assert "previous_strategy_review" in document
    assert 'id="club-situation-fieldset"' in document
    assert "function renderClubSituation" in document
    assert "payload.decision.club_event_choice" in document
    assert "payload.plan.manager_sporting_directive=sportingDirectivePayload()" in document
    assert "同一计划同时约束续约、普通招募与自由签约" in document
    assert "function renderSportingDirection" in document
    assert "renderSportingDirection(data.studio?.season)" in document
    assert "冻结体育总监计划" in document
    season_form = document.index('<form id="season-form"')
    season_form_end = document.index("</form>", season_form)
    for control_id in (
        'id="sporting-director-fieldset"', 'id="lifecycle-fieldset"',
        'id="recruitment-fieldset"', 'id="free-agent-fieldset"',
    ):
        assert season_form < document.index(control_id) < season_form_end
    assert "全局自由球员市场" in document
    assert "动态联赛生态" in document
    assert "查看各俱乐部决策与证据" in document
    assert "AI 俱乐部使用可重放的游戏策略" in document
    assert "function renderClubStrategyBriefing" in document
    assert "function renderClubStrategies" in document
    assert "season?.matchday_command_center?.club_strategy" in document
    assert "data.studio?.season?.club_strategies" in document
    assert "赛季身份与对手打法" in document
    assert "俱乐部赛季身份" in document
    assert "function syncClubResources" in document
    assert "manager_resources:manager?resources:null" in document
    assert "当场实力 +0" in document
    assert 'id="manager-profile"' in document
    assert 'id="manager-career-contract"' in document
    assert 'id="season-history"' in document
    assert 'id="season-history-summary"' in document
    assert "function renderManagerCareer" in document
    assert "function renderCareerContract" in document
    assert "s?.manager_career" in document
    assert "最近 ${career.history_scope?.retained_seasons" in document
    assert "contract.same_club_allowed" in document
    assert "seasonForm.dataset.startNext" in document
    assert "s?.season_history||[]" in document
    assert "s?.season_history_summary||{}" in document
    assert 'id="season-standings"' in document
    assert 'id="season-fixtures"' in document
    assert 'id="manager-manual-lineup"' in document
    assert 'id="manager-squad"' in document
    assert "function renderManagerSquad" in document
    assert "manualLineupPayload" in document
    assert "renderSeason(s?.season" in document
    assert "'/api/v1/seasons/next-matchday'" in document
    assert 'id="pair-panel"' in document
    assert 'id="pair-form"' in document
    assert "configurePairedMatch(data.match_capabilities" in document
    assert "configureWorldModelFork(data.match_capabilities" in document
    assert 'id="fork-panel"' in document
    assert "/api/v1/world-model-forks" in document
    assert "/api/v1/world-model-fork-sets" in document
    assert "forkBranchInput.name='branch_minutes'" in document
    assert "task.kind==='world_model_fork_set'" in document
    assert "predict_only → action_policy" in document
    assert "'/api/v1/paired-matches'" in document
    assert "task.kind==='paired_match'" in document
    assert "配对对决完成，三层复盘已开放" in document
    assert "认知模式含不受共享 seed 完全控制的供应商输出" in document
    assert "安全恢复配对事务" in document
    assert "resumeInterruptedTask(item.task_id" in document
    assert "studio_pair_transaction_resume" in document
    assert 'id="library-title"' in document
    assert 'id="library-filter"' in document
    assert 'id="library-list"' in document
    assert "renderLibrary(data.evidence_library)" in document
    assert ".innerHTML" not in document
    assert "meta.textContent=" in document
    assert "recommended_gate_id" in document
    assert "dataset.actionState" in document
    assert "evidence-kit-download" in document
    assert "/api/v1/excellence/evidence-kit.zip" in document
    assert "证据包" in document
    assert 'name="confirmation"' in document and 'name="replace"' in document
    assert 'type="file"' not in document
    assert "innerHTML" not in document


def test_health_is_liveness_only_and_never_calls_provider(tmp_path):
    response = _request(ProductWebApp(tmp_path), path="/healthz")
    assert response["json"] == {
        "schema_version": 1,
        "status": "ok",
        "service": "gfs-product-web",
        "external_calls_made": False,
        "background_worker_alive": None,
    }


def test_operations_are_aggregated_and_paths_are_privacy_normalized(tmp_path):
    app = ProductWebApp(tmp_path)
    _request(app, path="/api/v1/tasks/private-user-component")
    operations = _request(app, path="/api/v1/operations")
    assert operations["status"].startswith("200")
    metrics = operations["json"]
    assert metrics["requests"]["routes"]["/api/v1/tasks/{task_id}"] == 1
    assert metrics["privacy"]["raw_events_exposed"] is False
    assert "private-user-component" not in json.dumps(metrics)
    assert "events" not in metrics


def test_evidence_kit_download_is_deterministic_secret_free_and_memory_only(tmp_path):
    _copy_evidence_kit_inputs(tmp_path)
    app = ProductWebApp(tmp_path)
    first = _request(app, path="/api/v1/excellence/evidence-kit.zip")
    second = _request(app, path="/api/v1/excellence/evidence-kit.zip")
    assert first["status"].startswith("200")
    assert first["headers"]["Content-Type"] == "application/zip"
    assert first["headers"]["X-GFS-Template-Only"] == "true"
    assert first["headers"]["Content-Disposition"] == (
        'attachment; filename="gfs-excellence-evidence-kit-v1.zip"'
    )
    assert first["body"] == second["body"]
    assert first["headers"]["X-GFS-Artifact-SHA256"] == hashlib.sha256(
        first["body"]
    ).hexdigest()
    with zipfile.ZipFile(io.BytesIO(first["body"])) as archive:
        contents = b"".join(archive.read(name) for name in archive.namelist())
        assert "manifest.json" in archive.namelist()
    assert SECRET_PATTERN.search(contents) is None
    assert not (tmp_path / "build/evidence-kits").exists()
    metrics = _request(app, path="/api/v1/operations")["json"]
    assert metrics["requests"]["routes"][
        "/api/v1/excellence/evidence-kit.zip"
    ] == 2
    wrong_method = _request(
        app, method="POST", path="/api/v1/excellence/evidence-kit.zip",
    )
    assert wrong_method["status"].startswith("405")


def test_remote_evidence_kit_download_requires_authenticated_https_session(tmp_path):
    _copy_evidence_kit_inputs(tmp_path)
    token = "evidence-kit-test-token-" + "x" * 32
    app = ProductWebApp(tmp_path, access_policy=WebAccessPolicy(
        remote=True, access_token=token, allowed_hosts=("studio.example",),
    ))
    unauthorized = _request(
        app, path="/api/v1/excellence/evidence-kit.zip",
        host="studio.example", forwarded_proto="https",
    )
    assert unauthorized["status"].startswith("401")
    login = _request(
        app, "POST", "/api/v1/login", {"access_token": token},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    cookie = login["headers"]["Set-Cookie"].split(";", 1)[0]
    downloaded = _request(
        app, path="/api/v1/excellence/evidence-kit.zip", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert downloaded["status"].startswith("200")
    assert token.encode() not in downloaded["body"]


def test_telemetry_failure_does_not_break_liveness(tmp_path, monkeypatch):
    app = ProductWebApp(tmp_path)
    monkeypatch.setattr(app.telemetry, "record", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    response = _request(app, path="/healthz")
    assert response["status"].startswith("200")
    assert app.telemetry.write_failures == 1


def test_non_loopback_host_header_is_rejected_against_dns_rebinding(tmp_path):
    response = _request(ProductWebApp(tmp_path), path="/api/v1/studio", host="attacker.example")
    assert response["status"].startswith("400")
    assert response["json"]["error"]["code"] == "invalid_host"


def test_web_workflow_requires_csrf_then_creates_and_reads_studio(tmp_path):
    app = ProductWebApp(tmp_path)
    payload = {"name": "Beta Lab", "mode": "stable", "seed": 17}
    rejected = _request(app, "POST", "/api/v1/studio", payload)
    assert rejected["status"].startswith("403")
    assert rejected["json"]["error"]["code"] == "csrf_rejected"

    created = _request(
        app, "POST", "/api/v1/studio", payload, csrf=app.csrf_token,
    )
    assert created["status"].startswith("201")
    assert created["json"]["studio"]["name"] == "Beta Lab"
    assert created["json"]["studio"]["workflow"]["state"] == "blocked"

    status = _request(app, path="/api/v1/studio")
    assert status["json"]["configured"]
    assert status["json"]["studio"]["seed"] == 17
    assert status["headers"]["X-GFS-CSRF-Token"] == app.csrf_token

    duplicate = _request(
        app, "POST", "/api/v1/studio", payload, csrf=app.csrf_token,
    )
    assert duplicate["status"].startswith("409")
    assert duplicate["json"]["error"]["code"] == "studio_exists"


def test_match_route_validates_input_before_domain_execution(tmp_path):
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "brazil", "fast": True,
    }, csrf=app.csrf_token)
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "same_team"


def test_season_routes_create_and_idempotently_queue_next_matchday(
    tmp_path, monkeypatch,
):
    class Workspace:
        def create_season(self, plan):
            assert plan.teams == ("A", "B", "C", "D")
            return {"season_id": "season-0001", "next_matchday": 1, "revision": 0}

        def season_status(self):
            return {"season_id": "season-0001", "next_matchday": 1, "revision": 0}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    created = _request(app, "POST", "/api/v1/seasons", {
        "plan": {"teams": ["A", "B", "C", "D"], "legs": 1, "fast": True},
    }, csrf=app.csrf_token)
    assert created["status"].startswith("201")
    first = _request(
        app, "POST", "/api/v1/seasons/next-matchday", {},
        csrf=app.csrf_token,
    )
    repeated = _request(
        app, "POST", "/api/v1/seasons/next-matchday", {},
        csrf=app.csrf_token,
    )
    assert first["status"].startswith("202")
    assert repeated["status"].startswith("200")
    assert first["json"]["task"]["task_id"] == repeated["json"]["task"]["task_id"]
    assert first["json"]["task"]["kind"] == "season_matchday"


def test_season_route_explicitly_starts_next_completed_season_with_objective(
    tmp_path, monkeypatch,
):
    observed = {}

    class Workspace:
        def create_season(self, plan, *, replace=False):
            observed["plan"] = plan
            observed["replace"] = replace
            return {"season_id": "season-0002", "next_matchday": 1}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/seasons", {
        "start_next": True,
        "plan": {
            "teams": ["A", "B", "C", "D"], "manager_team": "A",
            "manager_objective": "points_target", "manager_points_target": 7,
            "manager_resources": {
                "recovery": 4, "medical": 1, "sports_science": 1,
            },
            "manager_recruitment": {
                "market_id": "market-0002-deadbeefdead",
                "moves": [{
                    "candidate_id": "candidate-1",
                    "outgoing_player_id": "player-1",
                }],
            },
            "manager_sporting_directive": {
                "schema_version": 1, "planning_id": "a" * 64,
                "philosophy": "win_now", "risk_level": "low",
                "priority_roles": ["CM", "ST"],
            },
            "manager_commitments": {
                "schema_version": 1, "tactic_policy": "adaptive",
                "rotation_policy": "trust_core",
            },
        },
    }, csrf=app.csrf_token)

    assert response["status"].startswith("201")
    assert observed["replace"] is True
    assert observed["plan"].manager_objective == "points_target"
    assert observed["plan"].manager_points_target == 7
    assert observed["plan"].manager_resources.recovery == 4
    assert observed["plan"].manager_resources.medical == 1
    assert observed["plan"].manager_resources.sports_science == 1
    assert observed["plan"].manager_recruitment.market_id == (
        "market-0002-deadbeefdead"
    )
    assert observed["plan"].manager_recruitment.moves[0].candidate_id == (
        "candidate-1"
    )
    assert observed["plan"].manager_sporting_directive.philosophy == "win_now"
    assert observed["plan"].manager_sporting_directive.risk_level == "low"
    assert observed["plan"].manager_sporting_directive.priority_roles == (
        "CM", "ST",
    )
    assert observed["plan"].manager_commitments.tactic_policy == "adaptive"
    assert observed["plan"].manager_commitments.rotation_policy == "trust_core"

    invalid = _request(app, "POST", "/api/v1/seasons", {
        "start_next": "yes", "plan": {"teams": ["A", "B", "C", "D"]},
    }, csrf=app.csrf_token)
    assert invalid["status"].startswith("422")
    assert invalid["json"]["error"]["code"] == "invalid_season"


def test_recruitment_market_route_is_read_only_and_decodes_team(tmp_path, monkeypatch):
    class Workspace:
        def recruitment_market(self, team):
            assert team == "A Team"
            return {
                "market_id": "market-0002-a", "team": team,
                "budget": 8, "candidates": [], "outgoing_players": [],
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(
        app, path="/api/v1/recruitment-markets/A%20Team",
    )

    assert response["status"].startswith("200")
    assert response["json"]["market"]["team"] == "A Team"
    assert response["json"]["market"]["budget"] == 8


def test_sporting_plan_route_is_read_only_and_decodes_team(tmp_path, monkeypatch):
    class Workspace:
        def sporting_plan(self, team):
            assert team == "A Team"
            return {
                "planning_id": "a" * 64, "team": team,
                "source_season_id": "season-0001",
                "target_season_id": "season-0002",
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    response = _request(
        ProductWebApp(tmp_path), path="/api/v1/sporting-plans/A%20Team",
    )

    assert response["status"].startswith("200")
    assert response["json"]["plan"]["team"] == "A Team"
    assert response["json"]["plan"]["planning_id"] == "a" * 64


def test_season_route_exposes_board_contract_conflict(tmp_path, monkeypatch):
    class Workspace:
        def create_season(self, _plan, *, replace=False):
            assert replace is True
            raise ValueError("manager was dismissed and must change club")

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/seasons", {
        "start_next": True,
        "plan": {
            "teams": ["A", "B", "C", "D"], "manager_team": "A",
            "manager_objective": "top_half",
        },
    }, csrf=app.csrf_token)

    assert response["status"].startswith("409")
    assert response["json"]["error"] == {
        "code": "season_transition_invalid",
        "message": "manager was dismissed and must change club",
    }


def test_manager_decision_route_validates_and_persists_frozen_choice(
    tmp_path, monkeypatch,
):
    observed = {}

    class Workspace:
        def set_manager_decision(
            self, decision, *, fixture_id=None, expected_revision=None,
            advice_adoption=None,
        ):
            observed.update(
                decision=decision, fixture_id=fixture_id,
                expected_revision=expected_revision,
                advice_adoption=advice_adoption,
            )
            return {
                "season_id": "season-0001", "revision": 4,
                "next_matchday": 1,
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    accepted = _request(app, "POST", "/api/v1/seasons/decision", {
        "fixture_id": "md01-fx01",
        "expected_revision": 3,
        "advice_adoption": {
            "schema_version": 1,
            "advice_identity": "a" * 64,
            "intent": "reviewed_then_selected",
        },
        "decision": {
            "team": "Brazil", "tactic": "gegenpress", "rotation": "rotate",
            "club_event_choice": {
                "schema_version": 1,
                "event_id": "season-0001:md01-fx01:form_response",
                "event_identity": "b" * 64,
                "choice_id": "reset_approach",
            },
        },
    }, csrf=app.csrf_token)
    assert accepted["status"].startswith("200")
    assert accepted["json"]["season"]["revision"] == 4
    assert observed["fixture_id"] == "md01-fx01"
    assert observed["expected_revision"] == 3
    assert observed["advice_adoption"]["advice_identity"] == "a" * 64
    assert observed["decision"].as_dict()["rotation"] == "rotate"
    assert observed["decision"].club_event_choice.choice_id == "reset_approach"

    rejected = _request(app, "POST", "/api/v1/seasons/decision", {
        "fixture_id": "md01-fx01",
        "decision": {
            "team": "Brazil", "tactic": "imaginary", "rotation": "rotate",
        },
    }, csrf=app.csrf_token)
    assert rejected["status"].startswith("422")
    assert rejected["json"]["error"]["code"] == "invalid_manager_decision"


def test_manager_decision_preview_route_is_csrf_protected_and_read_only(
    tmp_path, monkeypatch,
):
    observed = {}

    class Workspace:
        def preview_manager_decision(self, decision, *, fixture_id=None):
            observed.update(decision=decision, fixture_id=fixture_id)
            return {
                "schema_version": 1, "base_revision": 7,
                "normalized_decision": decision.as_dict(),
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {
        "fixture_id": "md01-fx01",
        "decision": {
            "team": "Brazil", "tactic": "balanced", "rotation": "strongest",
        },
    }
    rejected = _request(
        app, "POST", "/api/v1/seasons/decision-preview", payload,
    )
    assert rejected["status"].startswith("403")

    accepted = _request(
        app, "POST", "/api/v1/seasons/decision-preview", payload,
        csrf=app.csrf_token,
    )
    assert accepted["status"].startswith("200")
    assert accepted["json"]["preview"]["base_revision"] == 7
    assert observed["fixture_id"] == "md01-fx01"
    assert observed["decision"].tactic == "balanced"


def test_manager_decision_advice_route_is_csrf_protected_and_strict(
    tmp_path, monkeypatch,
):
    observed = {}

    class Workspace:
        def request_manager_decision_advice(self, *, fixture_id=None):
            observed["fixture_id"] = fixture_id
            return {
                "schema_version": 1, "available": True,
                "fixture_id": fixture_id, "advice_identity": "a" * 64,
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {"fixture_id": "md01-fx01"}
    rejected = _request(
        app, "POST", "/api/v1/seasons/decision-advice", payload,
    )
    assert rejected["status"].startswith("403")

    accepted = _request(
        app, "POST", "/api/v1/seasons/decision-advice", payload,
        csrf=app.csrf_token,
    )
    assert accepted["status"].startswith("200")
    assert accepted["json"]["advice"]["advice_identity"] == "a" * 64
    assert observed["fixture_id"] == "md01-fx01"

    extra = _request(
        app, "POST", "/api/v1/seasons/decision-advice",
        {**payload, "client_score": 1.0}, csrf=app.csrf_token,
    )
    assert extra["status"].startswith("422")
    assert extra["json"]["error"]["code"] == "manager_advice_unavailable"


def test_player_promise_route_parses_and_freezes_named_roles(tmp_path, monkeypatch):
    observed = {}

    class Workspace:
        def set_player_role_promises(self, plan):
            observed["plan"] = plan
            return {
                "season_id": "season-0001",
                "player_role_promises": {"control": "manager"},
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/seasons/player-promises", {
        "schema_version": 1,
        "promises": [
            {"player_id": "p-core", "role": "core"},
            {"player_id": "p-young", "role": "development"},
        ],
    }, csrf=app.csrf_token)

    assert response["status"].startswith("200")
    assert [(row.player_id, row.role) for row in observed["plan"].promises] == [
        ("p-core", "core"), ("p-young", "development"),
    ]

    invalid = _request(app, "POST", "/api/v1/seasons/player-promises", {
        "schema_version": 1, "promises": [],
    }, csrf=app.csrf_token)
    assert invalid["status"].startswith("422")
    assert invalid["json"]["error"]["code"] == "invalid_player_promises"


def test_manager_decision_route_parses_manual_lineup_contract(tmp_path, monkeypatch):
    observed = {}
    starters = [f"p{index:02d}" for index in range(11)]

    class Workspace:
        def set_manager_decision(self, decision, *, fixture_id=None):
            observed["decision"] = decision
            return {"season_id": "season-0001", "revision": 2}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: Workspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/seasons/decision", {
        "fixture_id": "md01-fx01",
        "decision": {
            "team": "Brazil", "tactic": "balanced", "rotation": "balanced",
            "lineup": {
                "starters": starters, "bench": ["p11", "p12"],
                "source": "manual", "roster_fingerprint": "a" * 64,
            },
        },
    }, csrf=app.csrf_token)
    assert response["status"].startswith("200")
    assert observed["decision"].lineup.starters == tuple(starters)
    assert observed["decision"].lineup.bench == ("p11", "p12")


def test_season_route_rejects_invalid_plan_before_workspace_load(tmp_path):
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/seasons", {
        "teams": ["A", "B", "C"],
    }, csrf=app.csrf_token)
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "invalid_season"


def test_match_route_rejects_unknown_tactic_before_queueing(tmp_path):
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
        "plan": {
            "experience": "tactical_lab",
            "home_tactic": "imaginary",
            "away_tactic": "team_identity",
            "reuse_last_seed": False,
        },
    }, csrf=app.csrf_token)
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "invalid_match_plan"
    assert app.task_queue.list_tasks() == []


def test_tactical_lab_queue_resolves_shared_seed_and_normalizes_plan(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

        def replay_seed(self, home, away):
            assert (home, away) == ("Brazil", "Argentina")
            return 77

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
        "plan": {
            "experience": "tactical_lab",
            "home_tactic": "gegenpress",
            "away_tactic": "low_block_counter",
            "reuse_last_seed": True,
        },
    }, csrf=app.csrf_token, idempotency_key="paired-tactic-run")
    assert response["status"].startswith("202")
    request = response["json"]["task"]["request"]
    assert request["seed_override"] == 77
    assert request["plan"]["score_path"] == "physics_official"
    assert request["plan"]["claim_boundary"].startswith(
        "single_run_descriptive_only"
    )


def test_task_web_adapter_exposes_only_safe_comparison_html_url(tmp_path):
    app = ProductWebApp(tmp_path)
    safe = app._task_for_web({"result": {
        "comparison_dashboard": (
            "outputs/studio/demo/matches/m2-paired-vs-m1.comparison.html"
        ),
    }})
    assert safe["result"]["comparison_url"].endswith(".comparison.html")
    unsafe = app._task_for_web({"result": {
        "comparison_dashboard": "outputs/studio/../../private.html",
    }})
    assert "comparison_url" not in unsafe["result"]
    assert app._safe_artifact_url("outputs/studio/demo/matches/m1.html") == (
        "/artifacts/outputs/studio/demo/matches/m1.html"
    )
    assert app._safe_artifact_url("outputs/studio/../private.html") is None
    pair = app._task_for_web({"result": {
        "baseline_dashboard": "outputs/studio/demo/matches/b.html",
        "treatment_dashboard": "outputs/studio/demo/matches/t.html",
        "comparison_dashboard": "outputs/studio/demo/matches/p.comparison.html",
    }})
    assert pair["result"]["baseline_url"].endswith("/b.html")
    assert pair["result"]["treatment_url"].endswith("/t.html")
    assert pair["result"]["comparison_url"].endswith(".comparison.html")
    escaped = app._task_for_web({"result": {
        "baseline_dashboard": "outputs/studio/../../secret.html",
        "treatment_dashboard": "javascript:alert(1)",
    }})
    assert "baseline_url" not in escaped["result"]
    assert "treatment_url" not in escaped["result"]


def _study_plan_payload(study_id="pressing-study"):
    return {
        "study_id": study_id,
        "fixture": {"home": "Brazil", "away": "Argentina"},
        "baseline": {
            "home_tactic": "balanced", "away_tactic": "low_block_counter",
        },
        "treatment": {
            "home_tactic": "gegenpress", "away_tactic": "low_block_counter",
        },
        "seeds": [101, 102, 103, 104],
        "fast": True,
    }


def _paired_plan_payload(seed=77):
    return {
        "seed": seed,
        "baseline": {
            "home_tactic": "balanced", "away_tactic": "low_block_counter",
        },
        "treatment": {
            "home_tactic": "gegenpress", "away_tactic": "low_block_counter",
        },
    }


def test_paired_match_route_is_mode_gated_validated_and_idempotent(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {
        "home": "Brazil", "away": "Argentina", "fast": True,
        "plan": _paired_plan_payload(),
    }
    first = _request(
        app, "POST", "/api/v1/paired-matches", payload,
        csrf=app.csrf_token, idempotency_key="pair-web-1",
    )
    assert first["status"].startswith("202")
    task = first["json"]["task"]
    assert task["kind"] == "paired_match"
    assert task["request"]["plan"]["seed"] == 77
    assert task["request"]["plan"]["focus_side"] == "home"
    duplicate = _request(
        app, "POST", "/api/v1/paired-matches", payload,
        csrf=app.csrf_token, idempotency_key="pair-web-1",
    )
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["task"]["task_id"] == task["task_id"]

    invalid = {**payload, "plan": _paired_plan_payload()}
    invalid["plan"]["treatment"] = invalid["plan"]["baseline"]
    rejected = _request(
        app, "POST", "/api/v1/paired-matches", invalid,
        csrf=app.csrf_token,
    )
    assert rejected["status"].startswith("422")
    assert rejected["json"]["error"]["code"] == "invalid_paired_match"


def test_paired_match_route_rejects_stable_mode(tmp_path, monkeypatch):
    class StableWorkspace:
        config = type("Config", (), {"mode": "stable"})()

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: StableWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(
        app, "POST", "/api/v1/paired-matches", {
            "home": "Brazil", "away": "Argentina", "fast": True,
            "plan": _paired_plan_payload(),
        }, csrf=app.csrf_token,
    )
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "invalid_paired_match"
    assert app.task_queue.list_tasks() == []


def test_world_model_fork_route_is_research_only_and_idempotent(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {
        "home": "Brazil", "away": "Argentina", "fast": True,
        "plan": {
            "home_tactic": "balanced", "away_tactic": "low_block_counter",
            "seed": 77,
        },
    }
    first = _request(
        app, "POST", "/api/v1/world-model-forks", payload,
        csrf=app.csrf_token, idempotency_key="fork-web-1",
    )
    assert first["status"].startswith("202")
    assert first["json"]["task"]["kind"] == "world_model_fork"
    assert first["json"]["task"]["request"]["plan"][
        "baseline_policy"
    ] == "predict_only"
    duplicate = _request(
        app, "POST", "/api/v1/world-model-forks", payload,
        csrf=app.csrf_token, idempotency_key="fork-web-1",
    )
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["task"]["task_id"] == first["json"]["task"][
        "task_id"
    ]

    FakeWorkspace.config.mode = "cognitive"
    rejected = _request(
        app, "POST", "/api/v1/world-model-forks", payload,
        csrf=app.csrf_token,
    )
    assert rejected["status"].startswith("422")
    assert rejected["json"]["error"]["code"] == "invalid_world_model_fork"


def test_interrupted_pair_requeue_api_preserves_task_identity_and_requires_csrf(
    tmp_path,
):
    app = ProductWebApp(tmp_path)
    task, _ = app.task_queue.submit_paired_match(
        "Brazil", "Argentina", fast=True, plan=_paired_plan_payload(),
    )
    app.task_queue.claim_next("crashed-web-worker")
    assert app.task_queue.recover_running(reason="process_restart") == 1
    path = f"/api/v1/tasks/{task['task_id']}/requeue"
    denied = _request(app, "POST", path, {
        "reason": "studio_pair_transaction_resume",
    })
    assert denied["status"].startswith("403")
    resumed = _request(app, "POST", path, {
        "reason": "studio_pair_transaction_resume",
    }, csrf=app.csrf_token)
    assert resumed["status"].startswith("200")
    assert resumed["json"]["task"]["task_id"] == task["task_id"]
    assert resumed["json"]["task"]["state"] == "queued"
    repeated = _request(app, "POST", path, {
        "reason": "studio_pair_transaction_resume",
    }, csrf=app.csrf_token)
    assert repeated["status"].startswith("409")
    assert repeated["json"]["error"]["code"] == "task_not_interrupted"


def test_world_model_fork_set_route_is_research_only_fixed_and_idempotent(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {
        "home": "Brazil", "away": "Argentina", "fast": True,
        "plan": {
            "home_tactic": "balanced",
            "away_tactic": "low_block_counter",
            "seed": 77,
            "branch_times_sec": [1800, 2700, 3600],
        },
    }
    first = _request(
        app, "POST", "/api/v1/world-model-fork-sets", payload,
        csrf=app.csrf_token, idempotency_key="fork-set-web-1",
    )
    assert first["status"].startswith("202")
    task = first["json"]["task"]
    assert task["kind"] == "world_model_fork_set"
    assert task["request"]["plan"]["fixed_scenario_budget"] == 3
    assert task["fork_set_progress"] == {
        "state": "queued", "scenarios_completed": 0,
        "fixed_scenario_budget": 3, "ranking_withheld": True,
    }
    duplicate = _request(
        app, "POST", "/api/v1/world-model-fork-sets", payload,
        csrf=app.csrf_token, idempotency_key="fork-set-web-1",
    )
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["task"]["task_id"] == task["task_id"]

    FakeWorkspace.config.mode = "stable"
    rejected = _request(
        app, "POST", "/api/v1/world-model-fork-sets", payload,
        csrf=app.csrf_token,
    )
    assert rejected["status"].startswith("422")
    assert rejected["json"]["error"]["code"] == (
        "invalid_world_model_fork_set"
    )


def test_manager_future_set_route_freezes_official_context_and_is_idempotent(
    tmp_path, monkeypatch,
):
    from src.product.manager_future import build_manager_future_context
    from src.product.season import (
        ManagerDecision, SeasonPlan, new_season_state,
    )

    season = new_season_state(
        SeasonPlan(
            ("A", "B", "C", "D"), fast=True, manager_team="A",
        ),
        season_id="season-0001", seed=7,
        created_at="2026-01-01T00:00:00Z",
    )
    fixture = next(
        row for row in season["fixtures"]
        if "A" in {row["home"], row["away"]}
    )
    fixture["manager_decision"] = ManagerDecision(
        team="A", tactic="gegenpress",
    ).as_dict()
    context = build_manager_future_context(season)

    observed_review = {}

    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

        def manager_future_set_context(self, *, fixture_id=None):
            assert fixture_id == fixture["fixture_id"]
            return context

        def review_manager_future_set(self, **kwargs):
            observed_review.update(kwargs)
            return {
                "season_id": context["season_id"],
                "revision": context["season_revision"] + 1,
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    payload = {
        "fixture_id": fixture["fixture_id"],
        "branch_times_sec": [1800, 2700, 3600],
    }
    denied = _request(
        app, "POST", "/api/v1/seasons/world-model-future-set", payload,
    )
    assert denied["status"].startswith("403")

    first = _request(
        app, "POST", "/api/v1/seasons/world-model-future-set", payload,
        csrf=app.csrf_token, idempotency_key="manager-future-1",
    )
    assert first["status"].startswith("202")
    task = first["json"]["task"]
    assert task["request"]["manager_context"] == context
    assert task["request"]["home"] == context["fixture"]["home"]
    assert task["request"]["plan"]["seed"] == context["match_seed"]

    duplicate = _request(
        app, "POST", "/api/v1/seasons/world-model-future-set", payload,
        csrf=app.csrf_token, idempotency_key="manager-future-1",
    )
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["task"]["task_id"] == task["task_id"]

    incomplete = _request(
        app, "POST", "/api/v1/seasons/world-model-future-review", {
            "task_id": task["task_id"],
            "fixture_id": fixture["fixture_id"],
            "expected_revision": context["season_revision"],
            "intent": "keep_after_review",
        }, csrf=app.csrf_token,
    )
    assert incomplete["status"].startswith("422")
    assert incomplete["json"]["error"]["code"] == (
        "invalid_manager_future_review"
    )

    claimed = app.task_queue.claim_next("review-test-worker")
    assert claimed["task_id"] == task["task_id"]
    app.task_queue.complete(
        task["task_id"], "review-test-worker",
        {"status": "complete", "manager_context": {}},
    )
    review_payload = {
        "task_id": task["task_id"],
        "fixture_id": fixture["fixture_id"],
        "expected_revision": context["season_revision"],
        "intent": "keep_after_review",
    }
    denied_review = _request(
        app, "POST", "/api/v1/seasons/world-model-future-review",
        review_payload,
    )
    assert denied_review["status"].startswith("403")
    reviewed = _request(
        app, "POST", "/api/v1/seasons/world-model-future-review",
        review_payload, csrf=app.csrf_token,
    )
    assert reviewed["status"].startswith("200")
    assert reviewed["json"]["season"]["revision"] == (
        context["season_revision"] + 1
    )
    assert observed_review["task_id"] == task["task_id"]
    assert observed_review["intent"] == "keep_after_review"
    assert observed_review["final_decision"] is None


def test_manager_future_review_runs_end_to_end_without_a_second_state(
    tmp_path, monkeypatch,
):
    from src.product.match_plan import WorldModelForkSetPlan
    from src.product.season import ManagerDecision, SeasonPlan
    from src.product.workspace import ProductWorkspace, StudioConfig

    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Review E2E", mode="research", seed=13),
    )
    created = workspace.create_season(
        SeasonPlan(
            ("A", "B", "C", "D"), fast=True, manager_team="A",
        )
    )
    fixture_id = created["next_manager_fixture"]["fixture_id"]
    decided = workspace.set_manager_decision(
        ManagerDecision(team="A", tactic="gegenpress"),
        fixture_id=fixture_id,
    )
    context = workspace.manager_future_set_context(fixture_id=fixture_id)
    plan = WorldModelForkSetPlan(
        context["home_tactic"], context["away_tactic"],
        context["match_seed"], (1800, 2700),
    )
    app = ProductWebApp(tmp_path)
    task, _ = app.task_queue.submit_world_model_fork_set(
        context["fixture"]["home"],
        context["fixture"]["away"],
        fast=context["fast"],
        plan=plan,
        manager_context=context,
    )
    aggregate = {
        "eligible_scenarios": 2,
        "verified_anchor_scenarios": 2,
        "action_divergence_scenarios": 1,
        "local_attribution_scenarios": 1,
        "descriptive_future_difference_scenarios": 1,
        "timing_sensitivity_observed": True,
        "status_counts": {
            "local_action_divergence_with_descriptive_future_difference": 1,
            "no_realized_action_divergence": 1,
        },
        "ranking_performed": False,
        "best_branch_time": None,
    }
    authority = {
        "descriptive_simulator_timing_sensitivity": True,
        "best_time_recommendation": False,
        "match_outcome_causality": False,
        "population_inference": False,
        "real_football_causality": False,
        "promotion_authorized": False,
    }

    def execute(_workspace, frozen_plan, **kwargs):
        root = (
            workspace.output_root
            / "fork_sets"
            / kwargs["set_id"]
        )
        root.mkdir(parents=True, exist_ok=True)
        result_path = root / "result.json"
        dashboard_path = root / "index.html"
        def identity(payload):
            return hashlib.sha256(json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()

        rows = []
        for index, branch in enumerate(frozen_plan.branch_times_sec):
            examples = []
            if index == 0:
                windows = []
                for duration in (30, 120):
                    window = {
                        "schema_version": 1,
                        "window_sec": duration,
                        "delta": {
                            "actions": 0, "passes": -1, "shots": 1,
                            "goals": 0, "turnovers": 1,
                        },
                        "additional_policy_changes": 0,
                        "causal_effect_authorized": False,
                    }
                    windows.append({
                        **window, "window_identity": identity(window),
                    })
                example = {
                    "schema_version": 1,
                    "opportunity_identity": "a" * 64,
                    "team": "A",
                    "t_sec": 1810.0,
                    "clock": "30:10",
                    "baseline_action": "hold",
                    "treatment_action": "pass",
                    "recommended_action": "pass",
                    "directly_observed": True,
                    "local_policy_attribution_eligible": True,
                    "downstream_windows": windows,
                    "downstream_causal_attribution_authorized": False,
                }
                examples.append({
                    **example, "example_identity": identity(example),
                })
            rows.append({
                "branch_at_sec": float(branch),
                "branch_minute": float(branch) / 60.0,
                "future_status": (
                    "local_action_divergence_with_descriptive_future_difference"
                    if index == 0 else "no_realized_action_divergence"
                ),
                "eligible": True,
                "branch_anchor_verified": True,
                "branch_state_identity": format(index + 1, "064x"),
                "changed_actions": 1 if index == 0 else 0,
                "locally_attributable_changes": 1 if index == 0 else 0,
                "descriptive_future_difference_count": (
                    1 if index == 0 else 0
                ),
                "simulator_local_action_attribution": index == 0,
                "outcome_causality": False,
                "real_football_causality": False,
                "mechanism_examples": examples,
                "mechanism_examples_truncated": False,
            })
        artifact = {
            "schema_version": 1,
            "set_id": kwargs["set_id"],
            "status": "complete",
            "fixture": {
                "home": kwargs["home"], "away": kwargs["away"],
            },
            "source_context": kwargs["source_context"],
            "plan": frozen_plan.as_dict(),
            "rows": rows,
            "aggregate": aggregate,
            "claim_authority": authority,
        }
        result_path.write_text(json.dumps(artifact), encoding="utf-8")
        dashboard_path.write_text("review evidence", encoding="utf-8")
        return {
            **artifact,
            "result_path": str(result_path),
            "dashboard_path": str(dashboard_path),
        }

    monkeypatch.setattr(
        "src.product.world_model_fork_set.execute_world_model_fork_set",
        execute,
    )
    assert BackgroundMatchWorker(
        app.task_queue, workspace_loader=lambda _root: workspace,
    ).run_once()
    completed = app.task_queue.get_task(task["task_id"])
    assert completed["state"] == "completed"
    assert len(completed["result"]["scenario_evidence"]) == 2
    projection = app._studio_status()["studio"]["season"]
    projected_set = projection["manager_future_sets"][0]
    intervention = projection["manager_intervention_workspace"]
    navigator = projection["manager_world_navigator"]
    assert intervention["workflow_state"] == "evidence_ready_for_review"
    assert intervention["allowed_actions"]["review_future_set"] is True
    assert intervention["stages"][0]["status"] == "decision_frozen"
    assert intervention["stages"][2]["status"] == (
        "scenario_evidence_available"
    )
    assert intervention["evidence_summary"][
        "local_attribution_scenarios"
    ] == 1
    assert (
        projection["matchday_command_center"][
            "manager_intervention_workspace"
        ]["workspace_identity"]
        == intervention["workspace_identity"]
    )
    assert navigator["primary_action"]["action_id"] == (
        "review_future_evidence"
    )
    assert navigator["current_chapter"]["workspace_identity"] == (
        intervention["workspace_identity"]
    )
    assert (
        projection["matchday_command_center"][
            "manager_world_navigator"
        ]["navigator_identity"]
        == navigator["navigator_identity"]
    )
    assert projected_set["scenario_evidence_available"] is True
    assert len(projected_set["scenario_evidence"]) == 2
    assert projected_set["scenario_evidence"][0][
        "simulator_local_action_attribution"
    ] is True
    assert projected_set["scenario_evidence"][0]["mechanism_examples"][0][
        "treatment_action"
    ] == "pass"

    response = _request(
        app, "POST", "/api/v1/seasons/world-model-future-review", {
            "task_id": task["task_id"],
            "fixture_id": fixture_id,
            "expected_revision": decided["revision"],
            "intent": "keep_after_review",
        }, csrf=app.csrf_token,
    )
    assert response["status"].startswith("200")
    season = workspace.season_status()
    receipt = season["next_manager_fixture"]["manager_future_reviews"][0]
    assert receipt["task_id"] == task["task_id"]
    assert receipt["intent"] == "keep_after_review"
    assert receipt["schema_version"] == 3
    assert len(receipt["scenario_evidence"]) == 2
    assert receipt["scenario_evidence"][0]["mechanism_examples"][0][
        "downstream_windows"
    ][0]["window_sec"] == 30
    assert season["revision"] == decided["revision"] + 1
    assert season["manager_decision_ledger"]["summary"][
        "world_model_future_reviews"
    ]["reviewed_future_sets"] == 1
    assert season["manager_decision_ledger"]["summary"][
        "world_model_future_reviews"
    ]["retained_mechanism_examples"] == 1
    reviewed_projection = app._studio_status()["studio"]["season"]
    reviewed_workspace = reviewed_projection[
        "manager_intervention_workspace"
    ]
    reviewed_navigator = reviewed_projection["manager_world_navigator"]
    assert reviewed_workspace["workflow_state"] == (
        "review_recorded_decision_refrozen"
    )
    assert reviewed_workspace["stages"][3]["status"] == (
        "review_recorded"
    )
    assert reviewed_workspace["evidence_summary"][
        "reviewed_future_sets"
    ] == 1
    assert reviewed_workspace["allowed_actions"][
        "advance_official_match"
    ] is True
    assert reviewed_navigator["primary_action"]["action_id"] == (
        "advance_official_world"
    )
    assert reviewed_navigator["history_chapters"] == []
    assert "manager_future_reviews" not in workspace._session()
    repeated = _request(
        app, "POST", "/api/v1/seasons/world-model-future-review", {
            "task_id": task["task_id"],
            "fixture_id": fixture_id,
            "expected_revision": decided["revision"],
            "intent": "keep_after_review",
        }, csrf=app.csrf_token,
    )
    assert repeated["status"].startswith("200")
    assert repeated["json"]["season"]["revision"] == season["revision"]
    assert len(
        workspace.season_status()["next_manager_fixture"][
            "manager_future_reviews"
        ]
    ) == 1


def test_tactical_study_route_is_research_only_and_idempotent(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    first = _request(
        app, "POST", "/api/v1/tactical-studies",
        {"plan": _study_plan_payload()}, csrf=app.csrf_token,
        idempotency_key="fixed-study-1",
    )
    assert first["status"].startswith("202")
    assert first["json"]["task"]["kind"] == "tactical_study"
    duplicate = _request(
        app, "POST", "/api/v1/tactical-studies",
        {"plan": _study_plan_payload()}, csrf=app.csrf_token,
        idempotency_key="fixed-study-1",
    )
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["created"] is False
    assert duplicate["json"]["task"]["task_id"] == first["json"]["task"]["task_id"]


def test_tactical_study_route_rejects_non_research_mode(tmp_path, monkeypatch):
    class StableWorkspace:
        config = type("Config", (), {"mode": "stable"})()

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: StableWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(
        app, "POST", "/api/v1/tactical-studies",
        {"plan": _study_plan_payload()}, csrf=app.csrf_token,
    )
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "study_requires_research_mode"
    assert app.task_queue.list_tasks() == []


def test_tactical_study_route_validates_plan_before_workspace_lookup(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load",
        lambda _root: pytest.fail("invalid plan must not load workspace"),
    )
    app = ProductWebApp(tmp_path)
    invalid = _study_plan_payload()
    invalid["treatment"] = invalid["baseline"]
    response = _request(
        app, "POST", "/api/v1/tactical-studies", {"plan": invalid},
        csrf=app.csrf_token,
    )
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "invalid_tactical_study"
    assert app.task_queue.list_tasks() == []


def test_tactical_study_task_never_discloses_interim_effects(
    tmp_path, monkeypatch,
):
    plan = TacticalStudyPlan.from_payload(_study_plan_payload())
    output_root = tmp_path / "outputs" / "studio" / "demo"
    progress_path = output_root / "studies" / plan.study_id / "progress.json"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(json.dumps({
        "state": "running", "pairs_completed": 2, "fixed_pair_budget": 4,
        "interim_effects_disclosed": False, "analysis": None,
        "completed_pairs": [{"secret_effect": 99.0}],
    }), encoding="utf-8")

    class FakeWorkspace:
        pass

    workspace = FakeWorkspace()
    workspace.output_root = output_root
    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: workspace,
    )
    app = ProductWebApp(tmp_path)
    task = app._task_for_web({
        "kind": "tactical_study", "state": "running",
        "request": {"plan": plan.as_dict()}, "result": None,
    })
    assert task["study_progress"] == {
        "state": "running", "pairs_completed": 2, "fixed_pair_budget": 4,
        "analysis_withheld": True,
    }
    assert "secret_effect" not in json.dumps(task)

    progress_path.write_text(json.dumps({
        "state": "running", "pairs_completed": 2, "fixed_pair_budget": 4,
        "interim_effects_disclosed": False,
        "analysis": {"primary_mean": 99.0},
    }), encoding="utf-8")
    invalid = app._task_for_web({
        "kind": "tactical_study", "state": "running",
        "request": {"plan": plan.as_dict()}, "result": None,
    })
    assert invalid["study_progress"] == {
        "state": "invalid_progress", "analysis_withheld": True,
    }
    assert "primary_mean" not in json.dumps(invalid)


def test_tactical_study_task_rejects_tampered_study_id_before_path_use(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load",
        lambda _root: pytest.fail("invalid plan must fail before workspace lookup"),
    )
    task = ProductWebApp(tmp_path)._task_for_web({
        "kind": "tactical_study", "state": "running",
        "request": {"plan": {"study_id": "../../escape"}}, "result": None,
    })
    assert task["study_progress"] == {
        "state": "invalid_progress", "analysis_withheld": True,
    }


def test_tactical_study_final_dashboard_url_is_html_and_workspace_scoped(tmp_path):
    app = ProductWebApp(tmp_path)
    safe = app._task_for_web({"result": {
        "study_dashboard": "outputs/studio/demo/studies/s1/result.html",
    }})
    assert safe["result"]["study_url"].endswith("/studies/s1/result.html")
    unsafe = app._task_for_web({"result": {
        "study_dashboard": "outputs/studio/../../secret.html",
    }})
    assert "study_url" not in unsafe["result"]


def test_studio_evidence_library_is_safe_bounded_and_effect_free(
    tmp_path, monkeypatch,
):
    plan = TacticalStudyPlan.from_payload(_study_plan_payload())
    output_root = tmp_path / "outputs/studio/demo"

    class FakeWorkspace:
        def __init__(self):
            self.output_root = output_root

        def status(self):
            return {
                "last_match": None,
                "match_history_truncated": False,
                "match_history": [{
                    "match_id": "m-safe", "home": "<Home>", "away": "Away",
                    "seed": 7, "fast": True,
                    "score": {"home": 1, "away": 0},
                    "integrity": "accepted", "experience": "tactical_lab",
                    "home_tactic": "gegenpress",
                    "away_tactic": "low_block_counter",
                    "dashboard": "outputs/studio/demo/matches/m-safe.html",
                    "comparison_dashboard": (
                        "outputs/studio/demo/matches/m-safe.comparison.html"
                    ),
                    "secret": "must-not-cross-library-boundary",
                }, {
                    "match_id": "m-unsafe", "home": "A", "away": "B",
                    "dashboard": "outputs/studio/../../secret.html",
                    "comparison_dashboard": "javascript:alert(1)",
                }],
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    session_path = tmp_path / "data/persistence/product_session.json"
    session_path.parent.mkdir(parents=True)
    session_path.write_text("{}", encoding="utf-8")
    app = ProductWebApp(tmp_path)
    task = {
        "task_id": "a" * 32, "kind": "tactical_study", "state": "completed",
        "request": {"plan": plan.as_dict()},
        "result": {
            "study_dashboard": "outputs/studio/demo/studies/s/index.html",
            "analysis": {"secret_interim_effect": 99},
        },
    }
    pair_task = {
        "task_id": "b" * 32, "kind": "paired_match", "state": "completed",
        "request": {
            "home": "Brazil", "away": "Argentina", "fast": True,
            "plan": _paired_plan_payload(),
        },
        "result": {
            "baseline_match_id": "m-base", "treatment_match_id": "m-treat",
            "baseline_dashboard": "outputs/studio/demo/matches/base.html",
            "treatment_dashboard": "outputs/studio/demo/matches/treat.html",
            "comparison_dashboard": (
                "outputs/studio/demo/matches/treat.comparison.html"
            ),
        },
    }
    fork_task = {
        "task_id": "c" * 32, "kind": "world_model_fork",
        "state": "completed",
        "request": {
            "home": "Brazil", "away": "Argentina", "fast": True,
            "plan": {
                "home_tactic": "balanced",
                "away_tactic": "low_block_counter", "seed": 88,
            },
        },
        "result": {
            "baseline_match_id": "wm-base", "treatment_match_id": "wm-policy",
            "baseline_dashboard": "outputs/studio/demo/matches/wm-base.html",
            "treatment_dashboard": "outputs/studio/demo/matches/wm-policy.html",
            "comparison_dashboard": (
                "outputs/studio/demo/matches/wm-policy.comparison.html"
            ),
            "propagation": {
                "available": True,
                "status": "direct_action_changes_observed",
                "changed_decisions": 4,
                "directly_observed_changes": 3,
                "locally_attributable_changes": 2,
                "replay_windows_available": True,
                "downstream_causal_attribution_authorized": False,
                "future_summary": {
                    "available": True,
                    "status": (
                        "local_action_divergence_with_"
                        "descriptive_future_difference"
                    ),
                    "changed_actions": 4,
                    "descriptive_outcome_difference_count": 6,
                    "simulator_local_action_attribution": True,
                    "match_outcome_causality": False,
                    "real_football_causality": False,
                },
            },
        },
    }
    monkeypatch.setattr(
        app.task_queue, "list_tasks",
        lambda *, limit=50: [pair_task, fork_task, task],
    )
    response = _request(app, path="/api/v1/studio")
    library = response["json"]["evidence_library"]
    assert response["status"].startswith("200")
    assert len(library["matches"]) == 2
    assert library["matches"][0]["dashboard_url"].endswith("m-safe.html")
    assert library["matches"][0]["comparison_url"].endswith(
        "m-safe.comparison.html"
    )
    assert library["matches"][1]["dashboard_url"] is None
    assert library["matches"][1]["comparison_url"] is None
    assert library["studies"][0]["study_url"].endswith("/studies/s/index.html")
    assert library["pairs"][0]["seed"] == 77
    assert library["pairs"][0]["focus_side"] == "home"
    assert library["pairs"][0]["baseline_url"].endswith("/base.html")
    assert library["pairs"][0]["treatment_url"].endswith("/treat.html")
    assert library["pairs"][0]["comparison_url"].endswith(
        "/treat.comparison.html"
    )
    assert library["forks"][0]["seed"] == 88
    assert library["forks"][0]["home_tactic"] == "balanced"
    assert library["forks"][0]["baseline_url"].endswith("/wm-base.html")
    assert library["forks"][0]["comparison_url"].endswith(
        "/wm-policy.comparison.html"
    )
    assert library["forks"][0]["propagation"] == {
        "available": True,
        "status": "direct_action_changes_observed",
        "changed_decisions": 4,
        "directly_observed_changes": 3,
        "locally_attributable_changes": 2,
        "replay_windows_available": True,
        "downstream_causal_attribution_authorized": False,
        "future_summary": {
            "available": True,
            "status": (
                "local_action_divergence_with_"
                "descriptive_future_difference"
            ),
            "changed_actions": 4,
            "descriptive_outcome_difference_count": 6,
            "simulator_local_action_attribution": True,
            "match_outcome_causality": False,
            "real_football_causality": False,
        },
    }
    serialized = json.dumps(library)
    assert "must-not-cross-library-boundary" not in serialized
    assert "secret_interim_effect" not in serialized
    assert '"seeds"' not in serialized
    assert "javascript:" not in serialized

    fork_task["result"]["propagation"]["future_summary"].update({
        "available": "true",
        "status": "forged_causal_success",
        "match_outcome_causality": True,
        "real_football_causality": True,
    })
    tampered = _request(app, path="/api/v1/studio")["json"][
        "evidence_library"
    ]["forks"][0]["propagation"]["future_summary"]
    assert tampered["status"] == "unknown_future_status"
    assert tampered["available"] is False
    assert tampered["match_outcome_causality"] is False
    assert tampered["real_football_causality"] is False


def test_archived_season_journal_links_remain_workspace_scoped(
    tmp_path, monkeypatch,
):
    class FakeWorkspace:
        output_root = tmp_path / "outputs/studio/demo"

        def status(self):
            return {
                "last_match": None, "match_history": [], "season": None,
                "season_history": [{
                    "manager_profile": {"journal": [
                        {"fixture_id": "safe", "dashboard": "outputs/studio/demo/matches/safe.html"},
                        {"fixture_id": "unsafe", "dashboard": "outputs/studio/../../secret.html"},
                    ]},
                }],
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    session = tmp_path / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True)
    session.write_text("{}", encoding="utf-8")

    payload = ProductWebApp(tmp_path)._studio_status()
    journal = payload["studio"]["season_history"][0]["manager_profile"]["journal"]

    assert journal[0]["dashboard_url"].endswith("/matches/safe.html")
    assert "dashboard_url" not in journal[1]


def test_match_route_queues_idempotently_then_returns_report_url(tmp_path, monkeypatch):
    dashboard = tmp_path / "outputs/studio/demo/matches/0001.html"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text("<html><body>report</body></html>", encoding="utf-8")

    class FakeWorkspace:
        config = type("Config", (), {"mode": "research"})()

        def readiness(self):
            return {"ready": True, "blockers": []}

        def run_match(self, home, away, *, fast, plan, seed_override):
            return {
                "match_id": "0001-brazil-vs-argentina",
                "fixture": {"home": home, "away": away, "fast": fast},
                "result": {"score": {"home": 1, "away": 0}},
                "integrity": {"accepted": True, "state": "accepted", "blockers": []},
                "report_path": str(dashboard.with_suffix(".json")),
                "dashboard_path": str(dashboard),
                "raw_summary": {"large": "must-not-cross-api-boundary"},
            }

        def replay_seed(self, home, away):
            return 17

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert response["status"].startswith("202")
    task_id = response["json"]["task"]["task_id"]
    duplicate = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["created"] is False
    assert duplicate["json"]["task"]["task_id"] == task_id
    conflict = _request(app, "POST", "/api/v1/matches", {
        "home": "France", "away": "Spain", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert conflict["status"].startswith("409")
    assert conflict["json"]["error"]["code"] == "idempotency_conflict"

    assert BackgroundMatchWorker(app.task_queue).run_once()
    observed = _request(app, path=f"/api/v1/tasks/{task_id}")
    assert observed["json"]["task"]["state"] == "completed"
    dashboard_url = observed["json"]["task"]["result"]["dashboard_url"]
    assert dashboard_url.endswith("/0001.html")
    assert "raw_summary" not in observed["json"]

    artifact = _request(app, path=dashboard_url)
    assert artifact["status"].startswith("200")
    assert artifact["body"].startswith(b"<html>")


def test_artifacts_are_html_only_and_cannot_escape_workspace(tmp_path):
    secret = tmp_path / "data/private.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("private", encoding="utf-8")
    app = ProductWebApp(tmp_path)
    traversal = _request(app, path="/artifacts/outputs/studio/../../../data/private.txt")
    assert traversal["status"].startswith("404")
    assert b"private" not in traversal["body"]


def test_mutations_fail_fast_when_another_operation_is_running(tmp_path):
    app = ProductWebApp(tmp_path)
    app._mutation_lock.acquire()
    try:
        response = _request(app, "POST", "/api/v1/studio", {
            "name": "Busy", "mode": "stable", "seed": 42,
        }, csrf=app.csrf_token)
    finally:
        app._mutation_lock.release()
    assert response["status"].startswith("409")
    assert response["json"]["error"]["code"] == "operation_in_progress"


def test_web_recovery_center_creates_verifies_and_explicitly_restores(tmp_path):
    app = ProductWebApp(tmp_path)
    created_studio = _request(
        app, "POST", "/api/v1/studio",
        {"name": "Recovery Center", "mode": "stable", "seed": 42},
        csrf=app.csrf_token,
    )
    assert created_studio["status"].startswith("201")
    empty = _request(app, path="/api/v1/recovery")
    assert empty["json"]["backups"] == []

    no_csrf = _request(app, "POST", "/api/v1/backups", {})
    assert no_csrf["status"].startswith("403")
    created = _request(
        app, "POST", "/api/v1/backups", {}, csrf=app.csrf_token,
    )
    assert created["status"].startswith("201")
    backup_id = created["json"]["backup_id"]
    assert created["json"]["valid"] is True
    assert str(tmp_path) not in json.dumps(created["json"])
    catalog = _request(app, path="/api/v1/recovery")["json"]
    assert catalog["count"] == 1
    assert catalog["backups"][0]["backup_id"] == backup_id

    verified = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/verify", {},
        csrf=app.csrf_token,
    )
    assert verified["status"].startswith("200") and verified["json"]["valid"]
    session = tmp_path / "data/persistence/product_session.json"
    session.write_text(session.read_text(encoding="utf-8").replace(
        "Recovery Center", "Changed After Backup",
    ), encoding="utf-8")

    wrong_confirmation = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": "错误确认", "replace": True}, csrf=app.csrf_token,
    )
    assert wrong_confirmation["status"].startswith("422")
    assert "Changed After Backup" in session.read_text(encoding="utf-8")

    app._mutation_lock.acquire()
    try:
        admission_blocked = _request(
            app, "POST", "/api/v1/matches",
            {"home": "Brazil", "away": "Argentina", "fast": True},
            csrf=app.csrf_token,
        )
    finally:
        app._mutation_lock.release()
    assert admission_blocked["status"].startswith("409")
    assert admission_blocked["json"]["error"]["code"] == "operation_in_progress"

    task, _ = app.task_queue.submit_match("Brazil", "Argentina", fast=True)
    blocked = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": backup_id, "replace": True}, csrf=app.csrf_token,
    )
    assert blocked["status"].startswith("409")
    assert blocked["json"]["error"]["code"] == "restore_blocked"
    claimed = app.task_queue.claim_next("test-worker")
    app.task_queue.complete(claimed["task_id"], "test-worker", {"ok": True})

    restored = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": backup_id, "replace": True}, csrf=app.csrf_token,
    )
    assert restored["status"].startswith("200")
    assert restored["json"]["restored"] is True
    assert restored["json"]["task_history_reset"] is True
    assert "Recovery Center" in session.read_text(encoding="utf-8")
    assert app.task_queue.list_tasks() == []
    assert task["task_id"] not in json.dumps(app.task_queue.list_tasks())

    invalid_path = _request(
        app, "POST", "/api/v1/backups/../escape/verify", {}, csrf=app.csrf_token,
    )
    assert invalid_path["status"].startswith("404")
    wrong_method = _request(app, "GET", "/api/v1/backups")
    assert wrong_method["status"].startswith("405")


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_allowed(host):
    assert _is_loopback_host(host)


def test_remote_binding_is_rejected_before_socket_creation(tmp_path):
    assert not _is_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="non-loopback Web binding"):
        create_product_web_server(tmp_path, host="0.0.0.0")


def test_server_serves_real_http_on_ephemeral_loopback_port(tmp_path):
    server = create_product_web_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/healthz", timeout=3,
        ) as response:
            payload = json.loads(response.read())
            assert response.status == 200
            assert payload["status"] == "ok"
            assert payload["external_calls_made"] is False
            assert response.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_server_lease_rejects_a_second_web_process_for_same_workspace(tmp_path):
    first = create_product_web_server(tmp_path, port=0)
    try:
        with pytest.raises(RuntimeError, match="lease is already owned"):
            create_product_web_server(tmp_path, port=0)
    finally:
        first.server_close()


def test_remote_login_requires_allowed_host_https_and_secure_session(tmp_path):
    access_token = "deployment-test-token-" + "x" * 32
    policy = WebAccessPolicy(
        remote=True, access_token=access_token,
        allowed_hosts=("studio.example",),
    )
    app = ProductWebApp(tmp_path, access_policy=policy)

    wrong_host = _request(
        app, path="/login", host="attacker.example", forwarded_proto="https",
    )
    assert wrong_host["status"].startswith("400")
    insecure = _request(
        app, path="/login", host="studio.example", forwarded_proto="http",
    )
    assert insecure["status"].startswith("426")

    redirected = _request(
        app, path="/", host="studio.example", forwarded_proto="https",
    )
    assert redirected["status"].startswith("302")
    assert redirected["headers"]["Location"] == "/login"
    login_page = _request(
        app, path="/login", host="studio.example", forwarded_proto="https",
    )
    assert login_page["status"].startswith("200")
    assert b"access_token" in login_page["body"]

    rejected = _request(
        app, "POST", "/api/v1/login", {"access_token": "wrong"},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert rejected["status"].startswith("401")
    accepted = _request(
        app, "POST", "/api/v1/login", {"access_token": access_token},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert accepted["status"].startswith("200")
    assert access_token.encode() not in accepted["body"]
    cookie_header = accepted["headers"]["Set-Cookie"]
    assert all(value in cookie_header for value in (
        "HttpOnly", "Secure", "SameSite=Strict",
    ))
    cookie = cookie_header.split(";", 1)[0]

    unauthorized = _request(
        app, path="/api/v1/studio",
        host="studio.example", forwarded_proto="https",
    )
    assert unauthorized["status"].startswith("401")
    authenticated = _request(
        app, path="/api/v1/studio", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert authenticated["status"].startswith("200")
    assert authenticated["json"]["access"]["mode"] == "remote_authenticated"
    assert access_token not in json.dumps(authenticated["json"])
    remote_document = _request(
        app, path="/", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )["body"].decode("utf-8")
    assert 'id="logout-button" type="button">' in remote_document

    malformed = _request(
        app, "POST", "/api/v1/login", {"access_token": 42},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert malformed["status"].startswith("401")
    for _ in range(4):
        attempt = _request(
            app, "POST", "/api/v1/login", {"access_token": "wrong"},
            csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
        )
        assert attempt["status"].startswith("401")
    limited = _request(
        app, "POST", "/api/v1/login", {"access_token": access_token},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert limited["status"].startswith("429")
    assert int(limited["headers"]["Retry-After"]) >= 1
    operations = _request(
        app, path="/api/v1/operations", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert operations["json"]["authentication"] == {
        "accepted": 1, "rejected": 6, "rate_limited": 1, "logged_out": 0,
    }
    assert access_token not in app.telemetry.path.read_text(encoding="utf-8")

    missing_csrf = _request(
        app, "POST", "/api/v1/logout", {}, cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert missing_csrf["status"].startswith("403")
    logout = _request(
        app, "POST", "/api/v1/logout", {}, csrf=app.csrf_token, cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert logout["status"].startswith("200")
    assert logout["json"]["revoked"] is True
    assert "Max-Age=0" in logout["headers"]["Set-Cookie"]
    expired = _request(
        app, path="/api/v1/studio", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert expired["status"].startswith("401")
    assert app.telemetry.snapshot()["authentication"]["logged_out"] == 1


def test_web_cli_parses_explicit_remote_security_boundary():
    args = build_parser().parse_args([
        "studio", "web", "--host", "0.0.0.0", "--allow-remote",
        "--allowed-host", "studio.example", "--allowed-host", "backup.example",
    ])
    assert args.allow_remote
    assert args.host == "0.0.0.0"
    assert args.allowed_host == ["studio.example", "backup.example"]


def test_local_web_cli_ignores_unrelated_remote_secret_file(monkeypatch, tmp_path):
    captured = {}

    class FakeServer:
        server_port = 8765

        @staticmethod
        def serve_forever():
            raise KeyboardInterrupt

        @staticmethod
        def server_close():
            return None

    def fake_create_server(*args, **kwargs):
        captured.update(kwargs)
        return FakeServer()

    monkeypatch.setenv("GFS_WEB_ACCESS_TOKEN_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr("src.product.create_product_web_server", fake_create_server)
    args = build_parser().parse_args(["--base-dir", str(tmp_path), "studio", "web"])
    assert cmd_studio_web(args) == 0
    assert captured["allow_remote"] is False
    assert captured["access_token"] == ""

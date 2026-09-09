#!/usr/bin/env python3
"""Audit Studio's code-level accessibility contract without running a match."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.web import (  # noqa: E402
    _INDEX_HTML,
    _LOGIN_HTML,
    create_product_web_server,
)


_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def rendered_text(self) -> str:
        return " ".join(
            [*self.text, *(child.rendered_text() for child in self.children)]
        ).strip()

    def has_ancestor(self, tag: str) -> bool:
        node = self.parent
        while node is not None:
            if node.tag == tag:
                return True
            node = node.parent
        return False


class _Document(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self.stack = [self.root]
        self.feed(source)
        self.close()

    @property
    def nodes(self) -> list[_Node]:
        result: list[_Node] = []

        def visit(node: _Node) -> None:
            result.extend(node.children)
            for child in node.children:
                visit(child)

        visit(self.root)
        return result

    def handle_starttag(self, tag: str, attrs) -> None:
        node = _Node(tag, {key: value or "" for key, value in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.stack[-1].text.append(data.strip())


def _relative_luminance(color: str) -> float:
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _semantic_checks(source: str, *, login: bool) -> dict[str, bool]:
    document = _Document(source)
    nodes = document.nodes
    ids = [node.attrs["id"] for node in nodes if node.attrs.get("id")]
    by_id = {node.attrs.get("id"): node for node in nodes if node.attrs.get("id")}
    controls = [node for node in nodes if node.tag in {"input", "select", "textarea"}]
    buttons = [node for node in nodes if node.tag == "button"]
    labelled_controls = all(
        node.has_ancestor("label")
        or bool(node.attrs.get("aria-label"))
        or (
            bool(node.attrs.get("id"))
            and any(
                label.tag == "label" and label.attrs.get("for") == node.attrs["id"]
                for label in nodes
            )
        )
        for node in controls
    )
    named_buttons = all(
        bool(node.attrs.get("aria-label") or node.rendered_text()) for node in buttons
    )
    labelled_sections = all(
        bool(node.attrs.get("aria-labelledby"))
        and node.attrs["aria-labelledby"] in by_id
        for node in nodes
        if node.tag == "section"
    )
    live_regions = [
        node
        for node in nodes
        if node.attrs.get("role") in {"status", "alert"}
        or node.attrs.get("aria-live") in {"polite", "assertive"}
    ]
    return {
        "document_language_declared": any(
            node.tag == "html" and node.attrs.get("lang") == "zh-CN" for node in nodes
        ),
        "unique_nonempty_document_title": len(
            [node for node in nodes if node.tag == "title" and node.rendered_text()]
        ) == 1,
        "single_main_and_h1": (
            len([node for node in nodes if node.tag == "main"]) == 1
            and len([node for node in nodes if node.tag == "h1"]) == 1
        ),
        "all_ids_unique": len(ids) == len(set(ids)),
        "all_form_controls_labelled": bool(controls) and labelled_controls,
        "all_buttons_named": bool(buttons) and named_buttons,
        "sections_programmatically_named": login or labelled_sections,
        "live_regions_are_atomic": bool(live_regions)
        and all(node.attrs.get("aria-atomic") == "true" for node in live_regions),
        "no_positive_tabindex": all(
            not node.attrs.get("tabindex", "").isdigit()
            or int(node.attrs["tabindex"]) <= 0
            for node in nodes
        ),
    }


def verify_web_accessibility() -> dict:
    checks: dict[str, bool] = {}
    index = _INDEX_HTML.replace("__CSRF__", "audit").replace(
        "__REMOTE_LOGOUT_HIDDEN__", " hidden"
    ).replace("__NONCE__", "audit")
    login = _LOGIN_HTML.replace("__CSRF__", "audit").replace("__NONCE__", "audit")

    checks.update({f"index_{key}": value for key, value in _semantic_checks(index, login=False).items()})
    checks.update({f"login_{key}": value for key, value in _semantic_checks(login, login=True).items()})
    checks["skip_link_focuses_main"] = (
        '<a class="skip" href="#main">' in index
        and '<main id="main" class="grid" tabindex="-1">' in index
    )
    checks["async_busy_state_contract"] = all(
        token in index
        for token in ("function setBusy(", "setBusy(form,true)", 'aria-busy="true"')
    ) and 'aria-busy="false"' in login
    checks["error_and_completion_focus_contract"] = all(
        token in index
        for token in (
            "kind==='error'?'alert':'status'",
            "kind==='error'?'assertive':'polite'",
            "announce(message,e.message,'error',true)",
            "if(task.state==='completed'){announce(message,",
            ",'success',true);showReport(",
        )
    )
    checks["restore_keyboard_focus_contract"] = all(
        token in index
        for token in (
            "restoreTrigger=trigger",
            "restoreTrigger?.isConnected",
            "event.key==='Escape'",
            "closeRestore(true)",
        )
    )
    checks["guided_workflow_focus_contract"] = all(
        token in index
        for token in (
            'id="workflow" class="status" role="status"',
            'id="workflow-action" type="button" hidden',
            "function renderUnifiedWorkflow(data)",
            "workflowAction._target=targets[action.id]||null",
            "target.scrollIntoView({block:'center'})",
            "if(focusable)focusable.focus()",
        )
    )
    checks["workspace_navigation_contract"] = all(
        token in index
        for token in (
            'id="workspace-nav" class="workspace-nav" aria-label=',
            'data-workspace-view="career" aria-pressed="true"',
            'data-workspace-view="lab" aria-pressed="false"',
            'data-workspace-view="evidence" aria-pressed="false"',
            'data-workspace-view="operations" aria-pressed="false"',
            'id="workspace-view-description" class="status" role="status"',
        )
    )
    checks["workspace_progressive_disclosure_contract"] = all(
        token in index
        for token in (
            ".workspace-area-hidden { display:none!important }",
            "const workspaceAreaDescriptions={career:",
            "section.classList.toggle('workspace-area-hidden',!visible)",
            "sessionStorage.getItem('gfs-workspace-area')",
            "sessionStorage.setItem('gfs-workspace-area',resolved)",
            "if(!workspaceAreaExplicit&&!active)currentWorkspaceArea=workflowWorkspaceArea(data)",
        )
    )
    checks["manager_product_journey_contract"] = all(
        token in index
        for token in (
            'id="manager-product-journey" class="manager-product-journey"',
            'aria-label="&#32463;&#29702;&#20135;&#21697;&#23436;&#25972;&#20027;&#27969;&#31243;"',
            "function renderManagerProductJourney(",
            "item.dataset.stage=id",
            "item.dataset.status=status",
            "item.setAttribute('aria-current','step')",
            ".row,.matchday-journey,.manager-product-journey",
        )
    )
    checks["manager_world_dossier_contract"] = all(
        token in index
        for token in (
            'id="manager-world-story-dossier" class="cards" role="list"',
            'aria-label="&#21516;&#31456;&#20915;&#31574;&#24433;&#21709;&#26723;&#26696;"',
            "function renderManagerWorldStoryDossier(season)",
            "managerWorldStoryDossier.replaceChildren()",
            "transition=action.transition_evidence||{}",
            "transition.transitions||[]",
            "trace.textContent='实际动作转移：'",
            "gap.textContent='精确动作转移不可用：'",
            "function appendManagerWorldStoryPlayerChanges(host,world)",
            "document.createElement('details')",
            "document.createElement('summary')",
            "list.setAttribute('aria-label','同章球员级持久世界变化')",
            "for(const row of evidence.players||[])",
            "appendManagerWorldStoryPlayerChanges(worldNode,world)",
            "function appendManagerWorldStorySocietyChanges(host,society)",
            "society.state_change_evidence",
            "list.setAttribute('aria-label','同章社会与心理持久状态变化')",
            "for(const row of evidence.changes||[])",
            "function renderManagerWorldStorySocietyChanges(season)",
            "renderManagerWorldStorySocietyChanges,",
            "for(const node of [choiceNode,reviewNode,actionNode,worldNode])",
            "node.setAttribute('role','listitem')",
            "managerWorldStoryDossier.append(",
        )
    )
    checks["workflow_navigation_reveals_target_before_focus"] = all(
        token in index
        for token in (
            "workflowAction._area=workflowAreaByAction[action]",
            "applyWorkspaceArea(workflowAction._area,{remember:true,configured:true})",
            "target.scrollIntoView({block:'center'})",
            "if(focusable)focusable.focus()",
        )
    ) and index.index(
        "applyWorkspaceArea(workflowAction._area,{remember:true,configured:true})"
    ) < index.index("target.scrollIntoView({block:'center'})")
    checks["manager_decision_preview_contract"] = all(
        token in index
        for token in (
            'id="manager-decision-preview" class="decision-preview"',
            'aria-live="polite" aria-atomic="true"',
            "function managerDecisionPayload(includeRevision=false)",
            "function scheduleManagerDecisionPreview()",
            "++managerPreviewSequence",
            "managerDecisionSubmit.disabled=true",
            "payload.expected_revision=managerPreviewBaseRevision",
            "renderManagerDecisionPreview(data.preview);managerDecisionSubmit.disabled=false",
            "function renderManagerDecisionPreviewWorldModelComparison(",
            "const MANAGER_DECISION_PREVIEW_RENDER_STAGES=Object.freeze([",
            "comparison.recommended_tactic",
            "comparison.selected_tactic",
            "authority.level==='exploratory_only'",
            "deltas.risk_adjusted_value",
            "adoptManagerAdvice.textContent=",
            "'/api/v1/seasons/decision-preview'",
            "不预测比分、胜率",
        )
    )
    checks["manager_world_model_advice_contract"] = all(
        token in index
        for token in (
            'id="manager-world-model-advice" class="decision-preview"',
            'id="manager-world-model-advice-summary" class="status"',
            'id="manager-world-model-advice-candidates" class="cards" role="list"',
            'id="request-manager-advice" type="button"',
            'id="adopt-manager-advice" type="button" disabled',
            "function requestManagerWorldModelAdvice()",
            "function adoptCurrentManagerAdvice()",
            "function renderManagerWorldModelAdviceStatusReset()",
            "const MANAGER_WORLD_MODEL_ADVICE_RENDER_STAGES=Object.freeze([",
            "managerAdviceIntent='adopt_recommendation'",
            "managerAdviceIntent='reviewed_then_selected'",
            "payload.advice_adoption=",
            "建议、经理选择和赛果分别取证",
        )
    )
    checks["manager_decision_ledger_contract"] = all(
        token in index
        for token in (
            'id="manager-decision-ledger" aria-labelledby="manager-decision-ledger-title"',
            'id="manager-decision-ledger-summary" class="status" role="status"',
            'id="manager-decision-ledger-list" class="journal-list" role="list"',
            "function renderManagerDecisionLedger(season)",
            "artifactLink('打开该场完整复盘',row.dashboard)",
            "单场赛果不证明决策效果",
            "世界状态：比赛后疲劳",
            "伤停为模拟状态",
        )
    )
    checks["manager_advisor_evidence_summary_contract"] = all(
        token in index
        for token in (
            'id="manager-advisor-evidence-summary" class="status"',
            "function renderManagerDecisionLedgerAdvisorEvidence(season)",
            "evidence.advised_decisions",
            "evidence.adopted_recommendation",
            "evidence.reviewed_then_selected",
            "evidence.unlinked_advice",
            "evidence.direct_execution_coverage",
            'id="manager-advisor-protocol-evidence" class="status"',
            "function renderActionAdoptionManagerProtocol(studio)",
            "studio?.evidence?.manager_advisor_adoption",
            "protocol.fixed_information_windows",
        )
    )
    checks["dynamic_backup_controls_are_named"] = (
        "verify.setAttribute('aria-label'" in index
        and "restore.setAttribute('aria-label'" in index
    )
    checks["responsive_single_column_contract"] = "@media(max-width:520px)" in index
    checks["reduced_motion_contract"] = "@media(prefers-reduced-motion:no-preference)" in index
    checks["forced_colors_contract"] = (
        "@media(forced-colors:active)" in index
        and "@media(forced-colors:active)" in login
    )
    checks["minimum_touch_target_contract"] = (
        "min-block-size:44px" in index and "min-block-size:44px" in login
    )
    checks["safe_dom_rendering_contract"] = (
        "innerHTML" not in index and "document.createElement" in index
    )
    checks["no_script_failure_is_visible"] = (
        "<noscript>" in index
        and '<p role="alert" aria-live="assertive" aria-atomic="true">' in index
    )

    contrast_pairs = {
        "body_text": ("#f7f6f0", "#090d0c", 4.5),
        "muted_panel_text": ("#aab2ad", "#151b19", 4.5),
        "primary_button_text": ("#07100c", "#7ee2a8", 4.5),
        "danger_button_text": ("#07100c", "#ff7b72", 4.5),
        "danger_panel_text": ("#ff7b72", "#151b19", 4.5),
        "focus_indicator": ("#ffd166", "#0b100f", 3.0),
    }
    contrast_ratios = {
        name: round(_contrast(foreground, background), 2)
        for name, (foreground, background, _) in contrast_pairs.items()
    }
    checks["declared_color_pairs_meet_wcag_contrast"] = all(
        contrast_ratios[name] >= minimum
        for name, (_, _, minimum) in contrast_pairs.items()
    )

    with tempfile.TemporaryDirectory(prefix="gfs-web-a11y-") as temporary:
        server = create_product_web_server(temporary, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = urlopen(
                f"http://127.0.0.1:{server.server_port}/", timeout=5
            )
            served = response.read().decode("utf-8")
            checks["real_socket_serves_audited_surface"] = (
                response.status == 200
                and '<html lang="zh-CN">' in served
                and "function announce(" in served
                and "@media(forced-colors:active)" in served
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        checks["server_shutdown_cleanly"] = not thread.is_alive()

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_studio_code_level_accessibility",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_automated_contract" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "contrast_ratios": contrast_ratios,
        "external_calls_made": False,
        "matches_executed": 0,
        "limitations": [
            "source and HTTP contract checks do not expose a browser accessibility tree",
            "no screen reader, browser engine, zoom, reflow, or assistive-technology interoperability test was run",
            "this is not an external WCAG conformance audit or target-user usability study",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_web_accessibility()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

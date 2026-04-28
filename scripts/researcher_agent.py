"""The Clio Researcher Agent.

Weekly: read live performance, ask Claude for ONE actionable parameter
change, submit as draft PR + email.

Conservative by design:
- ≥20 resolved trades required to propose anything; below that, always `hold`
- Whitelisted parameters only, bounded ranges
- Per-step delta capped at 50% of allowed range
- Always opens DRAFT PR — human merges
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from clio.agents.llm_anthropic import AnthropicLLMClient
from clio.research.agent import (
    ALLOWED_PARAMS,
    LiveMetrics,
    Proposal,
    compute_live_metrics,
    parse_proposal,
    validate_proposal,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("researcher")


MIN_RESOLVED_FOR_PROPOSAL = 20


_SYSTEM_PROMPT = """You are the Clio Researcher Agent.

Each week you read live paper-trading results and propose AT MOST ONE actionable parameter change for the next week of trading.

CRITICAL DIRECTIVES:

1. **Default to `hold`.** Most weeks have insufficient evidence for a confident change. Saying "no change" is the *correct answer* unless you can cite specific numerical evidence from the LIVE data (not the backtest) for the change you propose.

2. **Statistical evidence required.** A proposal must cite at least one slice of resolved trades where:
   - n ≥ 8 trades in the slice
   - The slice's hit rate or per-trade return deviates from overall by ≥ 10 percentage points
   - The proposed parameter change directly addresses that slice

3. **Whitelisted parameters only.** You may only propose changes to parameters listed below. You must respect their bounds.

4. **Bounded delta per step.** Your proposed change cannot exceed 50% of the parameter's allowed range in a single step. Small probes beat large jumps.

5. **No structural changes.** You cannot disable risk caps, change file structure, or invent new parameters. Only the listed knobs.

6. **Output JSON only.** No markdown, no commentary outside the JSON.

OUTPUT SCHEMA (exactly this, JSON only):
{
  "decision": "propose" | "hold",
  "summary": "<one sentence describing the change OR why no change>",
  "evidence": ["<numerical claim 1>", "<numerical claim 2>", ...],
  "parameter_changes": {"<param>": {"old": <number>, "new": <number>}},
  "expected_impact": "<one sentence projection>",
  "confidence": "low" | "medium" | "high",
  "rollback": "<condition under which this change should be reverted>"
}

If decision is "hold", parameter_changes must be {} and confidence should be "low".
"""


def _build_user_prompt(metrics: LiveMetrics, params_state: dict) -> str:
    allowed_block = "\n".join(
        f'  - {k}: range [{v["min"]}, {v["max"]}], current {params_state.get(k, v["default"])}, default {v["default"]}\n      {v["description"]}'
        for k, v in ALLOWED_PARAMS.items()
    )

    return f"""LIVE STATE (paper trading):

  Bankroll: ${metrics.bankroll_initial:,.0f} → ${metrics.bankroll_current:,.0f}
  Resolved trades: {metrics.n_resolved}
  Pending trades: {metrics.n_pending}
  Max drawdown: {100 * metrics.max_drawdown_pct:.1f}%
  Min trades to propose: {MIN_RESOLVED_FOR_PROPOSAL} ({"✓ enough" if metrics.n_resolved >= MIN_RESOLVED_FOR_PROPOSAL else "✗ insufficient — must hold"})

OVERALL PERFORMANCE:
  n={metrics.overall.n}  hit_rate={100*metrics.overall.hit_rate:.1f}%
  avg_edge={metrics.overall.avg_edge:.3f}
  avg_per_trade_return={100*metrics.overall.avg_return_pct:.2f}%

BY QTYPE:
{_format_slices(metrics.by_qtype)}

BY EDGE BAND:
{_format_slices(metrics.by_edge_band)}

BY LLM CONFIDENCE:
{_format_slices(metrics.by_confidence)}

BY SIDE:
{_format_slices(metrics.by_side)}

BY DAYS HELD:
{_format_slices(metrics.by_days_held_bucket)}

BACKTEST EXPECTATION (for comparison only — not actionable evidence):
  hit_rate: {100*metrics.backtest_expectations.get('hit_rate', 0):.1f}%
  monthly_compound: {100*metrics.backtest_expectations.get('monthly_compound', 0):.2f}%/mo
  CAGR: {100*metrics.backtest_expectations.get('cagr', 0):.0f}%
  profit_factor: {metrics.backtest_expectations.get('profit_factor', 0):.2f}
  best_edge_threshold (in backtest): {metrics.backtest_expectations.get('best_edge_threshold', 0):.2f}

ALLOWED PARAMETERS (you may modify exactly one, or none):
{allowed_block}

Now decide. Output JSON only.
"""


def _format_slices(slices: dict) -> str:
    if not slices:
        return "  (no resolved trades yet)"
    lines = []
    for k, s in sorted(slices.items()):
        lines.append(
            f"  {k:<20} n={s.n:<3} hit={100*s.hit_rate:>5.1f}%  "
            f"avg_edge={s.avg_edge:.3f}  avg_ret={100*s.avg_return_pct:+.2f}%  "
            f"total_pnl_pct={100*s.total_pnl_pct:+.2f}%"
        )
    return "\n".join(lines)


def _read_current_params() -> dict[str, float]:
    """Extract the currently-deployed param values from paper_trade_scan.py."""
    src = Path("scripts/paper_trade_scan.py").read_text()
    out: dict[str, float] = {}
    patterns = {
        "edge_threshold": r"abs\(edge\)\s*<\s*([\d.]+)",
        "max_position_pct": r"size_frac\s*=\s*min\(size_frac,\s*([\d.]+)\)",
        "kelly_fraction": r"size_frac\s*=\s*([\d.]+)\s*\+\s*([\d.]+)\s*\*\s*f_star",
        "notional_floor": r"size_frac\s*=\s*([\d.]+)\s*\+\s*[\d.]+\s*\*\s*f_star",
        "min_volume": r"min_volume\s*=\s*(\d+)",
        "max_days_remaining": r"max_days\s*=\s*(\d+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, src)
        if m:
            try:
                if k == "kelly_fraction":
                    out[k] = float(m.group(2))
                else:
                    out[k] = float(m.group(1))
            except (ValueError, IndexError):
                pass
    # Fall back to defaults for missing.
    for k, v in ALLOWED_PARAMS.items():
        out.setdefault(k, v["default"])
    return out


def _apply_param_change(param: str, new_value: float) -> bool:
    """Edit paper_trade_scan.py in-place to apply the change. Returns True
    if a substitution actually happened.
    """
    path = Path("scripts/paper_trade_scan.py")
    src = path.read_text()
    new_src = src

    if param == "edge_threshold":
        new_src = re.sub(
            r"(abs\(edge\)\s*<\s*)[\d.]+",
            rf"\g<1>{new_value:.3f}",
            new_src,
            count=1,
        )
    elif param == "max_position_pct":
        new_src = re.sub(
            r"(size_frac\s*=\s*min\(size_frac,\s*)[\d.]+(\))",
            rf"\g<1>{new_value:.3f}\g<2>",
            new_src,
            count=1,
        )
    elif param == "kelly_fraction":
        new_src = re.sub(
            r"(size_frac\s*=\s*[\d.]+\s*\+\s*)[\d.]+(\s*\*\s*f_star)",
            rf"\g<1>{new_value:.2f}\g<2>",
            new_src,
            count=1,
        )
    elif param == "notional_floor":
        new_src = re.sub(
            r"(size_frac\s*=\s*)[\d.]+(\s*\+\s*[\d.]+\s*\*\s*f_star)",
            rf"\g<1>{new_value:.3f}\g<2>",
            new_src,
            count=1,
        )
    elif param == "min_volume":
        new_src = re.sub(
            r"(min_volume\s*=\s*)\d+",
            rf"\g<1>{int(new_value)}",
            new_src,
            count=1,
        )
    elif param == "max_days_remaining":
        new_src = re.sub(
            r"(max_days\s*=\s*)\d+",
            rf"\g<1>{int(new_value)}",
            new_src,
            count=1,
        )
    else:
        return False

    if new_src == src:
        return False
    path.write_text(new_src)
    return True


def _create_pr_branch_and_push(proposal: Proposal, today: date) -> str | None:
    """Create branch, commit changes, push. Returns branch name if successful."""
    branch = f"clio-researcher/{today.isoformat()}"
    try:
        subprocess.run(["git", "checkout", "-b", branch], check=True,
                       capture_output=True)
        # Apply each param change.
        applied: list[str] = []
        for param, ch in proposal.parameter_changes.items():
            if _apply_param_change(param, float(ch["new"])):
                applied.append(f"  - {param}: {ch['old']} → {ch['new']}")
            else:
                log.warning("could not apply change to %s", param)

        # Save proposal artifact too.
        prop_path = Path(f"research_proposals/{today.isoformat()}.json")
        prop_path.parent.mkdir(parents=True, exist_ok=True)
        prop_path.write_text(json.dumps({
            **proposal.to_json(),
            "proposed_at": datetime.now(tz=timezone.utc).isoformat(),
        }, indent=2))

        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        commit_msg = f"researcher: {proposal.summary[:60]}\n\n" + "\n".join(applied)
        result = subprocess.run(
            ["git", "-c", "user.name=clio-researcher",
             "-c", "user.email=actions@github.com",
             "commit", "-m", commit_msg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning("nothing to commit (no actual changes applied)")
            subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
            subprocess.run(["git", "branch", "-D", branch], check=True, capture_output=True)
            return None

        subprocess.run(["git", "push", "-u", "origin", branch], check=True,
                       capture_output=True)
        # Switch back to main locally.
        subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
        return branch
    except subprocess.CalledProcessError as exc:
        log.error("git op failed: %s\nstdout: %s\nstderr: %s",
                  exc, exc.stdout, exc.stderr)
        # Attempt cleanup
        subprocess.run(["git", "checkout", "main"], capture_output=True)
        subprocess.run(["git", "branch", "-D", branch], capture_output=True)
        return None


def _open_draft_pr(branch: str, proposal: Proposal) -> str | None:
    """Use gh CLI to open a draft PR. Returns PR URL on success."""
    body_lines = [
        "## Researcher proposal",
        "",
        f"**Summary**: {proposal.summary}",
        "",
        "**Confidence**: " + proposal.confidence,
        "",
        "**Evidence**:",
    ]
    for e in proposal.evidence:
        body_lines.append(f"- {e}")
    body_lines.extend([
        "",
        "**Parameter changes**:",
        "```json",
        json.dumps(proposal.parameter_changes, indent=2),
        "```",
        "",
        f"**Expected impact**: {proposal.expected_impact}",
        "",
        f"**Rollback**: {proposal.rollback}",
        "",
        "---",
        "",
        "_Auto-generated by `scripts/researcher_agent.py`. This is a **draft** — review and merge or close._",
    ])
    body = "\n".join(body_lines)

    try:
        result = subprocess.run(
            ["gh", "pr", "create",
             "-R", "zhongdaweiai/clio",
             "--head", branch,
             "--base", "main",
             "--title", f"researcher: {proposal.summary[:60]}",
             "--body", body,
             "--draft"],
            capture_output=True, text=True, check=True,
        )
        url = result.stdout.strip()
        return url
    except subprocess.CalledProcessError as exc:
        log.error("gh pr create failed: %s\nstderr: %s", exc, exc.stderr)
        return None


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        log.info("DRY RUN: skipping git ops, PR creation, and email")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY required")
        return 1

    pt_dir = Path("paper_trades")
    if not pt_dir.exists():
        log.error("paper_trades/ does not exist")
        return 1

    today = datetime.now(tz=timezone.utc).date()
    metrics = compute_live_metrics(pt_dir)
    log.info("live metrics: resolved=%d pending=%d bankroll=$%.0f",
             metrics.n_resolved, metrics.n_pending, metrics.bankroll_current)

    # Always save the metrics snapshot.
    snap_path = Path(f"research_proposals/snapshot_{today.isoformat()}.json")
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(metrics.to_json(), indent=2))

    # Insufficient data → skip LLM call entirely.
    if metrics.n_resolved < MIN_RESOLVED_FOR_PROPOSAL:
        log.info("insufficient data (%d < %d resolved trades), skipping proposal",
                 metrics.n_resolved, MIN_RESOLVED_FOR_PROPOSAL)
        # Still emit a hold proposal so the email goes out.
        hold = Proposal(
            decision="hold",
            summary=f"Insufficient resolved trades ({metrics.n_resolved} < {MIN_RESOLVED_FOR_PROPOSAL}). No change.",
            evidence=[f"Only {metrics.n_resolved} trades resolved so far"],
            parameter_changes={},
            expected_impact="N/A",
            confidence="low",
            rollback="N/A",
        )
        Path(f"research_proposals/{today.isoformat()}.json").write_text(
            json.dumps({**hold.to_json(), "proposed_at": datetime.now(tz=timezone.utc).isoformat(),
                        "branch": None, "pr_url": None}, indent=2)
        )
        _send_email(hold, branch=None, pr_url=None, metrics=metrics)
        return 0

    params_state = _read_current_params()
    log.info("current params: %s", params_state)

    user_prompt = _build_user_prompt(metrics, params_state)

    llm = AnthropicLLMClient(model="claude-sonnet-4-6")
    raw = llm.complete(
        f"{_SYSTEM_PROMPT}\n\n{user_prompt}",
        max_tokens=600,
        temperature=0.0,
    )
    log.info("LLM response: %s", raw[:500])

    proposal = parse_proposal(raw)
    if proposal is None:
        log.error("could not parse LLM response into a proposal")
        return 2

    is_valid, errors = validate_proposal(proposal)
    if not is_valid:
        log.warning("proposal failed validation: %s", errors)
        # Treat as hold for safety.
        proposal.decision = "hold"
        proposal.parameter_changes = {}
        proposal.summary = f"[INVALID PROPOSAL DOWNGRADED TO HOLD] {proposal.summary}"
        proposal.evidence = [f"Validation error: {e}" for e in errors] + proposal.evidence

    branch = None
    pr_url = None
    if proposal.decision == "propose" and proposal.parameter_changes and not dry_run:
        branch = _create_pr_branch_and_push(proposal, today)
        if branch:
            pr_url = _open_draft_pr(branch, proposal)
            log.info("draft PR: %s", pr_url)
    elif dry_run and proposal.decision == "propose":
        log.info("[dry-run] would create branch + PR for: %s", proposal.parameter_changes)

    # Save the final artifact.
    Path(f"research_proposals/{today.isoformat()}.json").write_text(json.dumps({
        **proposal.to_json(),
        "proposed_at": datetime.now(tz=timezone.utc).isoformat(),
        "branch": branch,
        "pr_url": pr_url,
        "validation_errors": errors if not is_valid else [],
    }, indent=2))

    if not dry_run:
        _send_email(proposal, branch, pr_url, metrics)
    else:
        log.info("[dry-run] would email: subject would be researcher proposal/hold")
    return 0


def _send_email(proposal: Proposal, branch: str | None, pr_url: str | None,
                metrics: LiveMetrics) -> None:
    sys.path.insert(0, "scripts")
    from notify import send_email  # noqa

    to = os.environ.get("NOTIFY_EMAIL")
    if not to:
        log.info("NOTIFY_EMAIL not set, skipping email")
        return

    if proposal.decision == "propose":
        subj = f"🔬 [Clio Researcher] Proposes change: {proposal.summary[:60]}"
    else:
        subj = f"🔬 [Clio Researcher] No change this week ({metrics.n_resolved} resolved)"

    pr_block = ""
    if pr_url:
        pr_block = f'<p><a href="{pr_url}" style="background:#0066cc;color:white;padding:8px 12px;text-decoration:none;border-radius:4px">📋 Review draft PR</a></p>'

    evidence_html = "".join(f"<li>{e}</li>" for e in proposal.evidence)
    changes_html = (
        '<table cellpadding="6" style="border-collapse:collapse;font-size:13px"><tr style="background:#f4f4f4">'
        '<th>Param</th><th>Old</th><th>New</th><th>Δ</th></tr>'
        + "".join(
            f'<tr><td><code>{k}</code></td><td>{v["old"]}</td>'
            f'<td><strong>{v["new"]}</strong></td><td>{float(v["new"])-float(v["old"]):+.4f}</td></tr>'
            for k, v in proposal.parameter_changes.items()
        )
        + "</table>"
    ) if proposal.parameter_changes else "<em>(no parameter changes)</em>"

    html = f"""
    <html><body style="font-family:-apple-system,sans-serif;max-width:800px;margin:auto;color:#222">
    <h2>🔬 Clio Researcher Agent — {datetime.now(tz=timezone.utc).date().isoformat()}</h2>

    <p><strong>Decision:</strong> <code>{proposal.decision}</code> &middot;
       <strong>Confidence:</strong> {proposal.confidence}</p>

    <h3>Summary</h3>
    <p>{proposal.summary}</p>

    <h3>Evidence</h3>
    <ul>{evidence_html}</ul>

    <h3>Parameter changes</h3>
    {changes_html}

    <h3>Expected impact</h3>
    <p>{proposal.expected_impact}</p>

    <h3>Rollback</h3>
    <p>{proposal.rollback}</p>

    {pr_block}

    <hr>
    <p style="color:#666;font-size:12px">
        Live state: {metrics.n_resolved} resolved trades, {metrics.n_pending} pending,
        bankroll ${metrics.bankroll_current:,.0f} ({100 * (metrics.bankroll_current/metrics.bankroll_initial - 1):+.2f}%)
    </p>
    <p style="color:#666;font-size:12px">
        Auto-generated by <code>scripts/researcher_agent.py</code>. Validates proposal against
        whitelist + bounded ranges + max-delta-per-step. Branch + draft PR opened only if
        decision=propose and validation passes.
    </p>
    </body></html>
    """

    if send_email(to, subj, html):
        log.info("researcher email sent to %s", to)


if __name__ == "__main__":
    sys.exit(main())

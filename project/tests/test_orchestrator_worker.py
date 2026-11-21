"""Orchestrator-worker: grouping plan validation, code-computed figures and reports."""

import pytest

from agents.orchestrator_worker import OrchestratorWorkerPattern as OWP

MESSAGES = [
    {"message_id": "A", "amount": "100.00 USD", "sender_bic": "AAAAUS33XXX", "receiver_bic": "BBBBGB22XXX"},
    {"message_id": "B", "amount": "200.50 USD", "sender_bic": "AAAAUS33ABC", "receiver_bic": "CCCCDEFFXXX"},
    {"message_id": "C", "amount": "300.00 EUR", "sender_bic": "DDDDFRPPXXX", "receiver_bic": "BBBBGB22XXX",
     "fraud_status": "FRAUDULENT", "fraud_decision": "REJECT"},
]


def plan_for(response, monkeypatch):
    """Run the orchestrator with a canned LLM plan."""
    orchestrator = OWP.Orchestrator()
    monkeypatch.setattr(orchestrator, "respond", lambda prompt: response)
    return orchestrator.analyze_and_create_tasks(MESSAGES)


def covered_ids(plan):
    """Message ids across all tasks, in a sorted list (duplicates would show)."""
    return sorted(m["message_id"] for task in plan["tasks"] for m in task["data"]["messages"])


def test_group_keys():
    """Currency comes from the amount; banks are the first 8 BIC characters (branches merged)."""
    assert OWP.group_key(MESSAGES[1], "currency") == "USD"
    assert OWP.group_key(MESSAGES[0], "sender_bank") == OWP.group_key(MESSAGES[1], "sender_bank") == "AAAAUS33"
    assert OWP.group_key(MESSAGES[2], "receiver_bank") == "BBBBGB22"


def test_group_figures_are_exact():
    """Totals use Decimal and unparseable amounts are skipped, not guessed."""
    figures = OWP.group_figures(MESSAGES[:2] + [{"amount": "n/a"}])
    assert figures["totals"] == {"USD": "300.50"}
    assert figures["message_count"] == 3
    assert figures["chain_decisions"] == {"NOT_REVIEWED": 3}


def test_valid_bank_plan_is_kept(monkeypatch):
    """A good plan by sender bank is used as proposed, high priority first."""
    plan = plan_for({"analysis": "x", "grouping": {"dimension": "sender_bank", "rationale": "r"},
                     "tasks": [{"group_key": "AAAAUS33", "type": "summary_report", "priority": "low"},
                               {"group_key": "DDDDFRPP", "type": "fraud_analysis", "priority": "high"}]},
                    monkeypatch)
    assert plan["grouping"]["dimension"] == "sender_bank"
    order = [(t["task_id"], t["group_key"]) for t in plan["tasks"]]
    assert order == [("task_001", "DDDDFRPP"), ("task_002", "AAAAUS33")]
    assert plan["plan_notes"] == []
    assert covered_ids(plan) == ["A", "B", "C"]


@pytest.mark.parametrize("response", [
    {"grouping": {"dimension": "country"}, "tasks": [{"group_key": "XYZ"}]},            # invalid dimension/group
    {"grouping": {"dimension": "currency"},
     "tasks": [{"group_key": "USD", "type": "hack", "priority": "urgent"}, {"group_key": "USD"}]},  # dup/invalid
    {"error": "outage"},                                                                 # LLM failure
])
def test_broken_plans_are_repaired(response, monkeypatch):
    """Whatever the LLM returns, every message is reported exactly once with valid tasks."""
    plan = plan_for(response, monkeypatch)
    assert plan["grouping"]["dimension"] in OWP.GROUPING_DIMENSIONS
    assert covered_ids(plan) == ["A", "B", "C"]
    assert all(t["type"] in OWP.TASK_TYPES and t["priority"] in OWP.PRIORITIES for t in plan["tasks"])
    assert plan["plan_notes"]


def test_workers_report_every_group(monkeypatch):
    """One worker report per group; failures are reported, not raised."""
    monkeypatch.setattr(OWP.Orchestrator, "respond", lambda self, prompt: {
        "analysis": "x", "grouping": {"dimension": "currency", "rationale": "r"}, "tasks": []})
    monkeypatch.setattr(OWP.GenericAgent, "_complete_json", lambda self, system, user: (
        {"error": "outage"} if "EUR" in user else {"title": "t", "summary": "s", "findings": [],
                                                   "flagged_messages": [], "recommendation": "r"}))
    result = OWP().process_with_orchestrator(MESSAGES)
    status = {r["group_key"]: r["status"] for r in result["task_results"]}
    assert status == {"EUR": "failed", "USD": "completed"}


def test_empty_message_set_is_handled():
    """An empty report set (e.g. no fraud found) returns without calling the LLM."""
    assert OWP().process_with_orchestrator([])["task_results"] == []

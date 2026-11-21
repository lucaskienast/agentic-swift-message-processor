"""Parallel fraud detection: agents, decisive aggregation and message marking."""

import pytest

from agents.parallelization import ParallelizationPattern
from agents.workflow_agents.base_agents import (
    FraudAggAgent,
    FraudPatternDetectionAgent,
    GeographicRiskAgent,
)


def make_message(amount, sender, receiver, remittance=None, message_type="MT202"):
    """Build a minimal message for the fraud agents."""
    return {"message_id": f"{sender}-{receiver}-{amount}", "message_type": message_type, "amount": amount,
            "sender_bic": sender, "receiver_bic": receiver, "remittance_info": remittance}


@pytest.fixture
def geo_agent(monkeypatch):
    """GeographicRiskAgent whose LLM purpose check is stubbed and call-counted."""
    agent = GeographicRiskAgent()
    agent.llm_calls = []
    agent.llm_reply = {"implausible": False, "reason": ""}
    monkeypatch.setattr(agent, "respond", lambda prompt: agent.llm_calls.append(prompt) or agent.llm_reply)
    return agent


def test_pattern_agent_handles_mt202_without_remittance():
    """MT202 messages carry remittance_info=None; the agent must not crash."""
    result = FraudPatternDetectionAgent().analyze(make_message("100.00 USD", "CHASUS33XXX", "BARCGB22XXX"))
    assert result["risk_score"] == 0


def test_geo_sanctioned_country_fast_path(geo_agent):
    """A sanctioned receiver scores 1.0 without any LLM call."""
    result = geo_agent.analyze(make_message("50000.00 USD", "DEUTDEFFXXX", "BKIRIRTHXXX", "Invoice", "MT103"))
    assert result["risk_score"] == 1.0
    assert geo_agent.llm_calls == []


@pytest.mark.parametrize("amount, sender, receiver, expected_rules", [
    ("500000.00 EUR", "DEUTDEFFXXX", "UBSWCHZHXXX", [1]),        # into secrecy jurisdiction, > 100k
    ("5000.00 CAD", "BOTKJPJTXXX", "DEUTDEFFXXX", [2]),           # third currency
    ("3000000.00 USD", "CHASUS33XXX", "BARCGB22XXX", [3]),        # cross-border > 1M
    ("923931.38 USD", "THFRSGVLHLK", "BOTKJPJTXXX", []),          # SG -> JP: receiver not secrecy
    ("923931.38 USD", "BOTKJPJTXXX", "THFRSGVLHLK", [1]),         # JP -> SG: receiver is secrecy
    ("2000000.00 CHF", "UBSWCHZHXXX", "CRESCHZZXXX", []),         # domestic, not cross-border
    ("5000.00 CAD", "ABCDZZ11XXX", "DEUTDEFFXXX", []),            # unknown country: no rule 2 guess
])
def test_geo_factual_rules_are_computed_in_code(geo_agent, amount, sender, receiver, expected_rules):
    """Rules 1-3 depend only on the data, never on the LLM (MT202: no purpose, no call)."""
    result = geo_agent.analyze(make_message(amount, sender, receiver))
    rules = [int(reason.split(":")[0].split()[1]) for reason in result["fraud_reasons"]]
    assert rules == expected_rules
    assert result["risk_score"] == round(sum(GeographicRiskAgent.RULE_POINTS[r] for r in rules), 3)
    assert geo_agent.llm_calls == []


def test_geo_rule_4_comes_from_llm(geo_agent):
    """An implausible stated purpose (rule 4) is judged by the LLM and adds its points."""
    geo_agent.llm_reply = {"implausible": True, "reason": "salary of 4.5M"}
    result = geo_agent.analyze(make_message("4500000.00 USD", "CHASUS33XXX", "DBSSSGSGXXX", "Salary", "MT103"))
    assert len(geo_agent.llm_calls) == 1
    assert result["risk_score"] == 0.7   # rules 1 + 3 + 4
    assert any("salary of 4.5M" in reason for reason in result["fraud_reasons"])


def test_geo_llm_failure_keeps_rules_and_floors_score(geo_agent):
    """If the LLM fails, rules 1-3 still count and the score is raised to 0.5 for review."""
    geo_agent.llm_reply = {"error": "outage"}
    result = geo_agent.analyze(make_message("500000.00 EUR", "DEUTDEFFXXX", "UBSWCHZHXXX", "Investment", "MT103"))
    assert result["risk_score"] == 0.5
    assert any(reason.startswith("Rule 1") for reason in result["fraud_reasons"])
    assert any("manual review" in reason for reason in result["fraud_reasons"])


@pytest.mark.parametrize("scores, fraudulent, confidence", [
    ([0.4, 0, 1.0], True, 100.0),      # decisive sanctions hit, average only 0.47
    ([0, 0, 0], False, 0.0),
    ([0.5, 0.5, 0.5], True, 50.0),     # average reaches the threshold
    ([0.3, 0.6, 0.2], False, 36.67),   # moderate, nothing decisive
])
def test_aggregator(scores, fraudulent, confidence):
    """Average-vs-threshold rule plus the decisive single-agent rule."""
    results = [{"agent": f"A{i}", "risk_score": s, "fraud_reasons": []} for i, s in enumerate(scores)]
    aggregated = FraudAggAgent().aggregate_results(results)
    assert aggregated["is_fraudulent"] is fraudulent
    assert aggregated["confidence"] == confidence


def test_parallel_stage_marks_every_message(monkeypatch):
    """Three agents run per message and the aggregator sets the fraud fields."""
    monkeypatch.setattr(GeographicRiskAgent, "respond", lambda self, prompt: {"implausible": False})
    pattern = ParallelizationPattern()
    assert len(pattern.list_of_agents) == 3
    messages = [make_message("100.00 USD", "CHASUS33XXX", "BARCGB22XXX"),
                make_message("120.37 EUR", "DEUTDEFFXXX", "BKIRIRTHXXX")]
    processed = pattern.process_batch_parallel(messages)
    status = {m["receiver_bic"]: m["fraud_status"] for m in processed}
    assert status == {"BARCGB22XXX": "CLEAN", "BKIRIRTHXXX": "FRAUDULENT"}
    for message in processed:
        assert {"fraud_status", "fraud_score", "fraud_reasons"} <= message.keys()
        assert "fraud_analysis" not in message

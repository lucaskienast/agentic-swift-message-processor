"""Prompt chain: five stages in order, grounding facts and the structured step log."""

import json

from agents.prompt_chaining import PromptChainingPattern

STAGES = ["Initial Fraud Screener", "Technical Analyst", "Risk Assessment Specialist",
          "Compliance Officer", "Final Reviewer"]


def chain_reply(message_ids):
    """Fake LLM: per-message entries for every stage, final decisions for the reviewer."""
    def reply(kwargs):
        system = kwargs["messages"][0]["content"]
        if "Final Reviewer" in system:
            return {"final_decisions": [{"message_id": mid, "decision": "APPROVE", "confidence": 90,
                                         "justification": "clean"} for mid in message_ids]}
        # Nested like the free-form risk assessor's real output
        return {"analysis": {"messages": [{"message_id": mid} for mid in message_ids]}}
    return reply


def test_chain_runs_five_stages_in_order_with_grounding(valid_message, fake_client):
    """Every stage is called in order, each system prompt starts with the shared facts."""
    pattern = PromptChainingPattern()
    pattern.client = fake_client(chain_reply(["MSG-1"]))
    messages = [dict(valid_message)]
    results = pattern.process_chain(messages)

    calls = pattern.client.calls
    assert len(calls) == 5
    for call, stage in zip(calls, STAGES):
        system = call["messages"][0]["content"]
        assert system.startswith(PromptChainingPattern.ESTABLISHED_FACTS)
        assert stage in system
    assert list(results) == ["initial_screening", "technical_analysis", "risk_assessment",
                             "compliance_review", "final_review", "chain_log"]
    assert messages[0]["fraud_decision"] == "APPROVE"


def test_chain_log_is_structured_and_never_enters_prompts(valid_message, fake_client):
    """One ok entry per stage with full coverage; the log itself is not sent to the LLM."""
    pattern = PromptChainingPattern()
    pattern.client = fake_client(chain_reply(["MSG-1"]))
    results = pattern.process_chain([dict(valid_message)])

    log = results["chain_log"]
    assert [entry["step"] for entry in log] == [1, 2, 3, 4, 5]
    assert all(entry["status"] == "ok" and entry["messages_covered"] == 1 for entry in log)
    assert not any("chain_log" in json.dumps(call) for call in pattern.client.calls)


def test_chain_log_records_llm_failures(valid_message):
    """An API failure is logged as an error per stage instead of crashing the run."""
    pattern = PromptChainingPattern()

    def broken(**kwargs):
        raise RuntimeError("outage")
    pattern.client.chat.completions.create = broken
    results = pattern.process_chain([dict(valid_message)])
    assert [entry["status"] for entry in results["chain_log"]] == ["error"] * 5
    assert results["chain_log"][0]["error"] == "outage"


def test_find_message_ids_searches_nested_output():
    """Coverage counting finds message ids at any depth."""
    output = {"risk_analysis": {"messages": [{"message_id": "A"}, {"nested": [{"message_id": "B"}]}]}}
    assert PromptChainingPattern._find_message_ids(output) == {"A", "B"}

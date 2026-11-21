"""Evaluator-Optimizer: validation rules, correction parsing and guardrails."""

from agents.evaluator_optimizer import EvaluatorOptimizerPattern
from agents.workflow_agents.base_agents import SwiftCorrectionAgent


def test_valid_message_passes(valid_message):
    """A well-formed message has no validation errors."""
    assert EvaluatorOptimizerPattern().evaluate_message(valid_message) == (True, [])


def test_each_injected_defect_is_detected(valid_message):
    """Every defect the generator injects is caught by the evaluator."""
    evaluator = EvaluatorOptimizerPattern()
    defects = {
        "reference": ("REF123456-EXTRA-REF-DATA", "Reference too long"),
        "amount": ("2500.00 usd", "Invalid currency"),
        "sender_bic": ("CHAS US 33 XXX", "Invalid sender BIC"),
    }
    for field, (bad_value, expected_error) in defects.items():
        is_valid, errors = evaluator.evaluate_message({**valid_message, field: bad_value})
        assert not is_valid
        assert any(expected_error in error for error in errors)


def test_correction_unwraps_nested_reply_and_keeps_identity(valid_message, fake_client):
    """A nested LLM reply is unwrapped, and the message_id can never be renamed."""
    agent = SwiftCorrectionAgent()
    agent.llm_service.client = fake_client(
        {"corrected_message": {**valid_message, "message_id": "CORRECTED001", "reference": "REF123456"}})
    broken = {**valid_message, "reference": "REF123456-EXTRA-REF-DATA"}
    corrected = agent.respond(broken, ["Reference too long"])
    assert corrected["reference"] == "REF123456"
    assert corrected["message_id"] == "MSG-1"


def test_correction_requests_json_mode(valid_message, fake_client):
    """The correction call uses JSON mode (TODO 8)."""
    agent = SwiftCorrectionAgent()
    agent.llm_service.client = fake_client(valid_message)
    agent.respond(valid_message, ["x"])
    assert agent.llm_service.client.calls[0]["response_format"] == {"type": "json_object"}


def test_guardrails_revert_business_data_changes(valid_message):
    """Message type and numeric amount changes are reverted; currency case fixes are kept."""
    original = {**valid_message, "message_type": "MT999", "amount": "2500.00 usd"}
    tampered = {**original, "message_type": "MT998", "amount": "2600.00 USD"}
    result = SwiftCorrectionAgent._enforce_guardrails(original, tampered)
    assert result["message_type"] == "MT999"
    assert result["amount"].split()[0] == "2500.00"

    case_fix = SwiftCorrectionAgent._enforce_guardrails(original, {**original, "amount": "2500.00 USD"})
    assert case_fix["amount"] == "2500.00 USD"


def test_loop_corrects_fixable_message(valid_message, monkeypatch):
    """A fixable message becomes VALID after one correction."""
    evaluator = EvaluatorOptimizerPattern()
    monkeypatch.setattr(evaluator.correction_agent, "respond", lambda message, errors: {**message, "reference": "REF1"})
    [result] = evaluator.process_with_evaluator_optimizer([{**valid_message, "reference": "X" * 30}])
    assert result["validation_status"] == "VALID"
    assert result["reference"] == "REF1"


def test_loop_gives_up_after_max_iterations(valid_message, monkeypatch):
    """An unfixable message ends INVALID; the final iteration only evaluates."""
    evaluator = EvaluatorOptimizerPattern()
    calls = []
    monkeypatch.setattr(evaluator.correction_agent, "respond",
                        lambda message, errors: calls.append(errors) or message)
    [result] = evaluator.process_with_evaluator_optimizer([{**valid_message, "sender_bic": "BAD"}])
    assert result["validation_status"] == "INVALID"
    assert len(calls) == evaluator.MAX_ITERATIONS - 1

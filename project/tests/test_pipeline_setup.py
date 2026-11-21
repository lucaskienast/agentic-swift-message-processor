"""Pipeline setup in main.py: message conversion, injected test data and report filters."""

import json

import pytest

from agents.evaluator_optimizer import EvaluatorOptimizerPattern
from config import Config
from main import SWIFTProcessingSystem as System
from services.swift_generator import SWIFTGenerator


@pytest.fixture
def messages():
    """Twenty generated messages converted to the pipeline dict format."""
    return [System._to_pipeline_dict(m) for m in SWIFTGenerator().generate_messages(count=20, bank_count=5)]


def test_pipeline_dicts_are_json_and_carry_currency_in_amount(messages):
    """Messages are JSON-serializable and the amount is the single source of the currency."""
    json.dumps(messages)
    for message in messages:
        assert "currency" not in message
        value, currency = message["amount"].split()
        float(value)
        assert currency.isupper()


def test_generated_messages_are_valid(messages):
    """Before any injection, generated messages pass the evaluator."""
    evaluator = EvaluatorOptimizerPattern()
    assert all(evaluator.evaluate_message(m)[0] for m in messages)


def test_injected_defects_fail_validation(messages):
    """Each injected format defect makes exactly that message invalid."""
    System._inject_format_defects(messages, 3)
    evaluator = EvaluatorOptimizerPattern()
    assert sum(not evaluator.evaluate_message(m)[0] for m in messages) == 3


def test_suspicious_scenarios_stay_valid_and_skip_defective_messages(messages):
    """Scenarios keep messages valid, use MT103 for remittance text and avoid defective messages."""
    System._inject_format_defects(messages, 3)
    defective = {m["message_id"] for m in messages if not EvaluatorOptimizerPattern().evaluate_message(m)[0]}
    System._inject_suspicious_scenarios(messages, 2)

    sanctioned = [m for m in messages if m["receiver_bic"][4:6] == "IR"]
    pressured = [m for m in messages if m["receiver_bic"].startswith("TEST")]
    assert len(sanctioned) == 1
    assert len(pressured) <= 1   # needs an MT103 candidate
    for message in pressured:
        assert message["message_type"] == "MT103"
        assert "Urgent" in message["remittance_info"]
    for message in sanctioned + pressured:
        assert message["message_id"] not in defective
        assert EvaluatorOptimizerPattern().evaluate_message(message)[0]


def test_report_filters():
    """Each named report set selects the right messages; bad amounts never crash."""
    msgs = [
        {"message_id": "clean-small", "message_type": "MT103", "amount": "500.00 USD", "fraud_status": "CLEAN"},
        {"message_id": "clean-big", "message_type": "MT202", "amount": "75000.00 EUR", "fraud_status": "CLEAN"},
        {"message_id": "fraud", "message_type": "MT103", "amount": "60000.00 GBP", "fraud_status": "FRAUDULENT"},
        {"message_id": "bad-amount", "message_type": "MT202", "amount": "n/a", "fraud_status": "CLEAN"},
    ]
    expected = {
        "clean": ["clean-small", "clean-big", "bad-amount"],
        "fraudulent": ["fraud"],
        "high_value": ["clean-big", "fraud"],
        "mt103": ["clean-small", "fraud"],
    }
    for name, (_, keep) in System.REPORT_FILTERS.items():
        assert [m["message_id"] for m in msgs if keep(m, Config)] == expected[name]


def test_default_report_sets_are_two_distinct_filters():
    """By default every run produces the clean and the fraudulent report set."""
    assert Config.REPORT_SETS == ("clean", "fraudulent")

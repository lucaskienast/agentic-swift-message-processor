"""
Base Agent Classes for SWIFT Transaction Processing

This module contains the base classes that all agents inherit from.
You will implement the BaseAgent abstract class and the SwiftCorrectionAgent.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from config import Config
from services.llm_service import LLMService

# TODO 6: Create BaseAgent abstract class (10 points)
# INSTRUCTIONS:
# 1. Import ABC and abstractmethod from the abc module
# 2. Create a BaseAgent class that inherits from ABC
# 3. The class should have:
#    - An __init__ method that initializes self.config and self.llm_service
#    - An abstract method create_prompt(self, data) that returns a string
#    - A concrete method respond(self, prompt) that calls the LLM service
#
# HINT: Look at how the solution implements agent classes in the lessons
# EXAMPLE STRUCTURE:
"""
from abc import ABC, abstractmethod
from config import Config
from services.llm_service import LLMService

class BaseAgent(ABC):
    def __init__(self):
        self.config = Config()
        self.llm_service = # Initialize LLMService here

    @abstractmethod
    def create_prompt(self, data):
        '''Each agent must implement your own prompt creation'''
        pass

    def respond(self, prompt: str):
        '''Common method to get LLM response'''
        # Use self.llm_service to get response
        pass
"""

# YOUR CODE HERE - Implement BaseAgent class


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-backed agents.

    It centralizes how agents talk to the LLM service: every subclass gets a
    shared Config, an LLMService (OpenAI client + model name) and a single
    JSON-mode completion call. Subclasses only decide *what* to ask by
    implementing create_prompt().
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant.",
                 temperature: float = 0.1):
        """
        Initialize the agent with configuration and the LLM service.

        Args:
            system_prompt: Role/instructions sent as the system message.
            temperature: Sampling temperature; low values keep financial
                decisions consistent between runs.
        """
        self.config = Config()
        self.llm_service = LLMService()
        self.system_prompt = system_prompt
        self.temperature = temperature

    @abstractmethod
    def create_prompt(self, data: Any) -> str:
        """Build the user prompt for this agent from the given input data."""

    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Send the prompt to the LLM using this agent's system prompt.

        Args:
            prompt: User prompt, usually produced by create_prompt().

        Returns:
            The parsed JSON response, or {"error": ...} if the call fails.
        """
        return self._complete_json(self.system_prompt, prompt)

    def _complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Run one JSON-mode chat completion and parse the result.

        Errors are caught and returned as {"error": ...} so a single failed
        call never crashes the processing pipeline.
        """
        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature
            )
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {e}")
            return {"error": str(e)}


class SwiftCorrectionAgent:
    """Agent for correcting SWIFT messages based on validation errors."""

    def __init__(self):
        # TODO 7: Define LLMService (5 points)
        # INSTRUCTIONS: Initialize self.llm_service with an instance of LLMService
        # HINT: from services.llm_service import LLMService
        # Then: self.llm_service = LLMService()

        # Shared OpenAI client + model name (API key loaded from .env via Config)
        self.llm_service = LLMService()

    def create_prompt(self, message, errors):
        """
        Create a prompt for the LLM to correct a SWIFT message.

        Args:
            message: The SWIFT message data
            errors: List of validation errors to fix

        Returns:
            str: The formatted prompt for the LLM
        """
        system_prompt = """You are a SWIFT message correction expert.
        Fix the validation errors while maintaining the business intent.
        Return the corrected message in JSON format.

        Correction rules:
        - Make the smallest possible change that resolves each listed error.
        - Never invent, guess or replace identifiers or business data. Keep
          every character of a BIC (only remove whitespace or fix letter case),
          and never change the numeric amount, currency, message_type or
          message_id.
        - Shorten an over-long reference by keeping only its first 12
          characters.
        - If an error cannot be fixed without guessing (e.g. an amount outside
          the allowed range or an unknown BIC), leave that field unchanged so
          the message is routed to manual review.
        - Leave every field that has no error unchanged."""

        user_prompt = f"""
        Original SWIFT Message:
        {message}

        Validation Errors to Fix:
        {errors}

        Please correct these errors and return the complete corrected message in JSON format.
        """

        return system_prompt, user_prompt

    def respond(self, message, errors):
        """
        Get LLM response to correct the SWIFT message.

        Args:
            message: The SWIFT message to correct
            errors: The validation errors to fix

        Returns:
            dict: The corrected message data
        """
        system_prompt, user_prompt = self.create_prompt(message, errors)

        try:
            # Use the LLMService from TODO 7 instead of an ad-hoc client
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # TODO 8: Set response format (3 points)
                # INSTRUCTIONS: Add the response_format parameter to ensure JSON output
                # The format should be: response_format={"type": "json_object"}
                # This ensures the LLM returns valid JSON that we can parse
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # TODO 9: Parse response result (2 points)
            # INSTRUCTIONS: Extract and parse the JSON content from the response
            # HINT: The response content is in response.choices[0].message.content
            # You'll need to:
            # 1. Get the content from response.choices[0].message.content
            # 2. Use json.loads() to parse it into a Python dictionary
            # 3. Return the parsed result

            # Example:
            # content = response.choices[0].message.content
            # result = json.loads(content)
            # return result

            content = response.choices[0].message.content
            result = json.loads(content or "{}")

            # The LLM sometimes nests the message, e.g. {"corrected_message": {...}}
            if len(result) == 1:
                (only_value,) = result.values()
                if isinstance(only_value, dict):
                    result = only_value

            # The message identity is never the LLM's to change
            result.pop("message_id", None)

            # Overlay the corrections on the original message so fields the LLM
            # left out (message_id, statuses, timestamps) are never lost
            return self._enforce_guardrails(message, {**message, **result})

        except Exception as e:
            print(f"Error in SwiftCorrectionAgent: {e}")
            return message  # Return original if correction fails

    @staticmethod
    def _enforce_guardrails(original: Dict, corrected: Dict) -> Dict:
        """
        Deterministically undo corrections that would change business data.

        The prompt asks the LLM not to touch these fields, but a prompt is not
        a guarantee, so the message type and the numeric amount are restored
        if the LLM altered them. Fixing the amount's currency case is allowed.

        Args:
            original: Message before correction.
            corrected: Message after the LLM's corrections were applied.

        Returns:
            The corrected message with protected fields restored.
        """
        def numeric_amount(amount: Any) -> float:
            token = str(amount).split()[0] if str(amount).split() else ""
            return float("".join(c for c in token if c.isdigit() or c == "."))

        if corrected.get("message_type") != original.get("message_type"):
            print(f"  Guardrail: reverted message_type change "
                  f"{original.get('message_type')} -> {corrected.get('message_type')}")
            corrected["message_type"] = original.get("message_type")

        try:
            amount_changed = numeric_amount(corrected.get("amount")) != numeric_amount(original.get("amount"))
        except ValueError:
            amount_changed = True
        if amount_changed:
            print(f"  Guardrail: reverted amount change "
                  f"{original.get('amount')} -> {corrected.get('amount')}")
            corrected["amount"] = original.get("amount")

        return corrected


class FraudAmountDetectionAgent:
    """Agent for detecting fraud based on transaction amounts."""

    def __init__(self):
        self.rules = [
            {"condition": "amount > 10000", "risk_score": 0.3},
            {"condition": "round_amount", "risk_score": 0.2},
            {"condition": "unusual_precision", "risk_score": 0.1}
        ]

    def analyze(self, message):
        """
        Analyze a SWIFT message for amount-based fraud patterns.

        Args:
            message: The SWIFT message to analyze

        Returns:
            dict: Fraud analysis results with risk score and reasons
        """
        risk_score = 0
        fraud_reasons = []

        try:
            # Extract amount from message
            amount_str = message.get('amount', '0')
            # Remove currency code and convert to float
            amount = float(''.join(c for c in amount_str if c.isdigit() or c == '.'))

            # Rule 1: Large amounts
            if amount > 10000:
                risk_score += 0.3
                fraud_reasons.append(f"High amount transaction: {amount}")

            # Rule 2: Round amounts (multiples of 1000)
            if amount % 1000 == 0 and amount > 0:
                risk_score += 0.2
                fraud_reasons.append(f"Suspiciously round amount: {amount}")

            # Rule 3: Unusual precision for large amounts
            if amount > 100000 and (amount % 1) != 0:
                risk_score += 0.1
                fraud_reasons.append("Large amount with unusual decimal precision")

        except (ValueError, TypeError) as e:
            print(f"Error analyzing amount: {e}")

        return {
            "agent": "FraudAmountDetectionAgent",
            "risk_score": min(risk_score, 1.0),
            "fraud_reasons": fraud_reasons
        }


class FraudPatternDetectionAgent:
    """Agent for detecting fraud based on transaction patterns."""

    def __init__(self):
        self.high_risk_patterns = ['TEST', 'FAKE', 'DEMO', '999', '000000']
        self.suspicious_keywords = ['urgent', 'immediately', 'secret', 'confidential']

    def analyze(self, message):
        """
        Analyze a SWIFT message for pattern-based fraud indicators.

        Args:
            message: The SWIFT message to analyze

        Returns:
            dict: Fraud analysis results with risk score and reasons
        """
        risk_score = 0
        fraud_reasons = []

        # Check BIC codes for test patterns
        sender_bic = message.get('sender_bic', '')
        receiver_bic = message.get('receiver_bic', '')

        for pattern in self.high_risk_patterns:
            if pattern in sender_bic.upper() or pattern in receiver_bic.upper():
                risk_score += 0.4
                fraud_reasons.append(f"Test/fake pattern detected in BIC: {pattern}")

        # Check for same sender and receiver
        if sender_bic and sender_bic == receiver_bic:
            risk_score += 0.5
            fraud_reasons.append("Same sender and receiver BIC")

        # Check remittance info for suspicious keywords
        # MT202 messages carry remittance_info=None, so fall back to ''
        remittance = (message.get('remittance_info') or '').lower()
        for keyword in self.suspicious_keywords:
            if keyword in remittance:
                risk_score += 0.2
                fraud_reasons.append(f"Suspicious keyword in remittance: {keyword}")

        return {
            "agent": "FraudPatternDetectionAgent",
            "risk_score": min(risk_score, 1.0),
            "fraud_reasons": fraud_reasons
        }


class FraudAggAgent:
    """Agent for aggregating fraud detection results from multiple agents."""

    def __init__(self):
        self.threshold = 0.5  # Fraud threshold (50%)

    def aggregate_results(self, fraud_results):
        """
        Aggregate fraud detection results from multiple agents.

        Args:
            fraud_results: List of fraud detection results from different agents

        Returns:
            dict: Aggregated fraud assessment
        """
        if not fraud_results:
            return {
                "is_fraudulent": False,
                "confidence": 0,
                "total_risk_score": 0,
                "aggregated_reasons": []
            }

        # Calculate average risk score
        total_risk = sum(r.get('risk_score', 0) for r in fraud_results)
        avg_risk = total_risk / len(fraud_results)

        # Aggregate all fraud reasons
        all_reasons = []
        for result in fraud_results:
            agent_name = result.get('agent', 'Unknown')
            reasons = result.get('fraud_reasons', [])
            for reason in reasons:
                all_reasons.append(f"[{agent_name}] {reason}")

        # Determine if fraudulent based on threshold
        is_fraudulent = avg_risk >= self.threshold

        return {
            "is_fraudulent": is_fraudulent,
            "confidence": round(avg_risk * 100, 2),
            "total_risk_score": round(avg_risk, 3),
            "aggregated_reasons": all_reasons
        }


if __name__ == "__main__":
    # Self-test (makes a real OpenAI call). Run from the project/ folder:
    #   uv run python -m agents.workflow_agents.base_agents

    class BicDecoderAgent(BaseAgent):
        """Minimal concrete BaseAgent used to exercise the LLM round trip."""

        def __init__(self):
            super().__init__(
                system_prompt="You are a SWIFT reference-data expert. Respond in JSON."
            )

        def create_prompt(self, data: Any) -> str:
            """Ask the LLM to decode the structure of a BIC code."""
            return (
                f"Decode the BIC '{data}'. Return JSON with keys "
                '"bank_code", "country_code", "location_code", "branch_code".'
            )

    print("Testing BaseAgent...")
    try:
        BaseAgent()
    except TypeError as e:
        print(f"  OK - BaseAgent is abstract: {e}")

    agent = BicDecoderAgent()
    result = agent.respond(agent.create_prompt("DEUTDEFFXXX"))
    print(f"  LLM response: {result}")
    expected = {"bank_code": "DEUT", "country_code": "DE",
                "location_code": "FF", "branch_code": "XXX"}
    print(f"  Matches expected decoding: {result == expected}")

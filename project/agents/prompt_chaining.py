"""
Prompt Chaining Pattern for Multi-Stage Fraud Analysis

This module implements a chain of specialized agents for comprehensive fraud analysis.
Each agent in the chain builds upon the insights from previous agents.
"""

import json
import time
from typing import Dict, List, Any
from openai import OpenAI
from config import Config


class PromptChainingPattern:
    """
    Implements a chain of agents for progressive fraud analysis.
    Each agent adds your specialized analysis to build a comprehensive assessment.
    """

    # Facts the earlier pipeline stages already established. Prepended to every
    # agent's system prompt: without it, real runs showed gpt-4o calling every
    # well-formed but unfamiliar BIC "invalid" and holding clean payments.
    ESTABLISHED_FACTS = """ESTABLISHED FACTS - treat these as verified, do not contradict them:
- Every message already passed deterministic SWIFT format validation
  (Evaluator-Optimizer): BICs are 8 or 11 characters in the AAAA BB CC (DDD)
  format, references are at most 16 characters, amounts and currencies are
  valid. Do not report format, BIC or reference violations.
- You have no BIC directory or KYC records, so you cannot check whether a bank
  exists. A well-formed BIC you do not recognize is NOT invalid, fake or
  suspicious. Missing information you cannot check is not a red flag.
- fraud_status, fraud_score and fraud_reasons on each message come from the
  parallel fraud detection agents. They are the evidence base: build on them.
- A reason mentioning a "High-risk/sanctioned jurisdiction" is a confirmed
  sanctions hit (sanctions risk: confirmed). Sanctions risk is confirmed ONLY
  in that case; other red flags, however serious, do not make it confirmed.
- A "Test/fake pattern detected in BIC" reason is a confirmed fraud red flag.
- Clean payments should flow straight through: approve messages without a
  concrete red flag, and only hold or reject with specific evidence.

"""

    def __init__(self):
        """Initialize the prompt chaining pattern with OpenAI client."""
        self.config = Config()
        # Key and model come from Config (loaded from project/.env)
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY or None)
        self.model = self.config.OPENAI_MODEL
        self.temperature = 0.1  # Low temperature for consistent analysis
        # Timing and error of the most recent LLM call, read by _record_step
        self._last_call = {"seconds": 0.0, "error": None}

    def _create_initial_screener_prompt(self, messages: List[Dict]) -> tuple:
        """
        Create prompt for the initial screening agent.

        Args:
            messages: List of SWIFT messages to screen

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are an Initial Fraud Screener specializing in rapid triage of SWIFT transactions.
        Your role is to quickly categorize transactions into risk levels.

        For each transaction, assign:
        - GREEN: Low risk, standard processing
        - YELLOW: Medium risk, needs review
        - RED: High risk, immediate attention

        Consider: amounts, BIC codes, countries, and obvious red flags.
        Return your analysis in JSON format."""

        user_prompt = f"""Perform initial screening on these SWIFT messages:
        {json.dumps(messages, indent=2)}

        Return JSON with structure:
        {{
            "screening_results": [
                {{
                    "message_id": "...",
                    "risk_level": "GREEN|YELLOW|RED",
                    "initial_flags": ["..."],
                    "recommended_action": "..."
                }}
            ],
            "summary": "Overall batch assessment"
        }}"""

        return system_prompt, user_prompt

    def _create_technical_analyst_prompt(self, messages: List[Dict], initial_screening: Dict) -> tuple:
        """
        Create prompt for the technical analyst agent.

        Args:
            messages: Original SWIFT messages
            initial_screening: Results from initial screener

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a Technical Analyst specializing in SWIFT message format validation.
        Review the initial screening results and perform deep technical analysis.

        Focus on:
        - SWIFT format compliance (MT103/MT202 standards)
        - BIC code validation and legitimacy
        - Amount format and currency validation
        - Reference number patterns
        - Date format compliance

        Build upon the initial screening to identify technical anomalies.
        Return your analysis in JSON format."""

        user_prompt = f"""Review these SWIFT messages with initial screening results:

        Messages: {json.dumps(messages, indent=2)}

        Initial Screening: {json.dumps(initial_screening, indent=2)}

        Perform technical validation and return JSON with:
        {{
            "technical_analysis": [
                {{
                    "message_id": "...",
                    "format_compliance": true/false,
                    "bic_validation": {{"sender": "status", "receiver": "status"}},
                    "technical_issues": ["..."],
                    "risk_adjustment": "increase|maintain|decrease",
                    "technical_score": 0-100
                }}
            ],
            "technical_summary": "Overall technical assessment"
        }}"""

        return system_prompt, user_prompt

    def _create_compliance_officer_prompt(self, messages: List[Dict], chain_results: Dict) -> tuple:
        """
        Create prompt for the compliance officer agent.

        Args:
            messages: Original SWIFT messages
            chain_results: All previous analysis results

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a Compliance Officer specializing in AML and regulatory compliance.
        Review all previous analysis and assess regulatory compliance risks.

        Focus on:
        - AML (Anti-Money Laundering) red flags
        - Sanctions screening indicators
        - PEP (Politically Exposed Persons) risks
        - Regulatory reporting requirements
        - KYC (Know Your Customer) concerns

        Consider the complete analysis chain to make compliance determinations.
        Return your analysis in JSON format."""

        user_prompt = f"""Review these SWIFT messages with complete analysis chain:

        Messages: {json.dumps(messages, indent=2)}

        Analysis Chain Results: {json.dumps(chain_results, indent=2)}

        Perform compliance assessment and return JSON with:
        {{
            "compliance_review": [
                {{
                    "message_id": "...",
                    "aml_risk": "low|medium|high",
                    "sanctions_risk": "clear|potential|confirmed",
                    "compliance_issues": ["..."],
                    "required_actions": ["..."],
                    "compliance_score": 0-100
                }}
            ],
            "compliance_summary": "Overall compliance assessment",
            "escalation_required": true/false
        }}"""

        return system_prompt, user_prompt

    def _create_final_reviewer_prompt(self, messages: List[Dict], complete_chain: Dict) -> tuple:
        """
        Create prompt for the final review agent.

        Args:
            messages: Original SWIFT messages
            complete_chain: All analysis results from the chain

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are the Final Reviewer responsible for synthesizing all analysis.
        Make the final fraud determination based on the complete analysis chain.

        Review all findings from:
        1. Initial Screening
        2. Technical Analysis
        3. Risk Assessment (if completed)
        4. Compliance Review

        Make final decisions: APPROVE, HOLD, or REJECT each transaction.
        Provide clear justification based on the accumulated evidence.
        Return your analysis in JSON format."""

        user_prompt = f"""Make final determinations based on complete analysis:

        Messages: {json.dumps(messages, indent=2)}

        Complete Analysis Chain: {json.dumps(complete_chain, indent=2)}

        Return final decisions in JSON:
        {{
            "final_decisions": [
                {{
                    "message_id": "...",
                    "decision": "APPROVE|HOLD|REJECT",
                    "confidence": 0-100,
                    "key_factors": ["..."],
                    "justification": "...",
                    "follow_up_required": ["..."]
                }}
            ],
            "batch_summary": {{
                "approved": 0,
                "held": 0,
                "rejected": 0,
                "overall_risk": "low|medium|high"
            }}
        }}"""

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Make a call to the LLM with the given prompts.

        Every agent's system prompt is prefixed with ESTABLISHED_FACTS so all
        stages of the chain share the same verified evidence base.

        Args:
            system_prompt: System role prompt
            user_prompt: User message prompt

        Returns:
            Parsed JSON response from the LLM
        """
        started = time.perf_counter()
        self._last_call = {"seconds": 0.0, "error": None}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.ESTABLISHED_FACTS + system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature
            )

            return json.loads(response.choices[0].message.content or "{}")

        except Exception as e:
            print(f"Error calling LLM: {e}")
            self._last_call["error"] = str(e)
            return {}
        finally:
            self._last_call["seconds"] = round(time.perf_counter() - started, 2)

    def _record_step(self, chain_log: List[Dict], step: int, agent: str, output_key: str,
                     result: Dict, messages: List[Dict]) -> None:
        """
        Append one structured entry to the chain log and print a one-line status.

        Args:
            chain_log: The log being built for this chain run
            step: Position of the agent in the chain (1-5)
            agent: Human-readable agent role
            output_key: Key under which the result is stored in chain_results
            result: The agent's parsed JSON output
            messages: The messages under analysis (to measure coverage)
        """
        message_ids = {m.get('message_id') for m in messages}
        covered = set()
        for value in result.values():
            if isinstance(value, list):
                covered |= {e.get('message_id') for e in value if isinstance(e, dict)} & message_ids
        summary = next((result[k] for k in ("summary", "technical_summary", "compliance_summary",
                                            "batch_summary") if k in result), "")
        status = "error" if self._last_call["error"] else ("ok" if result else "empty")
        chain_log.append({
            "step": step,
            "agent": agent,
            "output_key": output_key,
            "status": status,
            "duration_seconds": self._last_call["seconds"],
            "messages_covered": len(covered),
            "messages_total": len(message_ids),
            "summary": str(summary)[:200],
            "error": self._last_call["error"],
        })
        print(f"  -> {status} in {self._last_call['seconds']}s, "
              f"covered {len(covered)}/{len(message_ids)} messages")

    @staticmethod
    def _print_chain_log(chain_log: List[Dict]) -> None:
        """Print the structured chain log as a compact table."""
        print("\nPrompt chain log:")
        print(f"  {'#':<2} {'Agent':<34} {'Output key':<20} {'Status':<6} {'Time':>6} {'Covered':>8}")
        for entry in chain_log:
            covered = f"{entry['messages_covered']}/{entry['messages_total']}"
            print(f"  {entry['step']:<2} {entry['agent']:<34} {entry['output_key']:<20} "
                  f"{entry['status']:<6} {entry['duration_seconds']:>5}s {covered:>8}")

    def process_chain(self, messages: List[Dict]) -> Dict:
        """
        Process messages through the complete prompt chain.

        Args:
            messages: List of SWIFT messages to analyze

        Returns:
            Complete analysis results from all agents in the chain, plus a
            structured 'chain_log' with one entry per step (agent, status,
            duration, message coverage, summary)
        """
        print("Starting Prompt Chaining Analysis...")
        chain_results = {}
        # Structured log of every step; attached to the results only at the end,
        # so it never leaks into the agents' prompts
        chain_log = []

        # Step 1: Initial Screening
        print("Step 1: Initial Screener analyzing messages...")
        system_prompt, user_prompt = self._create_initial_screener_prompt(messages)
        initial_results = self._call_llm(system_prompt, user_prompt)
        chain_results['initial_screening'] = initial_results
        self._record_step(chain_log, 1, "Initial Screener (junior analyst)", "initial_screening",
                          initial_results, messages)

        # TODO 13: Implement Step 2 (7 points)
        # INSTRUCTIONS:
        # 1. Call the technical analyst agent
        # 2. Use _create_technical_analyst_prompt to create prompts
        # 3. Pass messages and initial_results as parameters
        # 4. Use _call_llm to get the response
        # 5. Store the results in chain_results['technical_analysis']
        #
        # EXAMPLE:
        # print("Step 2: Technical Analyst reviewing messages...")
        # system_prompt, user_prompt = self._create_technical_analyst_prompt(messages, initial_results)
        # technical_results = self._call_llm(system_prompt, user_prompt)
        # chain_results['technical_analysis'] = technical_results

        # YOUR CODE HERE - Implement Step 2: Technical Analyst
        print("Step 2: Technical Analyst reviewing messages...")
        system_prompt, user_prompt = self._create_technical_analyst_prompt(messages, initial_results)
        technical_results = self._call_llm(system_prompt, user_prompt)
        chain_results['technical_analysis'] = technical_results
        self._record_step(chain_log, 2, "Technical Analyst", "technical_analysis", technical_results, messages)

        # Step 3: Risk Assessor (Provided as example)
        print("Step 3: Risk Assessor evaluating patterns...")
        # This step is implemented for you as an example
        risk_prompt_system = """You are a Risk Assessment Specialist.
        Analyze behavioral patterns and transaction risks based on previous findings.
        Focus on velocity, patterns, and behavioral anomalies."""

        risk_prompt_user = f"""Assess risk based on analysis so far:
        Messages: {json.dumps(messages, indent=2)}
        Current Analysis: {json.dumps(chain_results, indent=2)}

        Return JSON with risk scores and pattern analysis."""

        risk_results = self._call_llm(risk_prompt_system, risk_prompt_user)
        chain_results['risk_assessment'] = risk_results
        self._record_step(chain_log, 3, "Risk Assessor", "risk_assessment", risk_results, messages)

        # TODO 14: Implement Step 4 (8 points)
        # INSTRUCTIONS:
        # 1. Call the compliance officer agent
        # 2. Use _create_compliance_officer_prompt to create prompts
        # 3. Pass messages and chain_results (which now has 3 stages of analysis)
        # 4. Use _call_llm to get the response
        # 5. Store the results in chain_results['compliance_review']
        #
        # EXAMPLE:
        # print("Step 4: Compliance Officer reviewing for regulatory issues...")
        # system_prompt, user_prompt = self._create_compliance_officer_prompt(messages, chain_results)
        # compliance_results = self._call_llm(system_prompt, user_prompt)
        # chain_results['compliance_review'] = compliance_results

        # YOUR CODE HERE - Implement Step 4: Compliance Officer
        print("Step 4: Compliance Officer reviewing for regulatory issues...")
        system_prompt, user_prompt = self._create_compliance_officer_prompt(messages, chain_results)
        compliance_results = self._call_llm(system_prompt, user_prompt)
        chain_results['compliance_review'] = compliance_results
        self._record_step(chain_log, 4, "Compliance Officer", "compliance_review", compliance_results, messages)

        # Step 5: Final Reviewer (Provided)
        print("Step 5: Final Reviewer making decisions...")
        system_prompt, user_prompt = self._create_final_reviewer_prompt(messages, chain_results)
        final_results = self._call_llm(system_prompt, user_prompt)
        chain_results['final_review'] = final_results
        self._record_step(chain_log, 5, "Final Reviewer", "final_review", final_results, messages)

        # Update messages with final decisions
        if 'final_decisions' in final_results:
            decision_map = {d['message_id']: d for d in final_results['final_decisions']}
            for message in messages:
                msg_id = message.get('message_id')
                if msg_id in decision_map:
                    decision = decision_map[msg_id]
                    message['fraud_decision'] = decision['decision']
                    message['fraud_confidence'] = decision.get('confidence', 0)
                    message['fraud_justification'] = decision.get('justification', '')

        chain_results['chain_log'] = chain_log
        self._print_chain_log(chain_log)
        print("Prompt Chaining Analysis Complete!")
        return chain_results

    def test_chain(self):
        """
        Test the prompt chain with sample messages.
        You can use this to verify your implementations.
        """
        test_messages = [
            {
                'message_id': 'TEST001',
                'message_type': 'MT103',
                'amount': '50000.00 USD',
                'sender_bic': 'CHASUS33XXX',
                'receiver_bic': 'DEUTGB22XXX',
                'reference': 'REF123456',
                'remittance_info': 'Invoice payment for services'
            },
            {
                'message_id': 'TEST002',
                'message_type': 'MT103',
                'amount': '999999.99 USD',
                'sender_bic': 'TESTUS33XXX',
                'receiver_bic': 'FAKEGB22XXX',
                'reference': 'URGENT999',
                'remittance_info': 'Urgent confidential transfer'
            }
        ]

        print("Testing Prompt Chain with sample messages:")
        results = self.process_chain(test_messages)

        # Print summary of results
        print("\n=== Chain Results Summary ===")
        for stage, data in results.items():
            print(f"\n{stage.upper()}:")
            if isinstance(data, dict):
                # Print key findings from each stage
                if 'summary' in data:
                    print(f"  Summary: {data['summary']}")
                elif 'batch_summary' in data:
                    print(f"  Batch Summary: {data['batch_summary']}")

        return results


if __name__ == "__main__":
    # Test the prompt chaining pattern
    # Note: Requires OPENAI_API_KEY environment variable
    pattern = PromptChainingPattern()
    pattern.test_chain()

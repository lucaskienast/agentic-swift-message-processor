"""
SWIFT Transaction Processing System with Agent Patterns
Main application entry point

This is the main integration point where all agent patterns work together
to process SWIFT messages through a complete pipeline.
"""

import random
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from models.swift_message import SWIFTMessage
from services.swift_generator import SWIFTGenerator

# Import the agent patterns you'll be using
from agents.evaluator_optimizer import EvaluatorOptimizerPattern
from agents.parallelization import ParallelizationPattern
from agents.orchestrator_worker import OrchestratorWorkerPattern
from agents.prompt_chaining import PromptChainingPattern


class SWIFTProcessingSystem:
    """Main system orchestrating all agent patterns for SWIFT processing"""

    def __init__(self):
        self.config = Config()
        self.swift_generator = SWIFTGenerator()

        # Initialize agent patterns
        # NOTE: These classes are scaffolded in the agents folder
        # You'll need to complete the TODOs in each file for them to work properly
        self.evaluator_optimizer = EvaluatorOptimizerPattern()
        self.parallelization_agent = ParallelizationPattern()
        self.orchestrator_worker = OrchestratorWorkerPattern()
        self.prompt_chaining_agent = PromptChainingPattern()
    
    def generate_swift_messages(self) -> List[Dict]:
        """
        Generate SWIFT messages and convert them into pipeline dictionaries.

        The generator returns pydantic SWIFTMessage objects, but every agent
        pattern works on plain, JSON-serializable dicts, so each message is
        converted with _to_pipeline_dict before entering the pipeline.
        """
        swift_messages = self.swift_generator.generate_messages(
            count=self.config.MESSAGE_COUNT,
            bank_count=self.config.BANK_COUNT
        )
        messages = [self._to_pipeline_dict(message) for message in swift_messages]
        self._inject_format_defects(messages, self.config.MALFORMED_MESSAGE_COUNT)
        self._inject_suspicious_scenarios(messages, self.config.SUSPICIOUS_MESSAGE_COUNT)
        return messages

    @staticmethod
    def _inject_format_defects(messages: List[Dict], count: int) -> None:
        """
        Introduce deliberate, fixable format errors into a few messages.

        Real SWIFT traffic contains malformed messages, but the generator only
        produces valid ones. Each defect below violates a rule checked by the
        Evaluator-Optimizer and can be repaired without guessing business data:
        - long_reference: reference exceeds the 16-character SWIFT limit
        - lowercase_currency: currency code in the amount is not uppercase
        - spaced_bic: sender BIC typed in its human-readable grouped form
          ("AAAA BB CC DDD"), which is not a valid 8/11-character BIC

        Defects that would force the LLM to guess data (e.g. a truncated
        BIC) are avoided, because a guessed counterparty BIC is unacceptable.
        The defect is only logged, not stored on the message, so the
        correction agent has to work it out from the validation errors.
        """
        defects = ["long_reference", "lowercase_currency", "spaced_bic"]
        targets = random.sample(messages, min(count, len(messages)))

        for index, message in enumerate(targets):
            defect = defects[index % len(defects)]
            if defect == "long_reference":
                message["reference"] = f"{message['reference']}-EXTRA-REF-DATA"
            elif defect == "lowercase_currency":
                value, currency = message["amount"].split()
                message["amount"] = f"{value} {currency.lower()}"
            else:
                bic = message["sender_bic"]
                message["sender_bic"] = f"{bic[:4]} {bic[4:6]} {bic[6:8]} {bic[8:]}".strip()
            print(f"  Injected defect '{defect}' into message {message['message_id']}")

    @staticmethod
    def _inject_suspicious_scenarios(messages: List[Dict], count: int) -> None:
        """
        Turn a few well-formed messages into realistic suspicious payments.

        The generator only produces ordinary traffic, so the fraud agents would
        almost never have anything to flag. Each scenario keeps the message
        valid (it must pass the Evaluator-Optimizer) and alternates between:
        - sanctioned_corridor: the receiver bank is moved to Iran (country
          code IR), which GeographicRiskAgent treats as a sanctions hit
        - test_bank_pressure: the receiver BIC uses a TEST bank code and the
          remittance text pushes for an urgent, confidential payment, which
          FraudPatternDetectionAgent picks up (MT103 only, because MT202
          bank-to-bank transfers carry no remittance information)

        Messages that already carry a format defect are left alone, so each
        message tells one story. The scenario is only logged, not stored on
        the message, so the agents have to detect it themselves.
        """
        reference_limit = 16
        candidates = [
            m for m in messages
            if len(m["reference"]) <= reference_limit
            and m["amount"].split()[-1].isupper()
            and " " not in m["sender_bic"]
        ]
        random.shuffle(candidates)
        scenarios = ["sanctioned_corridor", "test_bank_pressure"]

        for index in range(count):
            scenario = scenarios[index % len(scenarios)]
            eligible = [m for m in candidates
                        if scenario == "sanctioned_corridor" or m["message_type"] == "MT103"]
            if not eligible:
                print(f"  No eligible message for scenario '{scenario}', skipped")
                continue
            message = eligible[0]
            candidates.remove(message)
            bic = message["receiver_bic"]
            if scenario == "sanctioned_corridor":
                message["receiver_bic"] = f"{bic[:4]}IR{bic[6:]}"
            else:
                message["receiver_bic"] = f"TEST{bic[4:]}"
                message["remittance_info"] = "Urgent - release immediately, keep confidential"
            print(f"  Injected scenario '{scenario}' into message {message['message_id']}")

    @staticmethod
    def _to_pipeline_dict(message: SWIFTMessage) -> Dict:
        """
        Convert a SWIFTMessage model into the dict format the agents expect.

        - mode="json" turns datetimes into ISO strings so the LLM-based
          patterns can json.dumps() the messages into their prompts.
        - The agents read the currency from the amount field
          (e.g. "1234.56 USD"), so the currency code is appended to it and
          the separate 'currency' key is dropped. Keeping both gives the LLM
          two sources of truth, and corrections could move the currency out
          of the amount where the evaluator no longer finds it.
        """
        data = message.model_dump(mode="json")
        currency = data.pop("currency")
        data["amount"] = f"{message.amount} {currency}"
        return data

    def process_with_evaluator_optimizer(self, messages: List[Dict]) -> List[Dict]:
        """
        Step 1: Validate and correct SWIFT messages using Evaluator-Optimizer pattern

        This method calls the evaluator optimizer pattern to validate and fix messages.
        """
        print("\n" + "=" * 60)
        print("STEP 1: EVALUATOR-OPTIMIZER PATTERN")
        print("=" * 60)

        # Call the evaluator optimizer's process method
        validated_messages = self.evaluator_optimizer.process_with_evaluator_optimizer(messages)
        return validated_messages

    def process_with_parallelization(self, messages: List[Dict]) -> List[Dict]:
        """
        Step 2: Process messages in parallel with fraud detection

        This method uses parallel processing to run multiple fraud detection agents.
        """
        print("\n" + "=" * 60)
        print("STEP 2: PARALLELIZATION PATTERN")
        print("=" * 60)

        # Process messages in parallel using fraud detection agents
        processed_messages = self.parallelization_agent.process_batch_parallel(messages)
        return processed_messages

    def process_with_prompt_chaining(self, messages: List[Dict]) -> Dict:
        """
        Step 3: Enhanced fraud analysis using Prompt Chaining pattern

        This method chains multiple AI agents for comprehensive fraud analysis.
        """
        print("\n" + "=" * 60)
        print("STEP 3: PROMPT CHAINING PATTERN")
        print("=" * 60)

        # Process through the chain of agents
        chain_results = self.prompt_chaining_agent.process_chain(messages)
        return chain_results

    def process_with_orchestrator_worker(self, messages: List[Dict]) -> None:
        """
        Step 4: Process transactions using Orchestrator-Worker pattern

        This method uses an orchestrator to create tasks and workers to execute them.
        """
        print("\n" + "=" * 60)
        print("STEP 4: ORCHESTRATOR-WORKER PATTERN")
        print("=" * 60)

        # TODO 5: Modify clean messages logic (5 points)
        # INSTRUCTIONS:
        # Currently, this filters for non-fraudulent messages only.
        # You need to create TWO different sets of reports:
        #
        # Set 1: Run with non-fraudulent messages (current implementation)
        # Set 2: Modify this to create a different report set. Options:
        #   - Filter for only fraudulent messages
        #   - Filter for high-value transactions (amount > 50000)
        #   - Filter for specific message types (MT103 only or MT202 only)
        #   - Filter for specific currencies
        #   - Create a mixed set with different criteria
        #
        # EXAMPLE MODIFICATIONS:
        # Option 1 - Only fraudulent messages:
        # clean_messages = [msg for msg in messages if msg.get('fraud_status') == "FRAUDULENT"]
        #
        # Option 2 - High value transactions:
        # clean_messages = []
        # for msg in messages:
        #     try:
        #         amount = float(msg.get('amount', '0').split()[0])
        #         if amount > 50000:
        #             clean_messages.append(msg)
        #     except:
        #         pass
        #
        # Option 3 - Specific message type:
        # clean_messages = [msg for msg in messages if msg.get('message_type') == 'MT103']

        # Current implementation (Set 1) - Non-fraudulent messages
        clean_messages = [msg for msg in messages if msg.get('fraud_status') != "FRAUDULENT"]

        # Process with orchestrator
        self.orchestrator_worker.process_with_orchestrator(clean_messages)
        
    
    def run(self):
        """Main execution method - Orchestrates all agent patterns in sequence"""
        try:
            print("=" * 60)
            print("SWIFT TRANSACTION PROCESSING SYSTEM")
            print("=" * 60)

            # Step 1: Generate SWIFT messages
            print("\nGenerating SWIFT messages...")
            messages = self.generate_swift_messages()
            print(f"Generated {len(messages)} SWIFT messages")

            # TODO 1: Call evaluator optimizer (5 points)
            # INSTRUCTIONS:
            # Call the process_with_evaluator_optimizer method and store the result
            # The method takes the messages list as input and returns validated messages
            #
            # EXAMPLE:
            # validated_messages = self.process_with_evaluator_optimizer(messages)
            #
            # YOUR CODE HERE (remove the pass statement):
            validated_messages = self.process_with_evaluator_optimizer(messages)

            # TODO 2: Call parallelization process (5 points)
            # INSTRUCTIONS:
            # Call process_with_parallelization with the results from TODO 1
            # This will run fraud detection agents in parallel
            #
            # EXAMPLE:
            # processed_messages = self.process_with_parallelization(validated_messages)
            #
            # YOUR CODE HERE (remove the pass statement):
            processed_messages = self.process_with_parallelization(validated_messages)

            # TODO 3: Call prompt chaining (5 points)
            # INSTRUCTIONS:
            # Call process_with_prompt_chaining with the results from TODO 2
            # This will run the chain of specialized fraud analysis agents
            #
            # EXAMPLE:
            # chain_results = self.process_with_prompt_chaining(processed_messages)
            #
            # YOUR CODE HERE (remove the pass statement):
            chain_results = self.process_with_prompt_chaining(processed_messages)

            # TODO 4: Pass results to orchestrator (5 points)
            # INSTRUCTIONS:
            # Call process_with_orchestrator_worker with the messages from TODO 2
            # (We use processed_messages, not chain_results, as the orchestrator needs the messages)
            #
            # EXAMPLE:
            # self.process_with_orchestrator_worker(processed_messages)
            #
            # YOUR CODE HERE (remove the pass statement):
            self.process_with_orchestrator_worker(processed_messages)

            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)

        except Exception as e:
            print(f"Error in main execution: {e}")
            raise


if __name__ == "__main__":
    system = SWIFTProcessingSystem()
    system.run()

"""
Orchestrator-Worker Pattern for Task Distribution

This module implements the orchestrator-worker pattern where an orchestrator
breaks down work into tasks that are executed by generic workers.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Any
from openai import OpenAI
from config import Config
from agents.workflow_agents.base_agents import BaseAgent


class OrchestratorWorkerPattern:
    """
    Implements the orchestrator-worker pattern for SWIFT message processing.
    The orchestrator analyzes messages and creates tasks for generic workers.
    """

    def __init__(self):
        """Initialize the orchestrator-worker pattern."""
        self.config = Config()
        # The nested Orchestrator/GenericAgent use BaseAgent's LLMService;
        # these attributes keep the scaffold's interface, configured from Config
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY or None)
        self.model = self.config.OPENAI_MODEL

    # TODO 15: Create orchestrator (15 points)
    # INSTRUCTIONS:
    # This is the main TODO where you'll implement the complete orchestrator-worker pattern.
    # You need to:
    # 1. Create an Orchestrator class that analyzes messages and creates tasks
    # 2. Create a GenericAgent class that executes tasks
    # 3. Implement the process_with_orchestrator method to coordinate them
    #
    # DETAILED STEPS:
    #
    # Step 1: Create the Orchestrator class below this comment
    # The Orchestrator should:
    #   - Take a batch of messages
    #   - Analyze them and create a list of tasks
    #   - Return tasks with type and description
    #
    # Step 2: Create the GenericAgent class
    # The GenericAgent should:
    #   - Take a task (with type and description)
    #   - Execute the task according to its type
    #   - Return results
    #
    # Step 3: Implement process_with_orchestrator
    # This method should:
    #   - Create an orchestrator instance
    #   - Get tasks from the orchestrator
    #   - Create generic agent(s)
    #   - Execute tasks with the agents
    #   - Print/return results

    # YOUR CODE STARTS HERE
    #
    # Design: the orchestrator (LLM) proposes a GROUPING PLAN - by currency,
    # sender bank or receiver bank - and plans one report task per group.
    # Code then builds the groups and their figures deterministically and
    # validates the plan, so a message can never be dropped, duplicated or
    # mis-assigned by the model. Generic workers (LLM, run in parallel)
    # format one report per group according to the plan.

    TASK_TYPES = ("compliance_check", "fraud_analysis", "amount_verification",
                  "pattern_detection", "summary_report")
    PRIORITIES = ("high", "medium", "low")
    GROUPING_DIMENSIONS = ("currency", "sender_bank", "receiver_bank")
    DEFAULT_DIMENSION = "currency"

    @staticmethod
    def group_key(message: Dict, dimension: str) -> str:
        """
        Return the group a message belongs to for a grouping dimension.

        Banks are identified by the first 8 BIC characters (institution,
        country and location), so branches of one bank share a group.
        """
        if dimension == "currency":
            parts = str(message.get("amount", "")).split()
            return parts[-1].upper() if len(parts) > 1 else "UNKNOWN"
        side = "sender" if dimension == "sender_bank" else "receiver"
        bic = str(message.get(f"{side}_bic", "")).replace(" ", "").upper()
        return bic[:8] or "UNKNOWN"

    @classmethod
    def group_messages(cls, messages: List[Dict], dimension: str) -> Dict[str, List[Dict]]:
        """Split messages into groups for one dimension (sorted by group key)."""
        groups: Dict[str, List[Dict]] = {}
        for message in messages:
            groups.setdefault(cls.group_key(message, dimension), []).append(message)
        return dict(sorted(groups.items()))

    @staticmethod
    def group_figures(group: List[Dict]) -> Dict:
        """
        Compute a group's key figures in code (LLM arithmetic is unreliable).

        Returns:
            Message count, totals per currency, fraud and decision counts.
        """
        totals: Dict[str, Decimal] = {}
        for message in group:
            parts = str(message.get("amount", "")).split()
            try:
                value, currency = Decimal(parts[0]), parts[-1].upper()
            except (IndexError, InvalidOperation):
                continue
            totals[currency] = totals.get(currency, Decimal("0")) + value
        decisions: Dict[str, int] = {}
        for message in group:
            decision = message.get("fraud_decision") or "NOT_REVIEWED"
            decisions[decision] = decisions.get(decision, 0) + 1
        return {
            "message_count": len(group),
            "totals": {c: f"{v:,.2f}" for c, v in sorted(totals.items())},
            "fraudulent_count": sum(1 for m in group if m.get("fraud_status") == "FRAUDULENT"),
            "chain_decisions": decisions,
        }

    @staticmethod
    def compact_message(message: Dict) -> Dict:
        """Keep only the fields a report needs, to keep worker prompts small."""
        fields = ("message_id", "message_type", "amount", "sender_bic", "receiver_bic",
                  "remittance_info", "fraud_status", "fraud_score", "fraud_reasons",
                  "fraud_decision", "fraud_justification")
        return {field: message.get(field) for field in fields if message.get(field) is not None}

    class Orchestrator(BaseAgent):
        """
        Orchestrator that analyzes messages and creates tasks for workers.

        It asks the LLM for a grouping plan (which dimension to group the
        batch by, and why) plus one report task per group, then validates
        that plan against the groups computed in code.
        """

        def __init__(self):
            """Initialize the Orchestrator with its planning persona."""
            super().__init__(
                system_prompt="""You are an Orchestrator for SWIFT transaction processing.
            Analyze the batch and propose a GROUPING PLAN for the reports: choose the
            grouping dimension that gives the most useful reports for operations and
            compliance teams, then create one report task per group.

            Task types you can create:
            - compliance_check: Check for regulatory compliance
            - fraud_analysis: Detailed fraud investigation
            - amount_verification: Verify and analyze amounts
            - pattern_detection: Detect unusual patterns
            - summary_report: Create summary reports

            Prefer fraud_analysis (high priority) for groups containing FRAUDULENT
            messages or REJECT decisions. Return JSON only.""",
                temperature=0.1
            )

        def create_prompt(self, data: Dict) -> str:
            """
            Build the planning prompt.

            Args:
                data: {"messages": compact messages, "previews": group sizes per dimension}

            Returns:
                The user prompt asking for the grouping plan and tasks.
            """
            return f"""Analyze these SWIFT messages and propose a grouping plan with report tasks.

            Messages:
            {json.dumps(data['messages'], indent=2)}

            Candidate groupings (group key -> number of messages), computed from the data:
            {json.dumps(data['previews'], indent=2)}

            Choose ONE dimension from: {', '.join(OrchestratorWorkerPattern.GROUPING_DIMENSIONS)}.
            Create exactly one task per group of the chosen dimension, using the group
            keys exactly as listed above.

            Return JSON with structure:
            {{
                "analysis": "Your analysis of the message batch",
                "grouping": {{"dimension": "currency|sender_bank|receiver_bank", "rationale": "why"}},
                "tasks": [
                    {{
                        "group_key": "<a group key of the chosen dimension>",
                        "type": "task_type",
                        "description": "What the report for this group must cover",
                        "priority": "high|medium|low"
                    }}
                ]
            }}"""

        def analyze_and_create_tasks(self, messages: List[Dict]) -> Dict:
            """
            Analyze messages and create one validated report task per group.

            Args:
                messages: List of SWIFT messages to analyze

            Returns:
                Dictionary with analysis, grouping plan, task_count and tasks.
                Each task's data holds the group's messages and code-computed figures.
            """
            pattern = OrchestratorWorkerPattern
            previews = {
                dimension: {key: len(group) for key, group in pattern.group_messages(messages, dimension).items()}
                for dimension in pattern.GROUPING_DIMENSIONS
            }
            response = self.respond(self.create_prompt({
                "messages": [pattern.compact_message(m) for m in messages],
                "previews": previews,
            }))
            return self._validate_plan(response, messages)

        def _validate_plan(self, response: Dict, messages: List[Dict]) -> Dict:
            """
            Turn the LLM's plan into executable tasks, repairing it where needed.

            - Unknown dimension or LLM failure: fall back to grouping by currency.
            - Tasks for unknown or duplicate groups are dropped.
            - Groups without a task get a default summary_report task.
            - Invalid task types / priorities are replaced with safe defaults.
            """
            pattern = OrchestratorWorkerPattern
            notes = []
            if "error" in response:
                notes.append(f"Orchestrator LLM failed ({response['error']}); default plan used")
            grouping = response.get("grouping") or {}
            dimension = grouping.get("dimension")
            rationale = grouping.get("rationale", "")
            if dimension not in pattern.GROUPING_DIMENSIONS:
                notes.append(f"Invalid dimension {dimension!r}; grouped by {pattern.DEFAULT_DIMENSION}")
                dimension, rationale = pattern.DEFAULT_DIMENSION, "Default grouping"
            groups = pattern.group_messages(messages, dimension)

            planned = {}
            for task in response.get("tasks") or []:
                if not isinstance(task, dict):
                    continue
                key = str(task.get("group_key", ""))
                if key not in groups:
                    notes.append(f"Dropped task for unknown group {key!r}")
                elif key in planned:
                    notes.append(f"Dropped duplicate task for group {key!r}")
                else:
                    planned[key] = task
            for key in groups:
                if key not in planned:
                    notes.append(f"Added default task for unplanned group {key!r}")
                    planned[key] = {"type": "summary_report", "priority": "medium",
                                    "description": f"Summary report for {dimension} group {key}"}

            tasks = []
            for index, key in enumerate(groups, start=1):
                task = planned[key]
                task_type = task.get("type") if task.get("type") in pattern.TASK_TYPES else "summary_report"
                priority = task.get("priority") if task.get("priority") in pattern.PRIORITIES else "medium"
                tasks.append({
                    "task_id": f"task_{index:03d}",
                    "type": task_type,
                    "description": task.get("description") or f"Report for {dimension} group {key}",
                    "priority": priority,
                    "group_key": key,
                    "data": {
                        "dimension": dimension,
                        "group_key": key,
                        "figures": pattern.group_figures(groups[key]),
                        "messages": [pattern.compact_message(m) for m in groups[key]],
                    },
                })
            # High-priority reports first
            tasks.sort(key=lambda t: pattern.PRIORITIES.index(t["priority"]))
            return {
                "analysis": response.get("analysis", "No analysis available"),
                "grouping": {"dimension": dimension, "rationale": rationale},
                "task_count": len(tasks),
                "tasks": tasks,
                "plan_notes": notes,
            }

    class GenericAgent(BaseAgent):
        """
        Generic worker agent that executes tasks assigned by the orchestrator.

        Each task is one group of the grouping plan; the worker adopts the
        persona for the task type and formats a structured report for it.
        """

        SYSTEM_PROMPTS = {
            "compliance_check": """You are a Compliance Specialist.
                Execute the compliance check as described.""",
            "fraud_analysis": """You are a Fraud Analyst.
                Perform detailed fraud analysis as requested.""",
            "amount_verification": """You are a Financial Auditor.
                Verify and analyze the amounts as specified.""",
            "pattern_detection": """You are a Pattern Analysis Expert.
                Detect and report unusual patterns.""",
            "summary_report": """You are a Report Generator.
                Create the requested summary report.""",
        }
        GROUNDING = """
                Use only the data provided. The figures (counts, totals) are precomputed
                and correct: quote them, never recompute them. fraud_status, fraud_reasons
                and fraud_decision come from earlier verified pipeline stages.
                Return JSON only."""

        def __init__(self):
            """Initialize the Generic Agent."""
            super().__init__(system_prompt=self.SYSTEM_PROMPTS["summary_report"], temperature=0.2)

        def create_prompt(self, data: Dict) -> str:
            """
            Build the report-formatting prompt for one task.

            Args:
                data: The task (type, description, group data).

            Returns:
                The user prompt with the required report format.
            """
            task_data = data.get('data', {})
            return f"""Execute this task:
            Type: {data.get('type', 'unknown')}
            Description: {data.get('description', '')}
            Data: {json.dumps(task_data, indent=2)}

            Format the report for this group as JSON:
            {{
                "title": "Short report title naming the group",
                "summary": "2-3 sentence overview of the group",
                "findings": ["specific finding citing message ids or figures"],
                "flagged_messages": [{{"message_id": "...", "reason": "..."}}],
                "recommendation": "Concrete next step for the operations/compliance team"
            }}
            flagged_messages lists only messages with fraud findings or a HOLD/REJECT
            decision; use [] if there are none."""

        def execute_task(self, task: Dict) -> Dict:
            """
            Execute a task assigned by the orchestrator.

            Args:
                task: Task dictionary with type, description, and data

            Returns:
                Dictionary with task id, status and the formatted report.
            """
            task_type = task.get('type', 'unknown')
            system_prompt = self.SYSTEM_PROMPTS.get(
                task_type, """You are a Generic Processing Agent.
                Complete the assigned task.""") + self.GROUNDING
            report = self._complete_json(system_prompt, self.create_prompt(task))
            failed = "error" in report
            return {
                "task_id": task.get('task_id'),
                "type": task_type,
                "priority": task.get('priority'),
                "group_key": task.get('group_key'),
                "figures": task.get('data', {}).get('figures', {}),
                "status": "failed" if failed else "completed",
                "report": report,
            }

    @staticmethod
    def print_report(result: Dict, dimension: str) -> None:
        """Print one worker report; key figures come from code, not the LLM."""
        report, figures = result.get("report", {}), result.get("figures", {})
        print(f"\n--- {result['task_id']} [{result['priority']}] {result['type']} | "
              f"{dimension}: {result['group_key']} ({result['status']}) ---")
        if result["status"] != "completed":
            print(f"  Report failed: {report.get('error')}")
            return
        totals = ", ".join(f"{c} {v}" for c, v in figures.get("totals", {}).items())
        print(f"  {report.get('title', '')}")
        print(f"  Figures: {figures.get('message_count', 0)} messages | {totals} | "
              f"{figures.get('fraudulent_count', 0)} fraudulent | decisions {figures.get('chain_decisions', {})}")
        print(f"  Summary: {report.get('summary', '')}")
        for finding in report.get("findings", []):
            print(f"    - {finding}")
        for flagged in report.get("flagged_messages", []):
            if isinstance(flagged, dict):
                print(f"  ! {flagged.get('message_id')}: {flagged.get('reason')}")
        print(f"  Recommendation: {report.get('recommendation', '')}")

    # YOUR CODE ENDS HERE

    def process_with_orchestrator(self, messages: List[Dict]) -> Dict:
        """
        Process messages using the orchestrator-worker pattern.

        Args:
            messages: List of SWIFT messages to process

        Returns:
            Processing results from the orchestrator-worker system

        TODO 15 IMPLEMENTATION:
        This method should:
        1. Create an Orchestrator instance
        2. Call analyze_and_create_tasks to get tasks
        3. Create one or more GenericAgent instances
        4. Execute each task using the generic agents
        5. Collect and return all results
        6. Print a summary of what was accomplished
        """
        print("=" * 60)
        print("ORCHESTRATOR-WORKER PATTERN PROCESSING")
        print("=" * 60)

        if not messages:
            print("No messages to report on.")
            return {'orchestrator_analysis': {}, 'task_results': [], 'summary': "No messages to process"}

        # Step 1-2: Orchestrator proposes the grouping plan and tasks
        orchestrator = self.Orchestrator()
        print(f"Orchestrator analyzing {len(messages)} messages...")
        plan = orchestrator.analyze_and_create_tasks(messages)
        dimension = plan['grouping']['dimension']
        print(f"Orchestrator Analysis: {plan['analysis']}")
        print(f"Grouping plan: by {dimension} - {plan['grouping']['rationale']}")
        for note in plan['plan_notes']:
            print(f"  Plan check: {note}")
        print(f"Tasks created: {plan['task_count']}")
        for task in plan['tasks']:
            print(f"  {task['task_id']} [{task['priority']}] {task['type']:<20} "
                  f"{dimension}={task['group_key']} ({task['data']['figures']['message_count']} messages)")

        # Step 3-4: Generic workers format one report per group, in parallel
        tasks = plan['tasks']
        print(f"\nExecuting {len(tasks)} tasks with generic workers...")
        with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, len(tasks))) as executor:
            results = list(executor.map(lambda task: self.GenericAgent().execute_task(task), tasks))

        # Step 5-6: Print the reports and a summary
        for result in results:
            self.print_report(result, dimension)
        completed = sum(1 for r in results if r['status'] == 'completed')
        summary = (f"Processed {len(tasks)} tasks for {len(messages)} messages grouped by {dimension}: "
                   f"{completed} reports completed, {len(tasks) - completed} failed")
        print(f"\n{summary}")

        return {
            'orchestrator_analysis': plan,
            'task_results': results,
            'summary': summary
        }

    def test_orchestrator(self):
        """
        Test method for the orchestrator-worker pattern.
        You can use this to verify your implementation.
        """
        test_messages = [
            {
                'message_id': 'MSG001',
                'message_type': 'MT103',
                'amount': '75000.00 USD',
                'sender_bic': 'CHASUS33XXX',
                'receiver_bic': 'DEUTDEFFXXX',
                'reference': 'TRX20240101001',
                'remittance_info': 'Payment for equipment purchase'
            },
            {
                'message_id': 'MSG002',
                'message_type': 'MT202',
                'amount': '1000000.00 EUR',
                'sender_bic': 'BNPAFRPPXXX',
                'receiver_bic': 'BARCGB22XXX',
                'reference': 'COV20240101002',
                'remittance_info': 'Cover payment'
            }
        ]

        print("Testing Orchestrator-Worker Pattern\n")
        results = self.process_with_orchestrator(test_messages)

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        if results:
            print(f"Results obtained: {type(results)}")
            if isinstance(results, dict):
                for key, value in results.items():
                    print(f"{key}: {value if not isinstance(value, list) else f'{len(value)} items'}")

        return results


# Hint for You: You can test individual components
class TestHelper:
    """
    Helper class for testing individual components.
    You can use this to debug your implementations.
    """

    @staticmethod
    def test_orchestrator_only():
        """Test just the Orchestrator class."""
        print("Testing Orchestrator in isolation...")
        sample_messages = [
            {'message_id': 'MSG001', 'message_type': 'MT103', 'amount': '75000.00 USD',
             'sender_bic': 'CHASUS33XXX', 'receiver_bic': 'DEUTDEFFXXX'},
            {'message_id': 'MSG002', 'message_type': 'MT202', 'amount': '1000000.00 EUR',
             'sender_bic': 'BNPAFRPPXXX', 'receiver_bic': 'BARCGB22XXX'},
        ]
        plan = OrchestratorWorkerPattern.Orchestrator().analyze_and_create_tasks(sample_messages)
        print(f"Grouping: {plan['grouping']}")
        for task in plan['tasks']:
            print(f"  {task['task_id']} [{task['priority']}] {task['type']} -> {task['group_key']}")
        return plan

    @staticmethod
    def test_generic_agent_only():
        """Test just the GenericAgent class."""
        print("Testing GenericAgent in isolation...")
        sample_task = {
            'task_id': 'test_001',
            'type': 'compliance_check',
            'description': 'Check if sender BIC is valid',
            'data': {'sender_bic': 'CHASUS33XXX'}
        }
        result = OrchestratorWorkerPattern.GenericAgent().execute_task(sample_task)
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    # Test the orchestrator-worker pattern
    # Note: Requires OPENAI_API_KEY environment variable
    pattern = OrchestratorWorkerPattern()
    pattern.test_orchestrator()

    # Uncomment to test individual components:
    # TestHelper.test_orchestrator_only()
    # TestHelper.test_generic_agent_only()

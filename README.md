# SWIFTGuard — Agentic SWIFT Message Processor

An agentic AI pipeline that validates, screens and reports on SWIFT payment messages. It combines four agentic workflow patterns — **Evaluator-Optimizer**, **Parallelization**, **Prompt Chaining** and **Orchestrator-Workers** — into one end-to-end system.

Udacity *Agentic AI for Financial Services* — project **SWIFTGuard**. All 15 TODOs are implemented, and the evidence of real runs is in [`project/evidence/`](project/evidence/).

---

## Pipeline

```
Generate 10 SWIFT messages (MT103 / MT202)
   │   3 get a deliberate format defect, 2 become suspicious scenarios
   ▼
STEP 1  Evaluator-Optimizer   validate each message → LLM correction agent fixes it → re-validate
   ▼
STEP 2  Parallelization       3 fraud agents analyse every message concurrently → FraudAggAgent
   │                          marks each message FRAUDULENT / CLEAN with a score and reasons
   ▼
STEP 3  Prompt Chaining       Initial Screener → Technical Analyst → Risk Assessor
   │                          → Compliance Officer → Final Reviewer (APPROVE / HOLD / REJECT)
   ▼
STEP 4  Orchestrator-Workers  for each report set: orchestrator plans the grouping and tasks,
                              generic workers write one report per group
```

| Step | Pattern | Where | What it does |
|---|---|---|---|
| 1 | Evaluator-Optimizer | `agents/evaluator_optimizer.py`, `SwiftCorrectionAgent` | Rule-based validation (required fields, message type, amount, BIC format, currency). Invalid messages go to an LLM correction agent with the exact errors, and are re-validated for up to 3 iterations. Code guardrails revert any LLM change to the amount or message type, so a correction can never alter the payment itself. |
| 2 | Parallelization | `agents/parallelization.py` | `ThreadPoolExecutor` runs the three fraud agents on every message at the same time; `FraudAggAgent` combines their results. |
| 3 | Prompt Chaining | `agents/prompt_chaining.py` | Five specialist LLM roles. Each step receives the previous steps' output, and the final reviewer decides per message. The chain log shows each step's input/output. |
| 4 | Orchestrator-Workers | `agents/orchestrator_worker.py` | The orchestrator picks a grouping dimension (e.g. currency) and creates one task per group; its plan is validated against the real messages. Generic workers turn each task into a report. |

---

## The third fraud agent: `GeographicRiskAgent` (TODO 10)

Defined in [`project/agents/workflow_agents/base_agents.py`](project/agents/workflow_agents/base_agents.py) and registered as the third agent in `ParallelizationPattern.list_of_agents`. It runs alongside the provided `FraudAmountDetectionAgent` and `FraudPatternDetectionAgent`.

It scores **geographic and corridor risk**: where the money comes from, where it goes, in which currency, and whether the stated purpose fits that route. It is a **hybrid** agent: every check that can be computed from the data is done in code, and the LLM is used only for the one judgement that needs language understanding.

| Check | How | Score |
|---|---|---|
| **Sanctioned / high-risk jurisdiction** — sender or receiver country (from BIC characters 5–6) is on the high-risk list (e.g. IR, KP, RU, SY) | deterministic, **no LLM call** | **1.0** immediately |
| **Rule 1 — secrecy jurisdiction**: cross-border payment over 100,000 into e.g. KY, PA, VG, CH | deterministic | +0.3 |
| **Rule 2 — third-currency payment**: currency is neither party's domestic currency nor USD/EUR | deterministic | +0.2 |
| **Rule 3 — large cross-border amount** over 1,000,000 | deterministic | +0.2 |
| **Rule 4 — implausible purpose**: remittance info doesn't fit the amount or corridor (e.g. a multi-million "salary") | **LLM** (`temperature=0`, JSON output) | +0.2 |

Design decisions:
- **Sanctions never depend on a model.** A sanctioned country must always be flagged, so that check runs first, in code, and returns 1.0 without calling the LLM.
- **Facts stay in code.** In early runs the LLM sometimes mixed up sender and receiver, so every rule that can be checked from the data (rules 1–3) is computed deterministically. The LLM is only asked about purpose plausibility, and only when a purpose is stated (MT202 messages have none, so they cost no call).
- **Fail safe.** If the LLM call fails, the agent returns at least 0.5 with a *"manual review needed"* reason, instead of silently passing the message.
- The country lists are illustrative for this project, not an official sanctions list.

### Aggregation: `FraudAggAgent` (TODOs 11–12)

A message is **FRAUDULENT** if the **average** risk of the three agents is at least **0.5**, **or** if any **single** agent scores **0.9 or higher**. The second rule is a decisive-signal override: without it, a sanctioned-country hit (1.0) averaged with two clean scores (0.0) would come out at 0.33 and pass. Each message gets `fraud_status`, `fraud_score` (confidence %) and `fraud_reasons`, each reason prefixed with the agent that raised it.

---

## Report sets (TODO 5)

Step 4 produces **one report set per entry in `REPORT_SETS`**, each built from a different message filter. Every set gets its own orchestrator run, grouping plan and reports.

| Name | Messages |
|---|---|
| `clean` *(default set 1)* | not flagged as fraudulent |
| `fraudulent` *(default set 2)* | flagged as fraudulent |
| `high_value` | amount above `HIGH_VALUE_THRESHOLD` (50,000) |
| `mt103` | MT103 customer transfers only |

Switch the sets with configuration only, no code changes:

```bash
REPORT_SETS=high_value,mt103 uv run python main.py
```

---

## Setup and running

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cd project
uv sync
cp .env.example .env        # then put your OpenAI API key in .env
```

```bash
uv run python main.py       # full pipeline, default report sets (clean, fraudulent)
```

```bash
uv run pytest -v            # offline test suite: 41 tests, no API calls
```

Settings live in [`project/config.py`](project/config.py): message count, number of injected defects and suspicious scenarios, worker count, agent timeout, report sets, and the OpenAI model (`gpt-4o` by default; override with `OPENAI_MODEL`). The API key is read from `project/.env`, which is git-ignored.

---

## Evidence of testing

Console output from real runs, in [`project/evidence/`](project/evidence/):

| File | Shows |
|---|---|
| [`run_default_report_sets.log`](project/evidence/run_default_report_sets.log) | Full run with real `gpt-4o` calls: 3 injected format defects fixed by the Evaluator-Optimizer (all 10 messages valid), 2/10 messages flagged by the fraud agents, the 5-step prompt chain log, and **report sets `clean` (8 messages, 5 reports) and `fraudulent` (2 messages, 2 reports)** |
| [`run_high_value_and_mt103_report_sets.log`](project/evidence/run_high_value_and_mt103_report_sets.log) | The same pipeline with the **`high_value` and `mt103`** report sets, selected by configuration only |
| [`pytest_results.log`](project/evidence/pytest_results.log) | 41 passed |

Messages are generated randomly, so IDs, amounts and banks differ on every run; the pipeline steps and checks stay the same.

---

## Project structure

```
project/
├── main.py                       # pipeline: generate → validate → fraud → chain → reports (TODOs 1-5)
├── config.py                     # settings; loads project/.env
├── agents/
│   ├── evaluator_optimizer.py    # Step 1
│   ├── parallelization.py        # Step 2 (TODOs 10-12)
│   ├── prompt_chaining.py        # Step 3 (TODOs 13-14)
│   ├── orchestrator_worker.py    # Step 4 (TODO 15)
│   └── workflow_agents/
│       └── base_agents.py        # BaseAgent, correction agent, fraud agents incl. GeographicRiskAgent,
│                                 # FraudAggAgent (TODOs 6-9)
├── models/                       # SWIFT message and bank models
├── services/                     # SWIFT message generator, LLM service
├── tests/                        # offline pytest suite (LLM calls mocked)
└── evidence/                     # console logs of real runs
```

`STARTER_README.md` is the original Udacity starter guide; `SWIFT_OVERVIEW.md` explains the SWIFT message formats.

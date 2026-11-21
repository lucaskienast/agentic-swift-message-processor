# Testing evidence

Console output captured on 2026-11-21 from the `project/` folder.

| File | Command | What it shows |
|---|---|---|
| `run_default_report_sets.log` | `uv run python main.py` | Full end-to-end run with real `gpt-4o` calls: 3 injected format defects fixed by the Evaluator-Optimizer, 2 injected suspicious scenarios flagged by the parallel fraud agents, the 5-step prompt chain log, and the two default report sets (`clean`, `fraudulent`) from the Orchestrator-Worker step |
| `run_high_value_and_mt103_report_sets.log` | `REPORT_SETS=high_value,mt103 uv run python main.py` | The same pipeline with two other report filters, selected by configuration only |
| `pytest_results.log` | `uv run pytest -v` | The offline test suite (41 tests, no API calls) |

The generated messages are random, so every run produces different message
IDs, amounts and banks. The pipeline steps and checks stay the same.

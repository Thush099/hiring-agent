# Offline Rubric Calibration, Evidence Span & Fairness Audit

`calibration_audit.py` and the `tests/` suite provide deterministic, zero-API-key verification for `hiring-agent`'s rubric loading, JSON extraction, Pydantic schema validation, score aggregation/clamping, evidence-span grounding, and counterfactual fairness invariance.

## Quickstart

Run the full test suite and standalone calibration audit from the repository root without any external LLM keys or optional `pymupdf`/`python-dotenv` packages:

```bash
# Run unit, scoring, schema resilience, and fairness tests
PYTHONPATH=. ../.venv/bin/pytest -v

# Run the standalone offline calibration & perturbation audit CLI
PYTHONPATH=. ../.venv/bin/python calibration_audit.py

# Machine-readable JSON report for CI gates
PYTHONPATH=. ../.venv/bin/python calibration_audit.py --role software_engineering_intern --json
```

Exit code is `0` when all integrity, calibration, counterfactual invariance, monotonicity, and resilience gates pass, and `1` otherwise.

## What Was Hardened

Auditing the core scoring pipeline uncovered and resolved several edge-case gaps while preserving upstream CLI and config compatibility:

- **Authoritative score bounds & clamping (`score.py`)**: Extracted pure `compute_evaluation_score(evaluation, role)` used by `print_evaluation_results()`. Category scores now clamp against `role.categories` `max` (rather than trusting LLM-reported `max` fields), bonuses clamp to `[0, role.bonus_max]`, deductions clamp to `>= 0`, and final scores clamp to `[role.min_final_score, role.max_final_score]`.
- **Optional environment & PDF guards (`config.py`, `pdf.py`)**: Added a fallback `.env` loader when `python-dotenv` is absent, rejected whitespace-only API keys in `provider_for()`, and guarded top-level `pymupdf` imports in `pdf.py` so scoring and audit modules import cleanly without PDF binary dependencies.
- **LLM response JSON recovery (`llm_utils.py`)**: Fixed `extract_json_from_response()` so whitespace and markdown fences following `<think>...</think>` blocks are stripped reliably, case-insensitive/bare fences are handled, and top-level JSON objects/arrays wrapped in prose (such as `"Note [1]: [{...}]"`) are recovered via `json.JSONDecoder.raw_decode`.
- **Schema & role validation (`models.py`, `roles.py`, `evaluator.py`, `transform.py`)**:
  - Replaced deprecated Pydantic `min_items`/`max_items` with `min_length`/`max_length`, disallowed `inf`/`nan` scores and deductions (`allow_inf_nan=False`), and rejected whitespace-only `evidence` strings.
  - Added role-name traversal checks, `max > 0` category checks, `min_final_score <= max_final_score` checks, null-safe `bonus_rules`, and `roles_dir` injection in `roles.py`.
  - Added optional `provider: Optional[LLMProvider]` injection on `ResumeEvaluator` and `PDFHandler`, and fixed `None` field crashes and input dict mutation in `transform.py`.

## Audit Components (`calibration_audit.py`)

| Component | Purpose |
| --- | --- |
| `audit_rubric_integrity` | Verifies role bounds, non-overlapping bonus tiers, Jinja template rendering, category coverage, and fairness clauses. |
| `normalize_weighted_criteria` / `classify_tier` | Normalizes category scores (proportional or custom weights) to `0–100` and assigns deterministic tiers (`Strong Fast-Track` ≥ 80, `Advance to Review` ≥ 60, `Borderline / Manual Check` ≥ 40, `Below Cutoff` < 40). |
| `audit_evidence_spans` | Matches category and bonus evidence anchors against rendered resume/GitHub/blog text with character span offsets, flags protected-attribute leakage (GPA/CGPA, elite institution names, gender/pronouns, location), and enforces rubric rules (`self_project`-only OSS ceiling ≤ 10, `key_strengths` consistency). |
| `DeterministicRubricProvider` | Implements `models.LLMProvider` offline by stripping protected PII/institution/grade/location lines and scoring strictly from technical evidence spans. |
| `audit_counterfactual_fairness` | Verifies zero score drift (`max_invariance_delta == 0.0`, `kendall_tau == 1.000`, `tier_flip_rate == 0.0%`) under demographic/pronoun, institution/GPA/location, and formatting perturbations, monotonic score shifts when adding/removing quantified achievements, and negative-control detection against an intentionally biased provider. |
| `run_resilience_checks` | Exercises `<think>` + markdown fence extraction, truncated JSON rejection, blank-evidence validation, category overshoot clamping, `min_final_score` clamping, Orchestrate tier validation, and `None`-safe CSV transform. |

## Verified Test Coverage

- `tests/test_scoring_and_models.py` — 7 test functions covering `config.provider_for`, `roles` loading/scaffolding/validation, `models` validation + all HackerRank Orchestrate rank tiers + 20-point bonus cap, mocked `OpenAICompatibleProvider` 429 retries, `llm_utils` extraction + `ResumeEvaluator` provider injection, `score.compute_evaluation_score` clamping/normalization/tiering, and `transform.py` `None` safety.
- `tests/test_fairness_calibration.py` — 6 test functions covering role rubric integrity, 5-candidate synthetic cohort tiering and rank ordering, counterfactual demographic/institution/formatting invariance, negative-control biased provider detection, evidence-span grounding/leakage/rubric-violation detection, and `calibration_audit.py` CLI execution.

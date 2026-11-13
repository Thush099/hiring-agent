"""Unit and integration tests for models, roles, config, LLM parsing, transforms, and score computation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import config
from calibration_audit import (
    TIER_ADVANCE,
    TIER_BELOW_CUTOFF,
    TIER_BORDERLINE,
    TIER_STRONG,
    classify_tier,
    normalize_weighted_criteria,
)
from evaluator import ResumeEvaluator
from llm_utils import extract_json_from_response, parse_json_response
from models import (
    Basics,
    CategoryScore,
    Deductions,
    JSONResume,
    Location,
    OpenAICompatibleProvider,
    build_evaluation_model,
)
from roles import load_role, scaffold_role
from score import compute_evaluation_score, print_evaluation_results
from transform import transform_evaluation_response, transform_parsed_data


def _valid_intern_payload() -> dict:
    return {
        "scores": {
            "open_source": {
                "score": 20,
                "max": 35,
                "evidence": "Merged 4 PRs to upstream open-source CLI tool.",
            },
            "self_projects": {
                "score": 22,
                "max": 30,
                "evidence": "Built distributed cache with live demo.",
            },
            "production": {
                "score": 18,
                "max": 25,
                "evidence": "Backend intern reducing p95 latency by 40%.",
            },
            "technical_skills": {
                "score": 8,
                "max": 10,
                "evidence": "Python, Go, PostgreSQL, Redis, Docker.",
            },
            "ai_fluency": {
                "score": 7,
                "max": 10,
                "evidence": "Built RAG pipeline with retrieval evals.",
            },
        },
        "bonus_points": {
            "items": {
                "gsoc": {"points": 5, "evidence": "Google Summer of Code 2025."},
                "girl_script": {"points": 0, "evidence": "Absent from resume."},
                "orchestrate": {
                    "rank": 42,
                    "points": 4,
                    "evidence": "HackerRank Orchestrate rank 42.",
                },
                "founder": {"points": 0, "evidence": "No founder experience."},
                "early_engineer": {"points": 0, "evidence": "Not evidenced."},
                "portfolio": {"points": 2, "evidence": "https://example.dev"},
                "linkedin": {"points": 1, "evidence": "LinkedIn profile present."},
                "technical_blogs": {"points": 0, "evidence": "No blog data provided."},
            }
        },
        "deductions": {"total": 0, "reasons": "None."},
        "key_strengths": [
            "Open source pull requests",
            "Production backend internship",
        ],
        "areas_for_improvement": ["Add more technical writing"],
    }


def test_config_provider_resolution_and_validation(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        config.provider_for("gemini-2.5-flash")

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        config.provider_for("gemini-2.5-flash")

    monkeypatch.setenv("GEMINI_API_KEY", "  test-gemini-key  ")
    resolved = config.provider_for("gemini-2.5-flash")
    assert resolved["api_key"] == "test-gemini-key"
    assert resolved["structured_output"] == "json_schema"
    assert "generativelanguage.googleapis.com" in resolved["base_url"]

    # Keyless local Ollama provider resolves without API key.
    ollama_cfg = config.provider_for("gemma4:latest")
    assert ollama_cfg["api_key"] is None
    assert ollama_cfg["extra_body"] == {"num_ctx": 32768}

    with pytest.raises(ValueError, match="Unknown model"):
        config.provider_for("nonexistent-model-xyz")


def test_roles_load_scaffold_and_validation(tmp_path):
    intern = load_role("software_engineering_intern")
    assert [c.key for c in intern.categories] == [
        "open_source",
        "self_projects",
        "production",
        "technical_skills",
        "ai_fluency",
    ]
    assert sum(c.max for c in intern.categories) == 110
    assert intern.bonus_max == 20
    assert intern.min_final_score == -20
    assert intern.max_final_score == 130
    assert len(intern.bonus_rules) == 8

    # Scaffold and load a custom role in tmp_path without bonus_rules.
    role_dir = scaffold_role("backend_intern", roles_dir=tmp_path)
    assert (role_dir / "role.json").is_file()
    custom = load_role("backend_intern", roles_dir=tmp_path)
    assert custom.name == "backend_intern"
    assert custom.bonus_rules == []

    # Reject path traversal and duplicate role scaffolding.
    with pytest.raises(ValueError, match="Invalid role name"):
        load_role("../etc", roles_dir=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        scaffold_role("backend_intern", roles_dir=tmp_path)

    # Reject invalid category max and bonus_rules=null gracefully converted to [].
    bad_dir = tmp_path / "bad_role"
    bad_dir.mkdir()
    (bad_dir / "criteria.jinja").write_text("{{ text_content }}", encoding="utf-8")
    (bad_dir / "system_message.jinja").write_text("System", encoding="utf-8")
    (bad_dir / "role.json").write_text(
        json.dumps(
            {
                "categories": [{"key": "coding", "label": "Coding", "max": 0}],
                "bonus_rules": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max must be > 0"):
        load_role("bad_role", roles_dir=tmp_path)

    # Reject min_final_score > max_final_score.
    (bad_dir / "role.json").write_text(
        json.dumps(
            {
                "categories": [{"key": "coding", "label": "Coding", "max": 50}],
                "min_final_score": 80,
                "max_final_score": 50,
                "bonus_rules": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_final_score"):
        load_role("bad_role", roles_dir=tmp_path)


def test_models_validation_and_orchestrate_tiers():
    role = load_role("software_engineering_intern")
    eval_model = build_evaluation_model(role)

    valid = eval_model(**_valid_intern_payload())
    assert valid.bonus_points.total == 12

    # Test finite & non-blank validators.
    with pytest.raises(ValidationError):
        CategoryScore(score=float("inf"), max=35, evidence="Valid text")
    with pytest.raises(ValidationError):
        CategoryScore(score=10, max=35, evidence="   ")
    with pytest.raises(ValidationError):
        Deductions(total=float("nan"), reasons="None")

    # Exhaustive Orchestrate rank tier enforcement.
    for rank, expected_pts in [
        (1, 4),
        (100, 4),
        (101, 3),
        (200, 3),
        (201, 1),
        (500, 1),
        (501, 0),
        (None, 0),
    ]:
        payload = _valid_intern_payload()
        payload["bonus_points"]["items"]["orchestrate"] = {
            "rank": rank,
            "points": expected_pts,
            "evidence": f"Rank {rank}",
        }
        parsed = eval_model(**payload)
        assert parsed.bonus_points.items.orchestrate.points == expected_pts

        wrong_pts = 1 if expected_pts == 0 else 0
        payload["bonus_points"]["items"]["orchestrate"]["points"] = wrong_pts
        with pytest.raises(ValidationError, match="HackerRank Orchestrate"):
            eval_model(**payload)

    # Bonus cap at role.bonus_max (20) when raw items total 26.
    capped_payload = _valid_intern_payload()
    capped_payload["bonus_points"]["items"] = {
        "gsoc": {"points": 5, "evidence": "GSoC"},
        "girl_script": {"points": 3, "evidence": "GSSoC"},
        "orchestrate": {"rank": 10, "points": 4, "evidence": "Rank 10"},
        "founder": {"points": 5, "evidence": "Founder"},
        "early_engineer": {"points": 3, "evidence": "Early engineer"},
        "portfolio": {"points": 2, "evidence": "Portfolio"},
        "linkedin": {"points": 1, "evidence": "LinkedIn"},
        "technical_blogs": {"points": 3, "evidence": "Blogs"},
    }
    capped_eval = eval_model(**capped_payload)
    assert capped_eval.bonus_points.total == 20
    assert "Subtotal: 26; total after 20-point cap: 20." in capped_eval.bonus_points.breakdown


def test_openai_compatible_provider_mocked(monkeypatch):
    import requests

    provider = OpenAICompatibleProvider(
        base_url="https://api.example.com/v1/",
        api_key="secret-token",
        structured_output="json_schema",
        extra_body={"num_ctx": 16384},
    )

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {"Retry-After": "0.01"}

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.headers = {}
    ok_resp.json.return_value = {
        "choices": [{"message": {"content": '{"status": "ok"}'}}]
    }

    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return rate_limited if len(calls) == 1 else ok_resp

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = provider.chat(
        model="demo-model",
        messages=[{"role": "user", "content": "score"}],
        options={"temperature": 0.1},
        format={"type": "object"},
    )
    assert result == {"message": {"role": "assistant", "content": '{"status": "ok"}'}}
    assert len(calls) == 2
    assert calls[0]["url"] == "https://api.example.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert calls[0]["json"]["num_ctx"] == 16384


def test_llm_json_extraction_and_evaluator_injection():
    # Think tags + newlines + uppercase JSON fence + prose
    raw = (
        "<think>Thinking carefully\nabout [brackets] and scores</think>\n\n"
        "Sure! Here is your structured result:\n```JSON\n"
        '{"a": 1, "b": [2, 3]}\n```\nDone.'
    )
    assert parse_json_response(raw) == {"a": 1, "b": [2, 3]}

    # Prose wrapping raw JSON array without fences (as in github.py project selection)
    arr_raw = 'Note [1]: selected repos follow:\n[{"name": "repo1"}, {"name": "repo2"}]\nEnd.'
    assert parse_json_response(arr_raw) == [{"name": "repo1"}, {"name": "repo2"}]

    # Trailing prose after an unfenced JSON object
    trailing_raw = '{"scores": {"open_source": 10}}\nHope that helps!'
    assert parse_json_response(trailing_raw) == {"scores": {"open_source": 10}}

    # Unclosed fence / empty input handling
    assert extract_json_from_response(None) == ""
    with pytest.raises(ValueError, match="no JSON content"):
        parse_json_response("   ")
    with pytest.raises(json.JSONDecodeError):
        parse_json_response('{"broken": ')

    # ResumeEvaluator with injected offline provider
    role = load_role("software_engineering_intern")
    eval_model = build_evaluation_model(role)
    payload_json = json.dumps(_valid_intern_payload())

    class _WrappedProvider:
        def chat(self, model, messages, options=None, **kwargs):
            assert "format" in kwargs
            return {
                "message": {
                    "role": "assistant",
                    "content": f"<think>internal</think>\n```json\n{payload_json}\n```",
                }
            }

    evaluator = ResumeEvaluator(
        role=role,
        evaluation_model=eval_model,
        provider=_WrappedProvider(),
    )
    evaluated = evaluator.evaluate_resume("=== SKILLS ===\nPython")
    assert evaluated.scores.open_source.score == 20


def test_score_aggregation_clamping_normalization_and_tiers(capsys):
    role = load_role("software_engineering_intern")
    eval_model = build_evaluation_model(role)

    # Case 1: LLM overshoots BOTH score AND reported max; score.py must clamp to role.categories max!
    overshoot_payload = _valid_intern_payload()
    overshoot_payload["scores"]["open_source"] = {
        "score": 80,
        "max": 80,
        "evidence": "Claimed inflated max.",
    }
    overshoot_eval = eval_model(**overshoot_payload)
    bd = compute_evaluation_score(overshoot_eval, role)
    assert bd.categories[0].capped_score == 35.0
    assert bd.categories[0].role_max == 35
    assert bd.categories[0].was_capped is True
    assert bd.categories[0].max_mismatch is True
    assert bd.max_base_score == 110
    assert bd.base_score == 35 + 22 + 18 + 8 + 7  # 90
    assert bd.bonus_score == 12
    assert bd.total_score == 102.0

    # print_evaluation_results should emit warning and authoritative denominator /110
    print_evaluation_results(overshoot_eval, role, candidate_name="Test Candidate")
    out = capsys.readouterr().out
    assert "open_source score capped from 80.0 to 35.0 (max: 35)" in out
    assert "OVERALL SCORE: 102.0/110" in out

    # Case 2: Extreme deductions clamped at role.min_final_score (-20)
    low_payload = _valid_intern_payload()
    low_payload["deductions"]["total"] = 500
    low_eval = eval_model(**low_payload)
    low_bd = compute_evaluation_score(low_eval, role)
    assert low_bd.unclamped_total < -20
    assert low_bd.total_score == -20.0
    assert low_bd.was_clamped_low is True

    # Case 3: Weighted normalization & tier classification
    norm_default = normalize_weighted_criteria(bd, role)
    assert norm_default.normalized_base_score == round((90 / 110) * 100, 2)
    assert norm_default.tier == TIER_STRONG

    equal_weights = {c.key: 1.0 for c in role.categories}
    norm_equal = normalize_weighted_criteria(bd, role, category_weights=equal_weights)
    assert sum(norm_equal.weights_used.values()) == pytest.approx(1.0, abs=1e-4)

    with pytest.raises(ValueError, match="Weight for category"):
        normalize_weighted_criteria(
            bd,
            role,
            category_weights={**equal_weights, "open_source": -1.0},
        )

    assert classify_tier(85.0) == TIER_STRONG
    assert classify_tier(65.0) == TIER_ADVANCE
    assert classify_tier(45.0) == TIER_BORDERLINE
    assert classify_tier(20.0) == TIER_BELOW_CUTOFF


def test_transform_defensive_none_and_non_mutation():
    raw_input = {
        "basics": {
            "name": "Alex",
            "profiles": [{"network": "", "url": "https://github.com/alex-dev"}],
        },
        "education": [{"institution": "Uni", "degree": None, "years": "2022-2026"}],
        "projects": [{"name": None, "description": "Demo", "technologies": "Python, Go"}],
    }
    original_snapshot = json.loads(json.dumps(raw_input))
    transformed = transform_parsed_data(raw_input)
    assert raw_input == original_snapshot  # input dicts were not mutated in-place
    assert transformed["basics"]["profiles"][0]["network"] == "GitHub"
    assert transformed["basics"]["profiles"][0]["username"] == "alex-dev"
    assert transformed["projects"][0]["name"] == ""
    assert transformed["projects"][0]["skills"] == ["Python", "Go"]

    role = load_role("software_engineering_intern")
    eval_model = build_evaluation_model(role)
    overshoot = _valid_intern_payload()
    overshoot["scores"]["open_source"]["score"] = 90
    overshoot["scores"]["open_source"]["max"] = 90
    eval_obj = eval_model(**overshoot)

    # Partial resume with city only (no region) must not emit "City, None"
    resume = JSONResume(
        basics=Basics(name="Alex", location=Location(city="London", region=None))
    )
    row = transform_evaluation_response(
        file_name="alex.pdf",
        resume_data=resume,
        evaluation=eval_obj,
        role=role,
    )
    assert row["location"] == "London"
    assert row["open_source_score"] == 35
    assert row["open_source_max"] == 35

    # Completely None resume & evaluation produces safe CSV defaults
    empty_row = transform_evaluation_response(
        file_name="empty.pdf", resume_data=None, evaluation=None, role=role
    )
    assert empty_row["name"] == ""
    assert empty_row["total_score"] == "N/A"
    assert empty_row["bonus_points"] == 0

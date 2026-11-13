"""Offline fairness perturbation, evidence span grounding, and calibration audit tests."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

import calibration_audit as ca
from models import (
    CategoryScore,
    Location,
    Project,
    build_evaluation_model,
)
from roles import Role, load_role
from transform import convert_json_resume_to_text


def test_role_rubric_integrity_and_mismatch_detection():
    role = load_role("software_engineering_intern")
    report = ca.audit_rubric_integrity(role)
    assert report.passed is True
    assert report.prompts_render_ok is True
    assert report.fairness_clauses_present is True
    assert report.missing_category_mentions == []
    assert report.base_max == 110
    assert report.max_final_score == 130

    # Synthetic broken role missing fairness instructions & overlapping bonus tiers must fail
    broken_role = Role(
        name="broken_test_role",
        position_title="Engineer",
        categories=role.categories,
        bonus_max=10,
        min_final_score=0,
        max_final_score=120,
        criteria_source="Evaluate {{ text_content }} only on open_source.",
        system_message_source="Score candidate quickly.",
        bonus_rules=[
            {
                "key": "contest",
                "label": "Contest",
                "tiers": [
                    {"min_rank": 1, "max_rank": 50, "points": 4},
                    {"min_rank": 40, "max_rank": 100, "points": 2},
                ],
                "description": "Overlapping tiers.",
            }
        ],
    )
    broken_report = ca.audit_rubric_integrity(broken_role)
    assert broken_report.passed is False
    assert broken_report.fairness_clauses_present is False
    assert len(broken_report.missing_category_mentions) == 4
    assert any("Overlapping rank tiers" in w for w in broken_report.warnings)


def test_calibration_cohort_ordering_tiers_and_bonus_cap():
    role = load_role("software_engineering_intern")
    cohort = ca.build_synthetic_cohort()
    provider = ca.DeterministicRubricProvider(role)

    results = []
    for fixture in cohort:
        evaluation, bd, norm, ev_audit = ca.evaluate_fixture_offline(
            fixture, role, provider=provider
        )
        results.append((fixture, evaluation, bd, norm, ev_audit))
        assert norm.tier == fixture.expected_tier
        assert ev_audit.passed is True, f"Evidence audit failed for {fixture.candidate_id}: {ev_audit}"
        assert ev_audit.leakage_flags == []
        assert ev_audit.rubric_violations == []

    # Fast-track > Advance > Borderline > Below Cutoff
    scores_by_id = {f.candidate_id: bd.total_score for f, _, bd, _, _ in results}
    assert scores_by_id["cand_05_bonus_cap_boundary"] > scores_by_id["cand_02_advance_builder"]
    assert scores_by_id["cand_01_fast_track"] > scores_by_id["cand_02_advance_builder"]
    assert scores_by_id["cand_02_advance_builder"] > scores_by_id["cand_03_borderline_intern"]
    assert scores_by_id["cand_03_borderline_intern"] > scores_by_id["cand_04_below_cutoff"]

    # cand_05 earns 26 raw bonus points across all 8 rules -> capped at bonus_max=20
    cand05_eval = results[4][1]
    cand05_bd = results[4][2]
    assert cand05_eval.bonus_points.total == 20
    assert cand05_bd.bonus_score == 20.0


def test_counterfactual_demographic_and_formatting_invariance():
    role = load_role("software_engineering_intern")
    cohort = ca.build_synthetic_cohort()

    cf_results, max_delta, tau, flip_rate = ca.audit_counterfactual_fairness(cohort, role)
    assert max_delta == 0.0
    assert tau == 1.0
    assert flip_rate == 0.0
    assert all(r.passed for r in cf_results)

    # Confirm both invariant perturbations and monotonic sensitivity perturbations ran
    kinds = {r.perturbation_kind for r in cf_results}
    assert {
        "demographic_swap",
        "institution_grade_swap",
        "formatting_noise",
        "achievement_added",
        "achievement_removed",
    }.issubset(kinds)

    added = [r for r in cf_results if r.perturbation_kind == "achievement_added"][0]
    removed = [r for r in cf_results if r.perturbation_kind == "achievement_removed"][0]
    assert added.score_delta > 0.0
    assert removed.score_delta < 0.0


def test_biased_provider_negative_control_is_caught():
    role = load_role("software_engineering_intern")
    cohort = ca.build_synthetic_cohort()

    def biased_factory(r):
        def inject_bias(raw_resume, payload):
            if "Stanford University" in raw_resume:
                payload["scores"]["open_source"]["score"] = min(
                    35, payload["scores"]["open_source"]["score"] + 10
                )
                payload["scores"]["open_source"]["evidence"] += " Attended Stanford University."
            if "(she/her)" in raw_resume:
                payload["scores"]["production"]["score"] = max(
                    0, payload["scores"]["production"]["score"] - 5
                )

        return ca.DeterministicRubricProvider(r, bias_hook=inject_bias)

    cf_results, max_delta, _, _ = ca.audit_counterfactual_fairness(
        cohort, role, provider_factory=biased_factory
    )
    assert max_delta > 0.0
    failed = [r for r in cf_results if r.expected_invariant and not r.passed]
    assert len(failed) >= 2


def test_evidence_span_grounding_leakage_and_rubric_violations():
    role = load_role("software_engineering_intern")
    cohort = ca.build_synthetic_cohort()
    fixture = cohort[2]  # cand_03_borderline_intern (all GitHub projects are self_project)
    resume_text = convert_json_resume_to_text(fixture.resume)

    evaluation, _, _, clean_audit = ca.evaluate_fixture_offline(fixture, role)
    assert clean_audit.passed is True
    assert any(m.span_offsets for m in clean_audit.matches)

    # 1. Inject hallucinated evidence + protected-attribute leakage + rubric violations
    eval_model = build_evaluation_model(role)
    dirty_dict = evaluation.model_dump()
    # computed_field values aren't accepted as input on BonusPoints; keep only items
    dirty_dict["bonus_points"] = {"items": dirty_dict["bonus_points"]["items"]}
    dirty_dict["scores"]["open_source"] = {
        "score": 25,  # violates <= 10 ceiling for self_project-only GitHub data!
        "max": 35,
        "evidence": "Maintained QuantumFluxEngine with 99999 stars; high CGPA from Stanford University.",
    }
    dirty_dict["bonus_points"]["items"]["gsoc"] = {
        "points": 5,
        "evidence": "Google Summer of Code 2025 contributor.",
    }
    dirty_dict["key_strengths"] = [
        "Active open source contributions",
        "Strong female candidate located in Toronto",
    ]
    dirty_eval = eval_model(**dirty_dict)

    dirty_audit = ca.audit_evidence_spans(
        dirty_eval, resume_text, role, github_data=fixture.github_data
    )
    assert dirty_audit.passed is False
    assert any("academic_grades" in f for f in dirty_audit.leakage_flags)
    assert any("institution_prestige" in f for f in dirty_audit.leakage_flags)
    assert any("demographic_pronoun_or_gender" in f for f in dirty_audit.leakage_flags)
    assert any("geographic_bias" in f for f in dirty_audit.leakage_flags)
    assert any("exceeds 10-point cap" in v for v in dirty_audit.rubric_violations)
    assert any("GSoC" in v for v in dirty_audit.rubric_violations)

    # 2. Verify "No evidence of significant open source" cannot co-exist with OSS key_strength
    contradict_dict = evaluation.model_dump()
    contradict_dict["bonus_points"] = {"items": contradict_dict["bonus_points"]["items"]}
    contradict_dict["scores"]["open_source"] = {
        "score": 4,
        "max": 35,
        "evidence": "No evidence of significant open source contributions beyond personal repos.",
    }
    contradict_dict["key_strengths"] = ["Open source projects"]
    contradict_eval = eval_model(**contradict_dict)
    contradict_audit = ca.audit_evidence_spans(contradict_eval, resume_text, role)
    assert contradict_audit.passed is False
    assert any("key_strengths" in v for v in contradict_audit.rubric_violations)


def test_full_audit_and_cli_execution(capsys):
    report = ca.run_full_audit("software_engineering_intern")
    assert report.passed is True
    assert report.tier_accuracy == 1.0
    assert report.rank_ordering_preserved is True
    assert report.biased_control_caught is True
    assert report.resilience_checks_passed == report.resilience_checks_total

    # CLI JSON output mode
    exit_code = ca.main(["--role", "software_engineering_intern", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    parsed = json.loads(captured)
    assert parsed["passed"] is True
    assert parsed["role_name"] == "software_engineering_intern"

    # CLI unknown role fails cleanly
    bad_exit = ca.main(["--role", "nonexistent_role"])
    assert bad_exit == 1

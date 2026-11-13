"""Offline rubric calibration, evidence span grounding, and fairness perturbation audit.

Evaluates rubric integrity, deterministic score aggregation/clamping, weighted
criteria normalization, tier classification, evidence-span grounding, protected-
attribute leakage, counterfactual demographic/formatting invariance, monotonic
achievement sensitivity, and malformed JSON/missing-field resilience — entirely
offline without paid LLM API calls.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from evaluator import ResumeEvaluator
from llm_utils import extract_json_from_response, parse_json_response
from models import (
    Award,
    Basics,
    Education,
    JSONResume,
    LLMProvider,
    Location,
    Profile,
    Project,
    Skill,
    Work,
    build_evaluation_model,
)
from prompts.template_manager import TemplateManager
from roles import Role, list_available_roles, load_role
from score import ScoreBreakdown, compute_evaluation_score
from transform import (
    convert_blog_data_to_text,
    convert_github_data_to_text,
    convert_json_resume_to_text,
    transform_evaluation_response,
)

TIER_STRONG = "Strong Fast-Track"
TIER_ADVANCE = "Advance to Review"
TIER_BORDERLINE = "Borderline / Manual Check"
TIER_BELOW_CUTOFF = "Below Cutoff"

_STOPWORDS = frozenset(
    {
        "about",
        "absent",
        "activity",
        "after",
        "against",
        "along",
        "also",
        "among",
        "an",
        "and",
        "any",
        "are",
        "areas",
        "awarded",
        "based",
        "beyond",
        "bonus",
        "built",
        "candidate",
        "category",
        "clearly",
        "code",
        "components",
        "concrete",
        "contributions",
        "could",
        "criteria",
        "data",
        "demonstrated",
        "depth",
        "details",
        "did",
        "does",
        "during",
        "each",
        "early-stage",
        "engineering",
        "enough",
        "evaluated",
        "evaluation",
        "evidence",
        "evidenced",
        "experience",
        "explicit",
        "features",
        "fluency",
        "for",
        "found",
        "from",
        "given",
        "have",
        "high",
        "impact",
        "implementation",
        "including",
        "into",
        "lack",
        "level",
        "limited",
        "listed",
        "matched",
        "missing",
        "more",
        "most",
        "multiple",
        "needed",
        "none",
        "not",
        "only",
        "other",
        "over",
        "overall",
        "participation",
        "personal",
        "points",
        "present",
        "production",
        "project",
        "projects",
        "provided",
        "published",
        "qualifies",
        "qualify",
        "qualifying",
        "quality",
        "rank",
        "real",
        "relevant",
        "reported",
        "resume",
        "role",
        "rubric",
        "score",
        "scored",
        "scores",
        "section",
        "several",
        "shows",
        "signal",
        "signals",
        "significant",
        "simple",
        "skills",
        "some",
        "stated",
        "strong",
        "such",
        "supporting",
        "system",
        "systems",
        "technical",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "through",
        "total",
        "under",
        "using",
        "verified",
        "very",
        "articles",
        "with",
        "without",
        "work",
        "work/production",
        "working",
        "zero",
    }
)

_NEGATIVE_EVIDENCE_PATTERNS = (
    re.compile(
        r"^no\s+(?:explicit\s+|stated\s+|verifiable\s+|production\s+|structured\s+)?.{0,45}"
        r"(?:evidence|experience|provided|listed|found|stated|present)\b",
        re.I,
    ),
    re.compile(r"\bnot\s+(?:evidenced|present|provided|found|stated)\b", re.I),
    re.compile(r"\babsent\b", re.I),
    re.compile(r"\bno\s+qualifying\b", re.I),
    re.compile(r"\bno\s+stated\s+rank\b", re.I),
    re.compile(r"\bmissing\b", re.I),
)

_LEAKAGE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "academic_grades",
        re.compile(r"\b(?:cgpa|gpa|grade point|academic grades?)\b", re.I),
    ),
    (
        "institution_prestige",
        re.compile(
            r"\b(?:ivy\s*league|tier[-\s]?[123]\s+(?:college|university|school|institution)|"
            r"(?:attended|graduated\s+from|from)\s+(?:stanford|mit|harvard|iit|nit|oxford|cambridge)|"
            r"prestigious\s+(?:college|university|institution))\b",
            re.I,
        ),
    ),
    (
        "demographic_pronoun_or_gender",
        re.compile(
            r"\b(?:he/him|she/her|they/them|male\s+candidate|female\s+candidate|"
            r"candidate(?:'s)?\s+(?:gender|sex|age|ethnicity|race|nationality))\b",
            re.I,
        ),
    ),
    (
        "geographic_bias",
        re.compile(
            r"\b(?:located\s+in|based\s+in|resident\s+of|candidate\s+location)\b",
            re.I,
        ),
    ),
)


@dataclass(frozen=True)
class TierThresholds:
    """Calibrated percentage thresholds (0-100 scale) for candidate tiering."""

    strong_hire: float = 80.0
    advance: float = 60.0
    borderline: float = 40.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.borderline <= self.advance <= self.strong_hire <= 100.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= borderline <= advance <= strong_hire <= 100"
            )


@dataclass(frozen=True)
class WeightedNormalizationResult:
    """Normalized weighted score breakdown and assigned rubric tier."""

    normalized_base_score: float
    normalized_final_score: float
    category_contributions: Dict[str, float]
    weights_used: Dict[str, float]
    tier: str


@dataclass(frozen=True)
class RubricIntegrityReport:
    """Integrity audit result for a role manifest and Jinja prompt templates."""

    role_name: str
    base_max: int
    bonus_max: int
    min_final_score: int
    max_final_score: int
    prompts_render_ok: bool
    missing_category_mentions: List[str]
    fairness_clauses_present: bool
    warnings: List[str]
    passed: bool


@dataclass(frozen=True)
class EvidenceSpanMatch:
    """Grounding and leakage check for one category or bonus evidence string."""

    category_or_item: str
    evidence_text: str
    matched_anchors: List[str]
    unmatched_anchors: List[str]
    grounding_ratio: float
    span_offsets: List[Tuple[int, int]]
    leakage_flags: List[str]


@dataclass(frozen=True)
class EvidenceAuditResult:
    """Aggregate evidence span grounding and fairness audit for one evaluation."""

    matches: List[EvidenceSpanMatch]
    mean_grounding_ratio: float
    leakage_flags: List[str]
    rubric_violations: List[str]
    passed: bool


@dataclass(frozen=True)
class SyntheticCandidateFixture:
    """Offline synthetic resume fixture with expected calibration band."""

    candidate_id: str
    expected_tier: str
    expected_rank_group: int
    resume: JSONResume
    github_data: Optional[Dict[str, Any]] = None
    blog_data: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass(frozen=True)
class CandidateCalibrationResult:
    """Calibration output for one synthetic candidate fixture."""

    candidate_id: str
    expected_tier: str
    assigned_tier: str
    base_score: float
    bonus_score: float
    deduction_score: float
    final_score: float
    normalized_base_score: float
    evidence_grounding_ratio: float
    leakage_count: int
    rubric_violations: List[str]


@dataclass(frozen=True)
class CounterfactualAuditResult:
    """Result of one counterfactual perturbation test against baseline evaluation."""

    baseline_id: str
    perturbation_name: str
    perturbation_kind: str
    baseline_score: float
    perturbed_score: float
    score_delta: float
    baseline_tier: str
    perturbed_tier: str
    expected_invariant: bool
    expected_direction: Optional[str]
    passed: bool
    details: str


@dataclass(frozen=True)
class ResilienceCheckResult:
    """Result of one malformed-input, clamping, or schema resilience check."""

    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class FullAuditReport:
    """End-to-end offline calibration, evidence, fairness, and resilience report."""

    role_name: str
    rubric_integrity: RubricIntegrityReport
    calibration_results: List[CandidateCalibrationResult]
    rank_ordering_preserved: bool
    tier_accuracy: float
    mean_grounding_ratio: float
    counterfactual_results: List[CounterfactualAuditResult]
    max_invariance_delta: float
    rank_kendall_tau: float
    tier_flip_rate: float
    biased_control_caught: bool
    resilience_results: List[ResilienceCheckResult]
    resilience_checks_passed: int
    resilience_checks_total: int
    passed: bool


def classify_tier(
    normalized_score: float, thresholds: Optional[TierThresholds] = None
) -> str:
    """Classify a 0-100 normalized score into a deterministic hiring triage tier."""
    th = thresholds or TierThresholds()
    if normalized_score >= th.strong_hire:
        return TIER_STRONG
    if normalized_score >= th.advance:
        return TIER_ADVANCE
    if normalized_score >= th.borderline:
        return TIER_BORDERLINE
    return TIER_BELOW_CUTOFF


def normalize_weighted_criteria(
    breakdown: ScoreBreakdown,
    role: Role,
    category_weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[TierThresholds] = None,
) -> WeightedNormalizationResult:
    """Normalize category scores to a 0-100 weighted index and classify tier.

    When ``category_weights`` is omitted, each category uses its proportional
    rubric weight (``category.max / sum(c.max)``). Custom weights are validated
    for non-negativity and normalized to sum to 1.0.
    """
    if not role.categories:
        raise ValueError(f"Role '{role.name}' has no categories to normalize")

    if category_weights is None:
        total_max = sum(c.max for c in role.categories)
        raw_weights = {c.key: c.max / total_max for c in role.categories}
    else:
        raw_weights = {}
        for cat in role.categories:
            if cat.key not in category_weights:
                raise ValueError(
                    f"Missing weight for category '{cat.key}' in custom category_weights"
                )
            weight = float(category_weights[cat.key])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"Weight for category '{cat.key}' must be finite and >= 0 (got {weight})"
                )
            raw_weights[cat.key] = weight

    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("Sum of category weights must be strictly positive")

    norm_weights = {k: v / weight_sum for k, v in raw_weights.items()}
    by_key = {cat.key: cat for cat in breakdown.categories}

    contributions: Dict[str, float] = {}
    normalized_base = 0.0
    for cat in role.categories:
        cat_bd = by_key.get(cat.key)
        ratio = (cat_bd.capped_score / cat.max) if (cat_bd and cat.max > 0) else 0.0
        contribution = ratio * norm_weights[cat.key] * 100.0
        contributions[cat.key] = round(contribution, 4)
        normalized_base += contribution

    score_span = role.max_final_score - role.min_final_score
    if score_span > 0:
        normalized_final = (
            (breakdown.total_score - role.min_final_score) / score_span
        ) * 100.0
    else:
        normalized_final = 0.0

    tier = classify_tier(normalized_base, thresholds=thresholds)
    return WeightedNormalizationResult(
        normalized_base_score=round(normalized_base, 2),
        normalized_final_score=round(normalized_final, 2),
        category_contributions=contributions,
        weights_used={k: round(v, 4) for k, v in norm_weights.items()},
        tier=tier,
    )


def audit_rubric_integrity(role: Role) -> RubricIntegrityReport:
    """Validate role bounds, category weights, bonus rules, and prompt template integrity."""
    warnings: List[str] = []
    base_max = sum(c.max for c in role.categories)
    expected_max = base_max + role.bonus_max
    if role.max_final_score != expected_max:
        warnings.append(
            f"max_final_score ({role.max_final_score}) differs from "
            f"sum(category.max)+bonus_max ({expected_max})"
        )
    if role.min_final_score > 0:
        warnings.append(
            f"min_final_score ({role.min_final_score}) > 0 may prevent zeroed scores"
        )

    tm = TemplateManager()
    prompts_render_ok = True
    rendered_criteria = ""
    rendered_system = ""
    try:
        rendered_criteria = tm.render_string(
            role.criteria_source, text_content="[SYNTHETIC_RESUME_PLACEHOLDER]"
        )
        rendered_system = tm.render_string(
            role.system_message_source,
            bonus_rules=role.bonus_rules,
            bonus_max=role.bonus_max,
        )
    except Exception as exc:
        prompts_render_ok = False
        warnings.append(f"Prompt template failed to render: {exc}")

    combined_prompt = f"{rendered_criteria}\n{rendered_system}".lower()
    missing_categories = [
        cat.key for cat in role.categories if cat.key.lower() not in combined_prompt
    ]
    if missing_categories:
        warnings.append(
            f"Categories missing from rendered prompt: {', '.join(missing_categories)}"
        )

    fairness_tokens = ("name", "gender", "location")
    fairness_clauses_present = all(tok in combined_prompt for tok in fairness_tokens)
    if not fairness_clauses_present:
        warnings.append("Prompt templates lack explicit demographic fairness clauses")

    for rule in role.bonus_rules:
        if "tiers" in rule:
            tiers = sorted(rule["tiers"], key=lambda t: t["min_rank"])
            for idx in range(1, len(tiers)):
                if tiers[idx]["min_rank"] <= tiers[idx - 1]["max_rank"]:
                    warnings.append(
                        f"Overlapping rank tiers in bonus rule '{rule['key']}'"
                    )

    passed = (
        prompts_render_ok
        and not missing_categories
        and fairness_clauses_present
        and not any("Overlapping" in w for w in warnings)
    )
    return RubricIntegrityReport(
        role_name=role.name,
        base_max=base_max,
        bonus_max=role.bonus_max,
        min_final_score=role.min_final_score,
        max_final_score=role.max_final_score,
        prompts_render_ok=prompts_render_ok,
        missing_category_mentions=missing_categories,
        fairness_clauses_present=fairness_clauses_present,
        warnings=warnings,
        passed=passed,
    )


def _extract_verifiable_anchors(evidence: str) -> List[str]:
    """Extract verifiable URLs, metrics, quoted spans, and domain tokens from evidence."""
    anchors: List[str] = []
    seen = set()

    def _add(token: str) -> None:
        norm = token.strip().strip(".,;:()[]{}\"'")
        low = norm.lower()
        if not norm or low in seen:
            return
        if len(norm) < 3 and not norm.isdigit():
            return
        if low in _STOPWORDS:
            return
        seen.add(low)
        anchors.append(norm)

    for match in re.findall(r"https?://[^\s,;)\]]+", evidence):
        _add(match)

    for match in re.findall(r'"([^"]{3,60})"|\'([^\']{3,60})\'', evidence):
        phrase = match[0] or match[1]
        _add(phrase)

    for match in re.findall(r"\b\d+(?:[.,]\d+)?%?(?:ms|k|m|x)?\b", evidence, re.I):
        if match not in {"0", "1", "2", "3", "4", "5", "10", "20", "25", "30", "35"}:
            _add(match)

    # Extract meaningful identifiers / technical nouns (skip leading rubric boilerplate).
    cleaned = re.sub(r"https?://\S+", " ", evidence)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_+.#/-]{2,}", cleaned):
        if token.lower() not in _STOPWORDS:
            _add(token)

    return anchors


def _find_anchor_span(anchor: str, source_text: str) -> Optional[Tuple[int, int]]:
    """Find character span of an anchor token in resume/enriched text."""
    idx = source_text.lower().find(anchor.lower())
    if idx != -1:
        return (idx, idx + len(anchor))

    # Retry comma-insensitive numeric match (e.g. "2400" vs "2,400").
    digits_only = anchor.replace(",", "")
    if digits_only.isdigit():
        pattern = re.compile(
            r"(?<!\d)" + r",?".join(re.escape(ch) for ch in digits_only) + r"(?!\d)"
        )
        m = pattern.search(source_text)
        if m:
            return m.span()
    return None


def _detect_leakage_in_text(text: str) -> List[str]:
    """Return protected-attribute leakage flag names triggered by ``text``."""
    flags: List[str] = []
    for label, pattern in _LEAKAGE_PATTERNS:
        if pattern.search(text):
            flags.append(label)
    return flags


def _is_explicit_absence_claim(evidence: str) -> bool:
    return any(pat.search(evidence) for pat in _NEGATIVE_EVIDENCE_PATTERNS)


def audit_evidence_spans(
    evaluation: Any,
    resume_text: str,
    role: Role,
    github_data: Optional[Dict[str, Any]] = None,
) -> EvidenceAuditResult:
    """Audit category & bonus evidence strings for source grounding, leakage, and rubric consistency."""
    matches: List[EvidenceSpanMatch] = []
    all_leakage_flags: List[str] = []
    rubric_violations: List[str] = []
    text_lower = resume_text.lower()

    scores_obj = getattr(evaluation, "scores", None)
    for category in role.categories:
        cat_entry = getattr(scores_obj, category.key, None) if scores_obj else None
        if cat_entry is None:
            rubric_violations.append(f"Missing score entry for category '{category.key}'")
            continue

        evidence = str(getattr(cat_entry, "evidence", ""))
        cat_score = min(float(getattr(cat_entry, "score", 0.0)), float(category.max))
        anchors = _extract_verifiable_anchors(evidence)
        matched: List[str] = []
        unmatched: List[str] = []
        spans: List[Tuple[int, int]] = []

        if cat_score == 0.0 and _is_explicit_absence_claim(evidence):
            grounding_ratio = 1.0
        elif not anchors:
            grounding_ratio = 1.0 if cat_score == 0.0 else 0.0
        else:
            for anchor in anchors:
                span = _find_anchor_span(anchor, resume_text)
                if span is not None:
                    matched.append(anchor)
                    spans.append(span)
                else:
                    unmatched.append(anchor)
            grounding_ratio = len(matched) / len(anchors)

        leak_flags = _detect_leakage_in_text(evidence)
        for flag in leak_flags:
            all_leakage_flags.append(f"{category.key}:{flag}")

        matches.append(
            EvidenceSpanMatch(
                category_or_item=category.key,
                evidence_text=evidence,
                matched_anchors=matched,
                unmatched_anchors=unmatched,
                grounding_ratio=round(grounding_ratio, 4),
                span_offsets=spans,
                leakage_flags=leak_flags,
            )
        )

    # Check bonus items evidence and grounding when bonus_rules are configured.
    bonus_obj = getattr(evaluation, "bonus_points", None)
    bonus_items = getattr(bonus_obj, "items", None) if bonus_obj is not None else None
    if bonus_items is not None:
        for rule in role.bonus_rules:
            key = rule["key"]
            item = getattr(bonus_items, key, None)
            if item is None:
                continue
            pts = int(getattr(item, "points", 0))
            ev = str(getattr(item, "evidence", ""))
            leak_flags = _detect_leakage_in_text(ev)
            for flag in leak_flags:
                all_leakage_flags.append(f"bonus.{key}:{flag}")

            if pts > 0:
                anchors = _extract_verifiable_anchors(ev)
                matched, unmatched, spans = [], [], []
                for anchor in anchors:
                    span = _find_anchor_span(anchor, resume_text)
                    if span is not None:
                        matched.append(anchor)
                        spans.append(span)
                    else:
                        unmatched.append(anchor)
                ratio = (len(matched) / len(anchors)) if anchors else 0.0
                matches.append(
                    EvidenceSpanMatch(
                        category_or_item=f"bonus.{key}",
                        evidence_text=ev,
                        matched_anchors=matched,
                        unmatched_anchors=unmatched,
                        grounding_ratio=round(ratio, 4),
                        span_offsets=spans,
                        leakage_flags=leak_flags,
                    )
                )

                # Rule-specific factual grounding checks.
                if key == "gsoc" and "google summer of code" not in text_lower and "gsoc" not in text_lower:
                    rubric_violations.append("Awarded GSoC bonus without GSoC evidence in resume")
                if key == "girl_script" and "girl script" not in text_lower and "girlscript" not in text_lower:
                    rubric_violations.append(
                        "Awarded Girl Script bonus without Girl Script evidence in resume"
                    )
                if key == "orchestrate":
                    rank = getattr(item, "rank", None)
                    if "orchestrate" not in text_lower or (
                        rank is not None and str(rank) not in text_lower
                    ):
                        rubric_violations.append(
                            f"Awarded Orchestrate bonus (rank={rank}) without matching span in resume"
                        )
                if key == "linkedin" and "linkedin" not in text_lower:
                    rubric_violations.append("Awarded LinkedIn bonus without LinkedIn profile")

    # Check list fields for protected-attribute leakage and OSS key_strengths rule.
    for field_name in ("key_strengths", "areas_for_improvement"):
        entries = getattr(evaluation, field_name, None) or []
        for idx, entry in enumerate(entries):
            for flag in _detect_leakage_in_text(str(entry)):
                all_leakage_flags.append(f"{field_name}[{idx}]:{flag}")

    deductions_obj = getattr(evaluation, "deductions", None)
    if deductions_obj is not None and getattr(deductions_obj, "reasons", ""):
        for flag in _detect_leakage_in_text(deductions_obj.reasons):
            all_leakage_flags.append(f"deductions:{flag}")

    # Enforce GitHub self_project open_source ceiling from role criteria.
    if github_data and isinstance(github_data.get("projects"), list) and github_data["projects"]:
        all_self = all(
            p.get("project_type") == "self_project" for p in github_data["projects"]
        )
        os_cat = getattr(scores_obj, "open_source", None) if scores_obj else None
        if all_self and os_cat is not None and float(os_cat.score) > 10.0:
            rubric_violations.append(
                f"open_source score ({os_cat.score}) exceeds 10-point cap when all GitHub projects are self_project"
            )

    os_entry = getattr(scores_obj, "open_source", None) if scores_obj else None
    if os_entry is not None:
        ev_low = os_entry.evidence.lower()
        no_oss_phrase = (
            "no evidence of significant open source" in ev_low
            or "no demonstrable open source" in ev_low
            or "personal repositories only" in ev_low
        )
        strengths = getattr(evaluation, "key_strengths", None) or []
        if no_oss_phrase and any("open source" in s.lower() for s in strengths):
            rubric_violations.append(
                "Listed open source in key_strengths despite evidence stating no significant OSS contributions"
            )

    mean_grounding = (
        sum(m.grounding_ratio for m in matches) / len(matches) if matches else 0.0
    )
    passed = (
        not all_leakage_flags
        and not rubric_violations
        and mean_grounding >= 0.70
    )
    return EvidenceAuditResult(
        matches=matches,
        mean_grounding_ratio=round(mean_grounding, 4),
        leakage_flags=all_leakage_flags,
        rubric_violations=rubric_violations,
        passed=passed,
    )


def _strip_protected_fields_for_scoring(resume_text: str) -> str:
    """Remove PII/demographic/institution/grade/location lines before rubric scoring.

    Mirrors the CRITICAL FAIRNESS REQUIREMENTS in ``system_message.jinja`` so the
    offline deterministic scorer is structurally blind to protected attributes.
    """
    filtered_lines: List[str] = []
    in_education = False
    for raw_line in resume_text.splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if upper.startswith("=== ") and upper.endswith(" ==="):
            in_education = upper == "=== EDUCATION ==="
            filtered_lines.append(raw_line)
            continue

        low = stripped.lower()
        if low.startswith(("name:", "email:", "phone:", "location:")):
            continue
        if in_education and low.startswith(("institution:", "score:")):
            continue

        # Strip explicit parenthetical pronouns while keeping technical text intact.
        cleaned = re.sub(
            r"\((?:he/him|she/her|they/them|he/they|she/they)\)",
            "",
            raw_line,
            flags=re.IGNORECASE,
        )
        filtered_lines.append(cleaned)
    return "\n".join(filtered_lines)


def _extract_resume_body_from_prompt(user_prompt: str) -> str:
    marker = "Resume to evaluate:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1].strip()
    return user_prompt


def _extract_section(text: str, header: str) -> str:
    pattern = re.compile(
        rf"===\s*{re.escape(header)}\s*===\s*(.*?)(?=\n===\s*[A-Z ]+\s*===|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_projects_entries(projects_section: str) -> List[Dict[str, Any]]:
    """Parse numbered entries from ``=== PROJECTS ===`` text cleanly."""
    if not projects_section:
        return []
    blocks = re.split(r"(?m)^\s*\d+\.\s+", projects_section)
    entries: List[Dict[str, Any]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        name = lines[0]
        full_text = " ".join(lines)
        has_url = bool(re.search(r"https?://\S+|github\.com/\S+", full_text, re.I))
        has_demo = bool(
            re.search(
                r"\b(?:live\s+demo|demo:|deployed\s+at|vercel\.app|netlify\.app|fly\.dev|render\.com|herokuapp\.com)\b",
                full_text,
                re.I,
            )
        )
        entries.append(
            {
                "name": name,
                "text": full_text,
                "has_url": has_url,
                "has_demo": has_demo,
            }
        )
    return entries


def _first_matching_phrase(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(0)
    return None


class DeterministicRubricProvider:
    """Offline ``LLMProvider`` scoring purely on technical evidence spans.

    Used by ``calibration_audit.py`` and unit tests to drive ``ResumeEvaluator``
    without external API calls while guaranteeing demographic & formatting invariance.
    """

    def __init__(
        self,
        role: Role,
        wrap_markdown: bool = False,
        include_think: bool = False,
        bias_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.role = role
        self.wrap_markdown = wrap_markdown
        self.include_think = include_think
        self.bias_hook = bias_hook

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del model, options, kwargs
        raw_user = messages[-1]["content"]
        raw_resume = _extract_resume_body_from_prompt(raw_user)
        sanitized_resume = _strip_protected_fields_for_scoring(raw_resume)
        payload = self._score_sanitized_resume(sanitized_resume)

        if self.bias_hook is not None:
            self.bias_hook(raw_resume, payload)

        serialized = json.dumps(payload, indent=2)
        if self.wrap_markdown:
            serialized = f"```json\n{serialized}\n```"
        if self.include_think:
            serialized = (
                "<think>Evaluating technical evidence against rubric categories.</think>\n\n"
                + serialized
            )
        return {"message": {"role": "assistant", "content": serialized}}

    def _score_sanitized_resume(self, text: str) -> Dict[str, Any]:
        basics_sec = _extract_section(text, "BASIC INFORMATION")
        work_sec = _extract_section(text, "WORK EXPERIENCE")
        volunteer_sec = _extract_section(text, "VOLUNTEER EXPERIENCE")
        skills_sec = _extract_section(text, "SKILLS")
        projects_sec = _extract_section(text, "PROJECTS")
        awards_sec = _extract_section(text, "AWARDS")
        github_sec = _extract_section(text, "GITHUB DATA")
        blog_sec = _extract_section(text, "BLOG DATA")

        combined_tech = "\n".join(
            part
            for part in (
                work_sec,
                volunteer_sec,
                skills_sec,
                projects_sec,
                awards_sec,
                github_sec,
                blog_sec,
            )
            if part
        )
        lower_tech = combined_tech.lower()

        # 1. Open Source (0-35)
        oss_score = 0
        oss_anchors: List[str] = []

        star_nums = [
            int(s.replace(",", ""))
            for s in re.findall(r"(?:Stars:\s*|with\s+)(\d[\d,]*)\s*(?:\+)?\s*(?:stars)?", combined_tech, re.I)
            if s.replace(",", "").isdigit()
        ]
        max_stars = max(star_nums, default=0)
        has_gsoc = bool(re.search(r"\b(?:google summer of code|gsoc)\b", lower_tech))
        has_girlscript = bool(
            re.search(r"\b(?:girl\s*script summer of code|girlscript|gssoc)\b", lower_tech)
        )
        has_ext_pr = bool(
            re.search(
                r"\b(?:merged\s+\d*\s*prs?|pull\s+requests?\s+to|upstream\s+prs?|maintainer\s+of|open[- ]source\s+contributor)\b",
                lower_tech,
            )
        )
        has_github_url = "github.com/" in text.lower()
        has_hacktoberfest_only = "hacktoberfest" in lower_tech and not has_ext_pr and not has_gsoc

        if has_gsoc:
            oss_score += 14
            m = _first_matching_phrase(combined_tech, [r"Google Summer of Code(?:\s*\(GSoC\))?", r"GSoC"])
            if m:
                oss_anchors.append(m)
        if has_ext_pr:
            oss_score += 12
            m = _first_matching_phrase(
                combined_tech,
                [
                    r"\d+\s+merged\s+PRs?[^.\n]*",
                    r"merged\s+PRs?[^.\n]*",
                    r"maintainer\s+of\s+[^.\n]+",
                    r"pull\s+requests?\s+to\s+[^.\n]+",
                ],
            )
            if m:
                oss_anchors.append(m.strip())
        if max_stars >= 1000:
            oss_score += 9
            oss_anchors.append(f"{max_stars} stars")
        elif max_stars >= 100:
            oss_score += 5
            oss_anchors.append(f"{max_stars} stars")
        if has_girlscript:
            oss_score += 6
            m = _first_matching_phrase(combined_tech, [r"Girl\s*Script\s+Summer\s+of\s+Code"])
            if m:
                oss_anchors.append(m)
        if has_hacktoberfest_only:
            oss_score = max(oss_score, 4)
            oss_anchors.append("Hacktoberfest")
        elif oss_score == 0 and has_github_url:
            oss_score = 4 if github_sec else 2
            gh_url = _first_matching_phrase(text, [r"https?://github\.com/[A-Za-z0-9_-]+"])
            if gh_url:
                oss_anchors.append(gh_url)

        oss_score = min(oss_score, 35)
        if oss_anchors:
            oss_evidence = f"Verified signals: {'; '.join(oss_anchors[:3])}."
        else:
            oss_evidence = "No evidence of significant open source contributions or GitHub links."

        # 2. Self Projects (0-30)
        project_entries = _parse_projects_entries(projects_sec)
        complex_patterns = [
            r"\bmicroservices?\b",
            r"\bdistributed\b",
            r"\breal-time\b",
            r"\bwebsocket\b",
            r"\bauthentication\b",
            r"\bdatabase\b",
            r"\bpostgresql\b",
            r"\bredis\b",
            r"\bkafka\b",
            r"\bkubernetes\b",
            r"\bcompiler\b",
            r"\bsearch\s+engine\b",
            r"\brag\b",
            r"\bllm\b",
            r"\bagent\b",
            r"\bevaluation\s+harness\b",
            r"\bguardrails?\b",
            r"\bconcurrent\b",
            r"\bthroughput\b",
            r"\blatency\b",
            r"\bactive\s+users?\b",
        ]
        simple_patterns = [
            r"\btodo\s*(?:list|app)?\b",
            r"\bcalculator\b",
            r"\bweather\s+app\b",
            r"\bbasic\s+crud\b",
            r"\bnote-taking\b",
            r"\bclassroom\s+assignment\b",
            r"\btutorial\b",
        ]

        proj_score = 0
        proj_anchors: List[str] = []
        simple_only_count = 0
        missing_url_count = 0
        missing_demo_count = 0

        for entry in project_entries:
            etext = entry["text"]
            complex_hits = sum(1 for pat in complex_patterns if re.search(pat, etext, re.I))
            is_simple = any(re.search(pat, etext, re.I) for pat in simple_patterns)
            if not entry["has_url"]:
                missing_url_count += 1
            elif not entry["has_demo"]:
                missing_demo_count += 1

            if is_simple and complex_hits == 0:
                simple_only_count += 1
                proj_score += 2
                proj_anchors.append(entry["name"])
            else:
                pts = min(13, 5 + complex_hits * 2)
                if entry["has_url"]:
                    pts += 1
                if entry["has_demo"]:
                    pts += 2
                proj_score += pts
                proj_anchors.append(entry["name"])

        if project_entries and simple_only_count == len(project_entries):
            proj_score = min(proj_score, 8)
        proj_score = min(proj_score, 30)

        if proj_anchors:
            proj_evidence = f"Evaluated projects: {', '.join(proj_anchors[:4])}."
        else:
            proj_evidence = "No projects section provided."

        # 3. Production Experience (0-25)
        prod_source = f"{work_sec}\n{volunteer_sec}".strip()
        prod_score = 0
        prod_anchors: List[str] = []
        if prod_source:
            role_matches = re.findall(r"(?m)^(?:\d+\.|•)\s+([^\n]+)", prod_source)
            quant_matches = re.findall(
                r"\b\d+(?:\.\d+)?(?:%|ms|x|k\+?|m\+?)\b[^.\n]*", prod_source, re.I
            )
            has_founder = bool(re.search(r"\b(?:co-founder|founder)\b", prod_source, re.I))
            has_early_eng = bool(
                re.search(
                    r"\b(?:founding\s+engineer|early[- ]stage\s+engineer|first\s+(?:5|10|15|20)\s+engineers?)\b",
                    prod_source,
                    re.I,
                )
            )
            has_intern_or_eng = bool(
                re.search(
                    r"\b(?:intern|engineer|developer|architect|sde)\b", prod_source, re.I
                )
            )
            if has_intern_or_eng:
                prod_score += min(12, 6 * max(1, len(role_matches)))
            else:
                prod_score += min(6, 3 * max(1, len(role_matches)))
            prod_score += min(8, len(quant_matches) * 4)
            if has_founder:
                prod_score += 5
            if has_early_eng:
                prod_score += 4

            for r in role_matches[:2]:
                prod_anchors.append(r.strip())
            for q in quant_matches[:2]:
                prod_anchors.append(q.strip())

        prod_score = min(prod_score, 25)
        if prod_anchors:
            prod_evidence = f"Work/production evidence: {'; '.join(prod_anchors)}."
        else:
            prod_evidence = "No production or internship work experience provided."

        # 4. Technical Skills (0-10)
        keyword_lines = re.findall(r"Keywords:\s*([^\n]+)", skills_sec)
        skill_tokens = [
            tok.strip()
            for line in keyword_lines
            for tok in line.split(",")
            if tok.strip()
        ]
        unique_skills = sorted({s.lower(): s for s in skill_tokens}.values())
        algo_signal = bool(
            re.search(
                r"\b(?:algorithms?|data\s+structures|concurrency|distributed|c\+\+|rust|go|python|typescript)\b",
                lower_tech,
            )
        )
        skill_score = min(10, (len(unique_skills) // 2) + (2 if algo_signal and unique_skills else 0))
        if unique_skills:
            skill_evidence = f"Demonstrated skills: {', '.join(unique_skills[:6])}."
        else:
            skill_evidence = "No structured technical skills listed."

        # 5. AI Fluency (0-10)
        ai_build_hits = len(
            re.findall(
                r"\b(?:rag|retrieval[- ]augmented|embeddings?|vector\s+search|llm\s+agents?|tool\s+use|"
                r"structured\s+outputs?|fine[- ]tuning|multimodal|pydantic|prompt\s+design)\b",
                lower_tech,
            )
        )
        ai_eval_hits = len(
            re.findall(
                r"\b(?:evals?|evaluation\s+harness|benchmark|hallucination|guardrails?|fallback|"
                r"precision|recall|f1|latency|cost\s+tradeoff|test\s+set|verification)\b",
                lower_tech,
            )
        )
        ai_keyword_only = bool(
            re.search(r"\b(?:ai|llm|openai|langchain|chatgpt|machine learning)\b", lower_tech)
        )
        if ai_build_hits >= 2 and ai_eval_hits >= 2:
            ai_score = min(10, 7 + min(3, ai_eval_hits))
        elif ai_build_hits >= 1:
            ai_score = min(7, 3 + ai_build_hits + ai_eval_hits)
        elif ai_keyword_only:
            ai_score = 2
        else:
            ai_score = 0

        if ai_score > 0:
            ai_phrases = re.findall(
                r"\b(?:RAG|embeddings?|agents?|tool\s+use|structured\s+outputs?|fine-tuning|"
                r"guardrails?|evals?|evaluation\s+harness|hallucination|fallback|ChatGPT|LangChain)\b",
                combined_tech,
                re.I,
            )
            unique_ai = list(dict.fromkeys(ai_phrases))[:4]
            if unique_ai:
                ai_evidence = f"AI fluency signals: {', '.join(unique_ai)}."
            else:
                ai_evidence = "Minimal AI library mention without evaluation workflow."
        else:
            ai_evidence = "No evidence of building or evaluating AI systems."

        # 6. Bonus Points (driven by role.bonus_rules when present)
        bonus_payload = self._build_bonus_payload(
            text=text,
            basics_sec=basics_sec,
            work_sec=work_sec,
            awards_sec=awards_sec,
            blog_sec=blog_sec,
            combined_tech=combined_tech,
            has_gsoc=has_gsoc,
            has_girlscript=has_girlscript,
        )

        # 7. Deductions
        deduction_total = 0.0
        deduction_reasons: List[str] = []
        if project_entries and simple_only_count == len(project_entries):
            deduction_total += 4.0
            deduction_reasons.append(
                f"Only simple tutorial/CRUD projects ({simple_only_count} total)"
            )
        if missing_url_count > 0:
            deduction_total += float(min(8, missing_url_count * 3))
            deduction_reasons.append(
                f"{missing_url_count} project(s) missing GitHub or URL"
            )
        elif missing_demo_count > 0 and prod_score < 12:
            deduction_total += float(min(4, missing_demo_count * 2))
            deduction_reasons.append(
                f"{missing_demo_count} project(s) without live demo link"
            )
        if not deduction_reasons:
            deduction_reasons.append("No rubric deductions triggered.")

        # 8. Strengths & Improvements (respecting OSS key_strengths rule!)
        key_strengths: List[str] = []
        if oss_score >= 15 and (has_ext_pr or has_gsoc or max_stars >= 100):
            key_strengths.append(f"Verified open source contributions ({oss_evidence})")
        if proj_score >= 16:
            key_strengths.append(f"Complex technical projects ({proj_evidence})")
        if prod_score >= 12:
            key_strengths.append(f"Production engineering impact ({prod_evidence})")
        if ai_score >= 6:
            key_strengths.append(f"Strong AI engineering & evaluation ({ai_evidence})")
        if skill_score >= 7:
            key_strengths.append(f"Broad technical stack ({skill_evidence})")
        if not key_strengths:
            key_strengths.append("Demonstrates foundational programming familiarity")

        improvements: List[str] = []
        if oss_score <= 10:
            improvements.append("Contribute pull requests to external open source repositories")
        if prod_score < 10:
            improvements.append("Add production internship or real-world deployment metrics")
        if ai_score < 4:
            improvements.append("Demonstrate AI workflows with evaluation metrics and guardrails")
        if missing_url_count > 0:
            improvements.append("Include active repository and demo links for listed projects")
        if not improvements:
            improvements.append("Continue expanding measurable user impact across systems")

        # Assemble scores dict matching role.categories dynamically
        base_map = {
            "open_source": (oss_score, 35, oss_evidence),
            "self_projects": (proj_score, 30, proj_evidence),
            "production": (prod_score, 25, prod_evidence),
            "technical_skills": (skill_score, 10, skill_evidence),
            "ai_fluency": (ai_score, 10, ai_evidence),
        }
        scores_dict: Dict[str, Dict[str, Any]] = {}
        for cat in self.role.categories:
            if cat.key in base_map:
                sc, default_max, ev = base_map[cat.key]
                scaled = round(min(sc, default_max) * (cat.max / default_max), 2)
                scores_dict[cat.key] = {
                    "score": min(scaled, float(cat.max)),
                    "max": cat.max,
                    "evidence": ev,
                }
            else:
                scores_dict[cat.key] = {
                    "score": min(float(cat.max), round(cat.max * 0.5, 2)),
                    "max": cat.max,
                    "evidence": f"Evaluated {cat.label} against provided resume evidence.",
                }

        return {
            "scores": scores_dict,
            "bonus_points": bonus_payload,
            "deductions": {
                "total": deduction_total,
                "reasons": "; ".join(deduction_reasons),
            },
            "key_strengths": key_strengths[:5],
            "areas_for_improvement": improvements[:3],
        }

    def _build_bonus_payload(
        self,
        text: str,
        basics_sec: str,
        work_sec: str,
        awards_sec: str,
        blog_sec: str,
        combined_tech: str,
        has_gsoc: bool,
        has_girlscript: bool,
    ) -> Dict[str, Any]:
        if not self.role.bonus_rules:
            return {"total": 0, "breakdown": "No role bonus rules configured."}

        items: Dict[str, Dict[str, Any]] = {}
        lower_text = text.lower()
        orchestrate_ranks = [
            int(m)
            for m in re.findall(
                r"HackerRank\s+Orchestrate[^\n]{0,80}?(?:rank|#|place)\s*#?\s*(\d+)",
                f"{awards_sec}\n{combined_tech}",
                re.I,
            )
        ]
        best_orch_rank = min(orchestrate_ranks) if orchestrate_ranks else None

        for rule in self.role.bonus_rules:
            key = rule["key"]
            if key == "gsoc":
                pts = 5 if has_gsoc else 0
                ev = (
                    "Google Summer of Code participation evidenced."
                    if pts
                    else "Google Summer of Code absent from resume."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "girl_script":
                pts = 3 if has_girlscript else 0
                ev = (
                    "Girl Script Summer of Code participation evidenced."
                    if pts
                    else "Girl Script Summer of Code absent from resume."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "orchestrate":
                pts = 0
                if best_orch_rank is not None:
                    for tier in rule["tiers"]:
                        if tier["min_rank"] <= best_orch_rank <= tier["max_rank"]:
                            pts = tier["points"]
                            break
                ev = (
                    f"HackerRank Orchestrate rank {best_orch_rank} verified."
                    if best_orch_rank is not None
                    else "No stated HackerRank Orchestrate rank."
                )
                items[key] = {"rank": best_orch_rank, "points": pts, "evidence": ev}
            elif key == "founder":
                founder_hit = re.search(
                    r"\b(?:startup\s+founder|co-founder|founder)\b",
                    f"{work_sec}\n{combined_tech}",
                    re.I,
                )
                pts = 5 if founder_hit else 0
                ev = (
                    f"Startup {founder_hit.group(0)} role verified."
                    if founder_hit
                    else "No startup founder/co-founder role stated."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "early_engineer":
                early_hit = re.search(
                    r"\b(?:founding\s+engineer|early[- ]stage\s+engineer|first\s+(?:5|10|15|20)\s+engineers?)\b",
                    f"{work_sec}\n{combined_tech}",
                    re.I,
                )
                pts = 3 if early_hit else 0
                ev = (
                    f"Early-stage startup signal: {early_hit.group(0)}."
                    if early_hit
                    else "No early-stage engineer evidence."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "portfolio":
                portfolio_hit = re.search(
                    r"(?:Portfolio:\s*(https?://\S+)|Website:\s*(https?://(?!github\.com|linkedin\.com)\S+))",
                    basics_sec,
                    re.I,
                )
                pts = 2 if portfolio_hit else 0
                url_found = (
                    (portfolio_hit.group(1) or portfolio_hit.group(2))
                    if portfolio_hit
                    else ""
                )
                ev = (
                    f"Personal portfolio website {url_found}."
                    if pts
                    else "No explicit personal portfolio website."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "linkedin":
                li_hit = re.search(r"linkedin\.com/\S+", basics_sec, re.I)
                pts = 1 if li_hit else 0
                ev = (
                    f"LinkedIn profile {li_hit.group(0)}."
                    if li_hit
                    else "No LinkedIn profile in resume basics."
                )
                items[key] = {"points": pts, "evidence": ev}
            elif key == "technical_blogs":
                has_blogs = bool(blog_sec and "total blogs found: 0" not in blog_sec.lower())
                pts = 3 if has_blogs else 0
                ev = (
                    "Technical blog data provided with published articles."
                    if pts
                    else "No external blog data provided."
                )
                items[key] = {"points": pts, "evidence": ev}
            else:
                if "tiers" in rule:
                    items[key] = {
                        "rank": None,
                        "points": 0,
                        "evidence": f"No evidence for {rule['label']}.",
                    }
                else:
                    items[key] = {
                        "points": 0,
                        "evidence": f"No evidence for {rule['label']}.",
                    }

        return {"items": items}


def build_synthetic_cohort() -> List[SyntheticCandidateFixture]:
    """Create 5 realistic synthetic resume fixtures spanning all rubric tiers."""
    strong_oss = SyntheticCandidateFixture(
        candidate_id="cand_01_fast_track",
        expected_tier=TIER_STRONG,
        expected_rank_group=1,
        description="GSoC + 2,400-star OSS maintainer + latency-verified internship + RAG evals + Orchestrate #42",
        resume=JSONResume(
            basics=Basics(
                name="Aarav Sharma (he/him)",
                email="aarav.sharma@example.edu",
                phone="+91-98000-11111",
                url="https://aaravsharma.dev",
                summary="Systems and ML engineering student building distributed search and verified AI tooling.",
                location=Location(city="Jaipur", region="RJ", countryCode="IN"),
                profiles=[
                    Profile(
                        network="GitHub",
                        username="aarav-oss",
                        url="https://github.com/aarav-oss",
                    ),
                    Profile(
                        network="LinkedIn",
                        username="aarav-sharma",
                        url="https://linkedin.com/in/aarav-sharma",
                    ),
                ],
            ),
            work=[
                Work(
                    name="CloudScale Systems",
                    position="Software Engineering Intern",
                    startDate="May 2025",
                    endDate="Aug 2025",
                    summary="Built concurrent Go/Rust indexing worker and query cache.",
                    highlights=[
                        "Reduced p95 search latency by 68% (420ms to 134ms) across 45k RPS traffic.",
                        "Added automated load regression suite catching 14 memory leaks pre-release.",
                    ],
                )
            ],
            education=[
                Education(
                    institution="State Technical Institute",
                    studyType="B.Tech",
                    area="Computer Science",
                    startDate="2022-08",
                    endDate="2026-05",
                    score="7.4/10",
                )
            ],
            skills=[
                Skill(
                    name="Languages",
                    keywords=["Python", "Go", "Rust", "TypeScript", "SQL"],
                ),
                Skill(
                    name="Systems & AI",
                    keywords=[
                        "PostgreSQL",
                        "Redis",
                        "Kafka",
                        "Kubernetes",
                        "RAG",
                        "PyTorch",
                    ],
                ),
            ],
            projects=[
                Project(
                    name="VectorGuard Query Engine",
                    url="https://github.com/aarav-oss/vectorguard",
                    description="Distributed RAG retrieval engine with hybrid BM25 + embeddings, structured outputs, hallucination guardrails, and an automated evaluation harness deployed at live demo https://vectorguard.vercel.app.",
                    highlights=[
                        "Achieved 0.91 precision@5 on 1,200-query benchmark with deterministic fallback retries.",
                        "Used by 1,800 active users with PostgreSQL + Redis authentication.",
                    ],
                ),
                Project(
                    name="AsyncTrace Profiler",
                    url="https://github.com/aarav-oss/asynctrace",
                    description="Real-time eBPF and WebSocket profiler with microservices visualization and live demo https://asynctrace.fly.dev.",
                    highlights=["14 merged PRs from external community contributors."],
                ),
            ],
            awards=[
                Award(
                    title="Google Summer of Code (GSoC) Contributor",
                    awarder="Google / Apache Arrow",
                    date="2025-08",
                    summary="Shipped 18 merged PRs to upstream open-source Arrow columnar engine (2,400 stars).",
                ),
                Award(
                    title="HackerRank Orchestrate Finalist",
                    awarder="HackerRank",
                    date="2025-10",
                    summary="Placed rank #42 overall in HackerRank Orchestrate systems challenge.",
                ),
            ],
        ),
        github_data={
            "profile": {"username": "aarav-oss", "public_repos": 28, "followers": 340},
            "projects": [
                {
                    "name": "arrow-rs-upstream",
                    "description": "Upstream PRs for Arrow columnar query execution",
                    "github_url": "https://github.com/apache/arrow-rs",
                    "project_type": "open_source",
                    "github_details": {"stars": 2400, "forks": 610, "language": "Rust"},
                },
                {
                    "name": "vectorguard",
                    "description": "RAG retrieval and evaluation engine",
                    "github_url": "https://github.com/aarav-oss/vectorguard",
                    "project_type": "self_project",
                    "github_details": {"stars": 190, "forks": 24, "language": "Python"},
                },
            ],
        },
        blog_data={
            "total_blogs": 4,
            "blog_score": 8.5,
            "blog_details": "Deep dives on tail-latency profiling and RAG evals",
            "blogs": [
                {
                    "url": "https://aaravsharma.dev/blog/rag-evals",
                    "score": 8.5,
                    "details": "Retrieval precision and hallucination guardrail design",
                }
            ],
        },
    )

    strong_builder = SyntheticCandidateFixture(
        candidate_id="cand_02_advance_builder",
        expected_tier=TIER_ADVANCE,
        expected_rank_group=2,
        description="Early-stage engineer + solid OSS PRs + real-time distributed project + AI agent evals",
        resume=JSONResume(
            basics=Basics(
                name="Maya Lin (she/her)",
                email="maya.lin@example.com",
                url="https://mayalin.io",
                summary="Full-stack and AI systems builder focused on reliable agent workflows.",
                location=Location(city="Austin", region="TX", countryCode="US"),
                profiles=[
                    Profile(
                        network="GitHub",
                        username="mayalin-dev",
                        url="https://github.com/mayalin-dev",
                    ),
                    Profile(
                        network="LinkedIn",
                        username="maya-lin",
                        url="https://linkedin.com/in/maya-lin",
                    ),
                ],
            ),
            work=[
                Work(
                    name="PulseGrid AI",
                    position="Founding Engineer (First 10 Engineers)",
                    startDate="Jan 2025",
                    endDate="Present",
                    summary="Built multi-tenant webhook ingestion and tool-use agent pipeline.",
                    highlights=[
                        "Cut LLM JSON parse failures by 92% via structured outputs and fallback retries.",
                        "Scaled PostgreSQL + Kafka workers to 12k events/min at 99.9% uptime.",
                    ],
                )
            ],
            education=[
                Education(
                    institution="Regional Engineering College",
                    studyType="B.S.",
                    area="Software Engineering",
                    startDate="2022-09",
                    endDate="2026-06",
                    score="3.4/4.0",
                )
            ],
            skills=[
                Skill(
                    name="Languages",
                    keywords=["Python", "TypeScript", "Go", "SQL", "Rust"],
                ),
                Skill(
                    name="Stack",
                    keywords=[
                        "FastAPI",
                        "PostgreSQL",
                        "Kafka",
                        "Redis",
                        "Docker",
                        "LLM Agents",
                        "Kubernetes",
                    ],
                ),
            ],
            projects=[
                Project(
                    name="SchemaShield Agent Router",
                    url="https://github.com/mayalin-dev/schemashield",
                    description="AI agent orchestration with tool use, structured outputs, guardrails, and evaluation harness benchmark test set at live demo https://schemashield.netlify.app.",
                    highlights=[
                        "340 stars and 6 merged PRs to upstream Pydantic tooling."
                    ],
                ),
                Project(
                    name="StreamPulse Distributed Queue",
                    url="https://github.com/mayalin-dev/streampulse",
                    description="Real-time distributed webhook router with Redis + PostgreSQL persistence and authentication at live demo https://streampulse.fly.dev.",
                    highlights=[
                        "Sustained 8k RPS throughput with automated latency benchmarks."
                    ],
                ),
            ],
            awards=[
                Award(
                    title="Girl Script Summer of Code",
                    awarder="GirlScript Foundation",
                    date="2024-08",
                    summary="Contributed 7 merged PRs to community accessibility tooling.",
                )
            ],
        ),
        github_data={
            "profile": {"username": "mayalin-dev", "public_repos": 16, "followers": 95},
            "projects": [
                {
                    "name": "pydantic-contrib",
                    "description": "Upstream PRs to schema validator",
                    "github_url": "https://github.com/pydantic/pydantic",
                    "project_type": "open_source",
                    "github_details": {"stars": 340, "forks": 40, "language": "Python"},
                }
            ],
        },
    )

    mid_borderline = SyntheticCandidateFixture(
        candidate_id="cand_03_borderline_intern",
        expected_tier=TIER_BORDERLINE,
        expected_rank_group=3,
        description="One internship + two full-stack projects (one missing live demo) + basic AI feature",
        resume=JSONResume(
            basics=Basics(
                name="Devon Brooks (they/them)",
                email="devon.brooks@example.com",
                location=Location(city="Toronto", region="ON", countryCode="CA"),
                profiles=[
                    Profile(
                        network="GitHub",
                        username="devon-b",
                        url="https://github.com/devon-b",
                    ),
                    Profile(
                        network="LinkedIn",
                        username="devon-brooks",
                        url="https://linkedin.com/in/devon-brooks",
                    ),
                ],
            ),
            work=[
                Work(
                    name="BrightRetail Corp",
                    position="Junior Software Intern",
                    startDate="May 2025",
                    endDate="Aug 2025",
                    summary="Maintained REST endpoints and dashboard queries.",
                    highlights=["Improved checkout query latency by 25% using SQL indexes."],
                )
            ],
            education=[
                Education(
                    institution="Northern Polytech",
                    studyType="B.Sc.",
                    area="Computer Science",
                    startDate="2023-09",
                    endDate="2027-05",
                    score="3.2/4.0",
                )
            ],
            skills=[
                Skill(name="Languages", keywords=["Python", "JavaScript", "SQL"]),
                Skill(name="Tools", keywords=["React", "Node.js", "PostgreSQL", "Docker"]),
            ],
            projects=[
                Project(
                    name="CollabBoard Real-Time Canvas",
                    url="https://github.com/devon-b/collabboard",
                    description="Real-time WebSocket whiteboard with authentication and PostgreSQL persistence, live demo https://collabboard.vercel.app.",
                ),
                Project(
                    name="DocSearch Embeddings Prototype",
                    url="https://github.com/devon-b/docsearch",
                    description="RAG search prototype using embeddings and FastAPI.",
                ),
            ],
        ),
        github_data={
            "profile": {"username": "devon-b", "public_repos": 9, "followers": 14},
            "projects": [
                {
                    "name": "collabboard",
                    "description": "Personal real-time whiteboard",
                    "github_url": "https://github.com/devon-b/collabboard",
                    "project_type": "self_project",
                    "github_details": {"stars": 6, "forks": 1, "language": "TypeScript"},
                }
            ],
        },
    )

    tutorial_low = SyntheticCandidateFixture(
        candidate_id="cand_04_below_cutoff",
        expected_tier=TIER_BELOW_CUTOFF,
        expected_rank_group=4,
        description="Tutorial/CRUD-only classroom projects without URLs, no OSS, no work, keyword AI only",
        resume=JSONResume(
            basics=Basics(
                name="Samira Patel",
                email="samira.patel@example.com",
                location=Location(city="Mumbai", region="MH", countryCode="IN"),
                profiles=[],
            ),
            education=[
                Education(
                    institution="Metro Engineering College",
                    studyType="B.E.",
                    area="Information Technology",
                    startDate="2023-08",
                    endDate="2027-05",
                    score="8.1/10",
                )
            ],
            skills=[
                Skill(name="Languages", keywords=["Python", "HTML", "CSS"]),
                Skill(name="AI", keywords=["ChatGPT", "OpenAI"]),
            ],
            projects=[
                Project(
                    name="Todo List CRUD App",
                    description="Classroom assignment basic CRUD todo list and calculator app.",
                ),
                Project(
                    name="Weather App Tutorial",
                    description="Tutorial weather app using public REST API and basic ChatGPT wrapper.",
                ),
            ],
        ),
    )

    bonus_cap_edge = SyntheticCandidateFixture(
        candidate_id="cand_05_bonus_cap_boundary",
        expected_tier=TIER_STRONG,
        expected_rank_group=1,
        description="Triggers >20 raw bonus points (26 total) to verify 20-point bonus_max cap & Orchestrate tiers",
        resume=JSONResume(
            basics=Basics(
                name="Elena Novak",
                email="elena.novak@example.org",
                url="https://elenanovak.dev",
                summary="Startup founder and distributed systems engineer.",
                profiles=[
                    Profile(
                        network="GitHub",
                        username="enovak",
                        url="https://github.com/enovak",
                    ),
                    Profile(
                        network="LinkedIn",
                        username="elena-novak",
                        url="https://linkedin.com/in/elena-novak",
                    ),
                ],
            ),
            work=[
                Work(
                    name="TraceMesh Labs",
                    position="Co-Founder & Founding Engineer (First 10 Engineers)",
                    startDate="Jan 2024",
                    endDate="Present",
                    summary="Built multi-region telemetry ingestion and distributed query compiler.",
                    highlights=[
                        "Cut query latency by 75% across 30k RPS production traffic for 4,500 active users."
                    ],
                )
            ],
            skills=[
                Skill(
                    name="Core",
                    keywords=[
                        "Rust",
                        "Go",
                        "Python",
                        "Kubernetes",
                        "PostgreSQL",
                        "Kafka",
                        "RAG",
                        "PyTorch",
                    ],
                )
            ],
            projects=[
                Project(
                    name="EvalMesh Telemetry",
                    url="https://github.com/enovak/evalmesh",
                    description="Distributed RAG agent platform with structured outputs, guardrails, evaluation harness, and live demo https://evalmesh.fly.dev.",
                ),
                Project(
                    name="KubeTrace Log Compiler",
                    url="https://github.com/enovak/kubetrace",
                    description="High-throughput Kubernetes real-time log compiler with PostgreSQL + Kafka and authentication at live demo https://kubetrace.vercel.app.",
                ),
            ],
            awards=[
                Award(
                    title="Google Summer of Code (GSoC)",
                    awarder="Google",
                    date="2024-08",
                    summary="12 merged PRs to upstream OSS project (1,800 stars).",
                ),
                Award(
                    title="Girl Script Summer of Code",
                    awarder="GirlScript",
                    date="2023-07",
                    summary="Merged 5 PRs to community repositories.",
                ),
                Award(
                    title="HackerRank Orchestrate",
                    awarder="HackerRank",
                    date="2025-09",
                    summary="Achieved rank #18 in HackerRank Orchestrate finals.",
                ),
            ],
        ),
        github_data={
            "profile": {"username": "enovak", "public_repos": 22, "followers": 210},
            "projects": [
                {
                    "name": "tracemesh-oss",
                    "description": "Upstream distributed telemetry collector",
                    "github_url": "https://github.com/tracemesh/tracemesh-oss",
                    "project_type": "open_source",
                    "github_details": {"stars": 1800, "forks": 240, "language": "Rust"},
                }
            ],
        },
        blog_data={
            "total_blogs": 3,
            "blog_score": 9.0,
            "blog_details": "Systems latency and LLM guardrail architecture",
            "blogs": [{"url": "https://elenanovak.dev/posts/guardrails", "score": 9.0, "details": "Guardrails"}],
        },
    )

    return [strong_oss, strong_builder, mid_borderline, tutorial_low, bonus_cap_edge]


def _render_resume_text(
    resume: JSONResume,
    github_data: Optional[Dict[str, Any]] = None,
    blog_data: Optional[Dict[str, Any]] = None,
) -> str:
    text = convert_json_resume_to_text(resume)
    if github_data:
        text += convert_github_data_to_text(github_data)
    if blog_data:
        text += convert_blog_data_to_text(blog_data)
    return text


def evaluate_fixture_offline(
    fixture: SyntheticCandidateFixture,
    role: Role,
    provider: Optional[LLMProvider] = None,
    thresholds: Optional[TierThresholds] = None,
) -> Tuple[Any, ScoreBreakdown, WeightedNormalizationResult, EvidenceAuditResult]:
    """Run ``ResumeEvaluator`` -> ``compute_evaluation_score`` -> ``audit_evidence_spans`` offline."""
    eval_model = build_evaluation_model(role)
    active_provider = provider or DeterministicRubricProvider(role)
    evaluator = ResumeEvaluator(
        role=role,
        evaluation_model=eval_model,
        provider=active_provider,
    )
    resume_text = _render_resume_text(
        fixture.resume, fixture.github_data, fixture.blog_data
    )
    evaluation = evaluator.evaluate_resume(resume_text)
    breakdown = compute_evaluation_score(evaluation, role)
    normalized = normalize_weighted_criteria(breakdown, role, thresholds=thresholds)
    evidence_audit = audit_evidence_spans(
        evaluation, resume_text, role, github_data=fixture.github_data
    )
    return evaluation, breakdown, normalized, evidence_audit


def _kendall_tau(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    """Compute Kendall's tau-b correlation between two score lists without external dependencies."""
    n = len(ranks_a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = ranks_a[i] - ranks_a[j]
            db = ranks_b[i] - ranks_b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denom if denom else 1.0


def _build_counterfactual_perturbations(
    cohort: Sequence[SyntheticCandidateFixture],
) -> List[
    Tuple[
        SyntheticCandidateFixture,
        str,
        str,
        JSONResume,
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        bool,
        Optional[str],
    ]
]:
    """Generate counterfactual invariant and monotonic sensitivity variants."""
    perturbations = []
    target = cohort[2]  # mid_borderline: clean headroom to move both up and down

    # 1. Demographic name/pronoun/contact swaps across cohort
    demographic_swaps = [
        ("Priya Nair (she/her)", "priya.nair@example.org", "+1-555-0102"),
        ("Carlos Mendez (he/him)", "carlos.mendez@example.net", "+34-600-000-111"),
        ("Jordan Taylor (they/them)", "jordan.taylor@example.io", "+44-7700-900077"),
    ]
    for idx, cand in enumerate(cohort[:3]):
        new_name, new_email, new_phone = demographic_swaps[idx]
        swapped = deepcopy(cand.resume)
        swapped.basics.name = new_name
        swapped.basics.email = new_email
        swapped.basics.phone = new_phone
        perturbations.append(
            (
                cand,
                f"demographic_swap_{idx+1}",
                "demographic_swap",
                swapped,
                cand.github_data,
                cand.blog_data,
                True,
                None,
            )
        )

    # 2. Institution, GPA & location swap (must be invariant!)
    for idx, cand in enumerate(cohort[:2]):
        elite = deepcopy(cand.resume)
        elite.basics.location = Location(
            city="Palo Alto", region="CA", countryCode="US"
        )
        if elite.education:
            elite.education[0].institution = "Stanford University"
            elite.education[0].score = "4.0/4.0"
        perturbations.append(
            (
                cand,
                f"institution_gpa_location_swap_{idx+1}",
                "institution_grade_swap",
                elite,
                cand.github_data,
                cand.blog_data,
                True,
                None,
            )
        )

    # 3. Formatting noise: extra whitespace + bullet order permutation within work/projects
    noisy = deepcopy(target.resume)
    if noisy.work and noisy.work[0].summary:
        noisy.work[0].summary = f"  {noisy.work[0].summary}   "
    if noisy.projects and len(noisy.projects) > 1:
        noisy.projects = list(reversed(noisy.projects))
    if noisy.skills:
        for sk in noisy.skills:
            if sk.keywords:
                sk.keywords = list(reversed(sk.keywords))
    perturbations.append(
        (
            target,
            "formatting_whitespace_and_order_noise",
            "formatting_noise",
            noisy,
            target.github_data,
            target.blog_data,
            True,
            None,
        )
    )

    # 4. Monotonic positive perturbation: add quantified production metric + AI evals + Orchestrate rank
    upgraded = deepcopy(target.resume)
    upgraded.work[0].highlights.append(
        "Scaled caching cluster to 35k RPS with 99.95% uptime across 2,000 active users."
    )
    upgraded.projects[1].description = (
        "RAG search engine with embeddings, structured outputs, guardrails, "
        "evaluation harness achieving 0.89 precision@5, live demo https://docsearch.vercel.app."
    )
    upgraded.awards = [
        Award(
            title="HackerRank Orchestrate",
            awarder="HackerRank",
            date="2025-10",
            summary="Placed rank #55 in HackerRank Orchestrate.",
        )
    ]
    perturbations.append(
        (
            target,
            "add_quantified_achievements_and_evals",
            "achievement_added",
            upgraded,
            target.github_data,
            target.blog_data,
            False,
            "increase",
        )
    )

    # 5. Monotonic negative perturbation: degrade projects to URL-less Todo tutorial, drop metrics
    degraded = deepcopy(target.resume)
    degraded.work = []
    degraded.projects = [
        Project(
            name="Todo List App",
            description="Basic CRUD todo list and calculator tutorial without links.",
        )
    ]
    perturbations.append(
        (
            target,
            "remove_metrics_and_degrade_to_tutorial",
            "achievement_removed",
            degraded,
            None,
            None,
            False,
            "decrease",
        )
    )

    return perturbations


def audit_counterfactual_fairness(
    cohort: Sequence[SyntheticCandidateFixture],
    role: Role,
    provider_factory: Optional[Callable[[Role], LLMProvider]] = None,
    thresholds: Optional[TierThresholds] = None,
) -> Tuple[List[CounterfactualAuditResult], float, float, float]:
    """Run counterfactual invariance & monotonicity tests across synthetic fixtures."""
    make_provider = provider_factory or (lambda r: DeterministicRubricProvider(r))
    results: List[CounterfactualAuditResult] = []
    baseline_cache: Dict[str, Tuple[float, str]] = {}
    invariant_baseline_scores: List[float] = []
    invariant_perturbed_scores: List[float] = []
    invariant_deltas: List[float] = []
    tier_flips = 0
    invariant_count = 0

    for fixture in cohort:
        _, bd, norm, _ = evaluate_fixture_offline(
            fixture, role, provider=make_provider(role), thresholds=thresholds
        )
        baseline_cache[fixture.candidate_id] = (bd.total_score, norm.tier)

    for (
        base_fixture,
        pert_name,
        pert_kind,
        pert_resume,
        pert_gh,
        pert_blog,
        expected_invariant,
        expected_direction,
    ) in _build_counterfactual_perturbations(cohort):
        base_score, base_tier = baseline_cache[base_fixture.candidate_id]
        pert_fixture = SyntheticCandidateFixture(
            candidate_id=f"{base_fixture.candidate_id}__{pert_name}",
            expected_tier=base_fixture.expected_tier,
            expected_rank_group=base_fixture.expected_rank_group,
            resume=pert_resume,
            github_data=pert_gh,
            blog_data=pert_blog,
        )
        _, pert_bd, pert_norm, _ = evaluate_fixture_offline(
            pert_fixture, role, provider=make_provider(role), thresholds=thresholds
        )
        delta = round(pert_bd.total_score - base_score, 4)

        if expected_invariant:
            invariant_count += 1
            invariant_baseline_scores.append(base_score)
            invariant_perturbed_scores.append(pert_bd.total_score)
            invariant_deltas.append(abs(delta))
            if pert_norm.tier != base_tier:
                tier_flips += 1
            passed = abs(delta) < 1e-6 and pert_norm.tier == base_tier
            details = (
                f"Invariant preserved (delta={delta:+.2f})"
                if passed
                else f"FAIRNESS DRIFT: delta={delta:+.2f}, tier {base_tier} -> {pert_norm.tier}"
            )
        elif expected_direction == "increase":
            passed = delta > 0.0
            details = (
                f"Score increased monotonically ({base_score:.1f} -> {pert_bd.total_score:.1f})"
                if passed
                else f"Expected increase, got delta={delta:+.2f}"
            )
        else:
            passed = delta < 0.0
            details = (
                f"Score decreased monotonically ({base_score:.1f} -> {pert_bd.total_score:.1f})"
                if passed
                else f"Expected decrease, got delta={delta:+.2f}"
            )

        results.append(
            CounterfactualAuditResult(
                baseline_id=base_fixture.candidate_id,
                perturbation_name=pert_name,
                perturbation_kind=pert_kind,
                baseline_score=base_score,
                perturbed_score=pert_bd.total_score,
                score_delta=delta,
                baseline_tier=base_tier,
                perturbed_tier=pert_norm.tier,
                expected_invariant=expected_invariant,
                expected_direction=expected_direction,
                passed=passed,
                details=details,
            )
        )

    max_delta = max(invariant_deltas) if invariant_deltas else 0.0
    tau = _kendall_tau(invariant_baseline_scores, invariant_perturbed_scores)
    flip_rate = (tier_flips / invariant_count) if invariant_count else 0.0
    return results, max_delta, round(tau, 4), round(flip_rate, 4)


def run_resilience_checks(role: Role) -> List[ResilienceCheckResult]:
    """Validate malformed JSON extraction, missing fields, tier validation, and score clamping."""
    checks: List[ResilienceCheckResult] = []
    eval_model = build_evaluation_model(role)
    cohort = build_synthetic_cohort()
    sample_fixture = cohort[0]
    raw_valid = DeterministicRubricProvider(role)._score_sanitized_resume(
        _strip_protected_fields_for_scoring(
            _render_resume_text(
                sample_fixture.resume,
                sample_fixture.github_data,
                sample_fixture.blog_data,
            )
        )
    )

    # 1. Think tags + whitespace + prose + markdown code fence
    wrapped = (
        "<think>Comparing projects and OSS stars...</think>\n\n"
        "Here is the JSON evaluation:\n```JSON\n"
        + json.dumps(raw_valid)
        + "\n```\nLet me know if you need more details."
    )
    try:
        parsed = parse_json_response(wrapped)
        validated = eval_model(**parsed)
        checks.append(
            ResilienceCheckResult(
                name="think_tag_and_markdown_extraction",
                passed=validated.deductions.total >= 0,
                details="Extracted and validated JSON wrapped in <think>, prose, and ```JSON fence",
            )
        )
    except Exception as exc:
        checks.append(
            ResilienceCheckResult(
                name="think_tag_and_markdown_extraction",
                passed=False,
                details=f"Failed: {exc}",
            )
        )

    # 2. Truncated / malformed JSON raises cleanly
    try:
        parse_json_response('{"scores": {"open_source": ')
        checks.append(
            ResilienceCheckResult(
                name="malformed_json_rejection",
                passed=False,
                details="Truncated JSON unexpectedly parsed",
            )
        )
    except (json.JSONDecodeError, ValueError):
        checks.append(
            ResilienceCheckResult(
                name="malformed_json_rejection",
                passed=True,
                details="Truncated JSON cleanly raised JSONDecodeError",
            )
        )

    # 3. Missing required category or whitespace-only evidence rejected
    missing_cat = deepcopy(raw_valid)
    first_key = role.categories[0].key
    missing_cat["scores"][first_key]["evidence"] = "   "
    try:
        eval_model(**missing_cat)
        checks.append(
            ResilienceCheckResult(
                name="blank_evidence_rejection",
                passed=False,
                details="Whitespace-only evidence unexpectedly passed validation",
            )
        )
    except ValidationError:
        checks.append(
            ResilienceCheckResult(
                name="blank_evidence_rejection",
                passed=True,
                details="Whitespace-only evidence cleanly rejected by ValidationError",
            )
        )

    # 4. Authoritative role max clamping when LLM overshoots both score & reported max
    overshoot = deepcopy(raw_valid)
    overshoot["scores"][first_key]["score"] = 99
    overshoot["scores"][first_key]["max"] = 99
    overshoot_eval = eval_model(**overshoot)
    overshoot_bd = compute_evaluation_score(overshoot_eval, role)
    expected_cat_max = role.categories[0].max
    capped_cat = overshoot_bd.categories[0]
    clamped_ok = (
        capped_cat.capped_score == float(expected_cat_max)
        and capped_cat.was_capped
        and capped_cat.max_mismatch
        and overshoot_bd.max_base_score == sum(c.max for c in role.categories)
    )
    checks.append(
        ResilienceCheckResult(
            name="authoritative_category_max_clamping",
            passed=clamped_ok,
            details=(
                f"LLM score=99/max=99 clamped to authoritative {capped_cat.capped_score}/{capped_cat.role_max}"
            ),
        )
    )

    # 5. Lower-bound clamping on extreme deductions
    extreme_ded = deepcopy(raw_valid)
    extreme_ded["deductions"]["total"] = 500
    extreme_eval = eval_model(**extreme_ded)
    extreme_bd = compute_evaluation_score(extreme_eval, role)
    low_clamp_ok = (
        extreme_bd.total_score == float(role.min_final_score)
        and extreme_bd.was_clamped_low
    )
    checks.append(
        ResilienceCheckResult(
            name="min_final_score_clamping",
            passed=low_clamp_ok,
            details=(
                f"Unclamped {extreme_bd.unclamped_total:.1f} clamped to min_final_score={extreme_bd.total_score:.1f}"
            ),
        )
    )

    # 6. Bonus cap & Orchestrate tier mismatch rejection (when bonus_rules present)
    if any(r.get("key") == "orchestrate" for r in role.bonus_rules):
        bad_tier = deepcopy(raw_valid)
        bad_tier["bonus_points"]["items"]["orchestrate"] = {
            "rank": 150,
            "points": 4,
            "evidence": "HackerRank Orchestrate rank 150",
        }
        try:
            eval_model(**bad_tier)
            checks.append(
                ResilienceCheckResult(
                    name="orchestrate_tier_validation",
                    passed=False,
                    details="Rank 150 with 4 points unexpectedly passed validation",
                )
            )
        except ValidationError:
            checks.append(
                ResilienceCheckResult(
                    name="orchestrate_tier_validation",
                    passed=True,
                    details="Mismatched Orchestrate rank 150 / 4 pts rejected by ValidationError",
                )
            )

    # 7. Partial/None fields in transform_evaluation_response
    csv_row = transform_evaluation_response(
        file_name="missing_sections.pdf",
        resume_data=JSONResume(basics=None, work=None, education=None),
        github_data=None,
        evaluation=None,
        role=role,
    )
    csv_ok = (
        csv_row["file_name"] == "missing_sections.pdf"
        and csv_row["total_score"] == "N/A"
        and csv_row["bonus_points"] == 0
        and csv_row["deductions"] == 0
    )
    checks.append(
        ResilienceCheckResult(
            name="csv_transform_none_resilience",
            passed=csv_ok,
            details="transform_evaluation_response handled None resume/evaluation safely",
        )
    )

    return checks


def run_full_audit(
    role_name: str = "software_engineering_intern",
    thresholds: Optional[TierThresholds] = None,
) -> FullAuditReport:
    """Execute the complete offline calibration, evidence, fairness, and resilience audit."""
    role = load_role(role_name)
    rubric_integrity = audit_rubric_integrity(role)
    cohort = build_synthetic_cohort()

    calibration_results: List[CandidateCalibrationResult] = []
    provider = DeterministicRubricProvider(role)
    for fixture in cohort:
        _, bd, norm, ev_audit = evaluate_fixture_offline(
            fixture, role, provider=provider, thresholds=thresholds
        )
        calibration_results.append(
            CandidateCalibrationResult(
                candidate_id=fixture.candidate_id,
                expected_tier=fixture.expected_tier,
                assigned_tier=norm.tier,
                base_score=bd.base_score,
                bonus_score=bd.bonus_score,
                deduction_score=bd.deduction_score,
                final_score=bd.total_score,
                normalized_base_score=norm.normalized_base_score,
                evidence_grounding_ratio=ev_audit.mean_grounding_ratio,
                leakage_count=len(ev_audit.leakage_flags),
                rubric_violations=ev_audit.rubric_violations,
            )
        )

    # Verify rank ordering across rank groups (group 1 > group 2 > group 3 > group 4)
    group_scores: Dict[int, List[float]] = {}
    for fixture, res in zip(cohort, calibration_results):
        group_scores.setdefault(fixture.expected_rank_group, []).append(res.final_score)
    sorted_groups = sorted(group_scores)
    rank_ordering_preserved = all(
        min(group_scores[sorted_groups[i]]) > max(group_scores[sorted_groups[i + 1]])
        for i in range(len(sorted_groups) - 1)
    )

    tier_matches = sum(
        1 for res in calibration_results if res.assigned_tier == res.expected_tier
    )
    tier_accuracy = tier_matches / len(calibration_results) if calibration_results else 0.0
    mean_grounding = (
        sum(r.evidence_grounding_ratio for r in calibration_results)
        / len(calibration_results)
        if calibration_results
        else 0.0
    )

    cf_results, max_delta, tau, flip_rate = audit_counterfactual_fairness(
        cohort, role, thresholds=thresholds
    )

    # Negative control: intentionally biased provider MUST be caught!
    def _biased_factory(r: Role) -> DeterministicRubricProvider:
        def _inject_bias(raw_resume: str, payload: Dict[str, Any]) -> None:
            if "Stanford University" in raw_resume:
                first_key = r.categories[0].key
                current = float(payload["scores"][first_key]["score"])
                cat_max = float(r.categories[0].max)
                payload["scores"][first_key]["score"] = min(cat_max, current + 8.0)
                payload["scores"][first_key]["evidence"] += " Attended Stanford University."

        return DeterministicRubricProvider(r, bias_hook=_inject_bias)

    biased_cf, biased_max_delta, _, _ = audit_counterfactual_fairness(
        cohort, role, provider_factory=_biased_factory, thresholds=thresholds
    )
    biased_control_caught = biased_max_delta > 0.0 and any(
        not item.passed for item in biased_cf if item.expected_invariant
    )

    resilience_results = run_resilience_checks(role)
    res_passed = sum(1 for c in resilience_results if c.passed)
    res_total = len(resilience_results)

    all_passed = (
        rubric_integrity.passed
        and rank_ordering_preserved
        and tier_accuracy == 1.0
        and all(r.leakage_count == 0 and not r.rubric_violations for r in calibration_results)
        and mean_grounding >= 0.70
        and all(c.passed for c in cf_results)
        and max_delta == 0.0
        and flip_rate == 0.0
        and biased_control_caught
        and res_passed == res_total
    )

    return FullAuditReport(
        role_name=role.name,
        rubric_integrity=rubric_integrity,
        calibration_results=calibration_results,
        rank_ordering_preserved=rank_ordering_preserved,
        tier_accuracy=round(tier_accuracy, 4),
        mean_grounding_ratio=round(mean_grounding, 4),
        counterfactual_results=cf_results,
        max_invariance_delta=max_delta,
        rank_kendall_tau=tau,
        tier_flip_rate=flip_rate,
        biased_control_caught=biased_control_caught,
        resilience_results=resilience_results,
        resilience_checks_passed=res_passed,
        resilience_checks_total=res_total,
        passed=all_passed,
    )


def format_audit_report(report: FullAuditReport) -> str:
    """Render a concise human-readable terminal summary of ``FullAuditReport``."""
    lines: List[str] = []
    lines.append("=" * 84)
    lines.append(
        f"OFFLINE RUBRIC CALIBRATION, EVIDENCE & FAIRNESS AUDIT  [role: {report.role_name}]"
    )
    lines.append("=" * 84)

    ri = report.rubric_integrity
    lines.append(
        f"\n[1] RUBRIC INTEGRITY: {'PASS' if ri.passed else 'FAIL'} "
        f"(base_max={ri.base_max}, bonus_max={ri.bonus_max}, bounds=[{ri.min_final_score}, {ri.max_final_score}], "
        f"prompts_ok={ri.prompts_render_ok}, fairness_clauses={ri.fairness_clauses_present})"
    )
    for warn in ri.warnings:
        lines.append(f"    ! {warn}")

    lines.append(
        f"\n[2] CALIBRATION COHORT ({len(report.calibration_results)} synthetic candidates): "
        f"tier_accuracy={report.tier_accuracy * 100:.1f}%, "
        f"rank_ordering={'PASS' if report.rank_ordering_preserved else 'FAIL'}, "
        f"mean_evidence_grounding={report.mean_grounding_ratio * 100:.1f}%"
    )
    lines.append(
        f"    {'Candidate':<28} {'Base':>6} {'Bonus':>6} {'Ded':>5} {'Final':>7} {'Norm%':>6} {'Ground%':>8} {'Tier'}"
    )
    lines.append("    " + "-" * 78)
    for r in report.calibration_results:
        lines.append(
            f"    {r.candidate_id:<28} {r.base_score:>6.1f} {r.bonus_score:>6.1f} "
            f"{r.deduction_score:>5.1f} {r.final_score:>7.1f} {r.normalized_base_score:>6.1f} "
            f"{r.evidence_grounding_ratio * 100:>7.1f}% {r.assigned_tier}"
        )

    lines.append(
        f"\n[3] COUNTERFACTUAL FAIRNESS & MONOTONICITY ({len(report.counterfactual_results)} perturbations):"
    )
    lines.append(
        f"    max_invariance_delta={report.max_invariance_delta:.2f} | "
        f"kendall_tau={report.rank_kendall_tau:.3f} | "
        f"tier_flip_rate={report.tier_flip_rate * 100:.1f}% | "
        f"biased_control_caught={report.biased_control_caught}"
    )
    for cf in report.counterfactual_results:
        status = "PASS" if cf.passed else "FAIL"
        lines.append(
            f"    [{status}] {cf.baseline_id} :: {cf.perturbation_name} -> "
            f"{cf.baseline_score:.1f} -> {cf.perturbed_score:.1f} (delta={cf.score_delta:+.1f}) | {cf.details}"
        )

    lines.append(
        f"\n[4] SCHEMA & MALFORMED INPUT RESILIENCE: "
        f"{report.resilience_checks_passed}/{report.resilience_checks_total} passed"
    )
    for rc in report.resilience_results:
        status = "PASS" if rc.passed else "FAIL"
        lines.append(f"    [{status}] {rc.name}: {rc.details}")

    lines.append("\n" + "=" * 84)
    lines.append(f"OVERALL AUDIT STATUS: {'PASS' if report.passed else 'FAIL'}")
    lines.append("=" * 84)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    available = list_available_roles()
    parser = argparse.ArgumentParser(
        description="Run offline rubric calibration, evidence span, and fairness perturbation audit."
    )
    parser.add_argument(
        "--role",
        default="software_engineering_intern",
        help="Role under roles/ to audit (available: "
        + (", ".join(available) if available else "none")
        + ")",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON audit report instead of formatted text.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_full_audit(role_name=args.role)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(format_audit_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

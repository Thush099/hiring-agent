"""Role definitions for the resume scorer.

A role lives in ``roles/<name>/`` and bundles three parts:

- ``role.json``          — categories, weights and score bounds (machine-readable)
- ``criteria.jinja``     — the evaluation criteria prompt (rendered with ``text_content``)
- ``system_message.jinja`` — the system prompt

The categories/weights in ``role.json`` drive the dynamic scoring schema, the
printout, the CSV columns and the score-cap math, so every role can score against
its own rubric.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json
import re

ROLES_DIR = Path(__file__).parent / "roles"
_VALID_ROLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class Category:
    """One scoring category and its maximum weight."""

    key: str
    label: str
    max: int
    icon: str = "•"


@dataclass(frozen=True)
class Role:
    """A fully-loaded role definition."""

    name: str
    position_title: str
    categories: List[Category]
    bonus_max: int
    min_final_score: int
    max_final_score: int
    criteria_source: str
    system_message_source: str
    bonus_rules: list = field(default_factory=list)


def _validate_role_name(name: str) -> str:
    """Validate that a role name is a safe single directory identifier."""
    if not isinstance(name, str) or not _VALID_ROLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid role name '{name}'. Use alphanumeric characters, '_' or '-' only."
        )
    return name


def list_available_roles(roles_dir: Optional[Path] = None) -> List[str]:
    """Return the names of every role directory under ``roles/``."""
    base_dir = roles_dir or ROLES_DIR
    if not base_dir.is_dir():
        return []
    return sorted(p.name for p in base_dir.iterdir() if (p / "role.json").is_file())


# Scaffold contents for a freshly-created role. Placeholder categories keep the
# new role immediately runnable; edit them to fit the real rubric.
_SCAFFOLD_MANIFEST = {
    "position_title": "TODO: describe the position (e.g. 'Backend Engineer at HackerRank')",
    "categories": [
        {"key": "category_one", "label": "Category One", "max": 50, "icon": "⭐"},
        {"key": "category_two", "label": "Category Two", "max": 50, "icon": "⭐"},
    ],
    "bonus_max": 10,
    "min_final_score": 0,
    "max_final_score": 110,
}

_SCAFFOLD_SYSTEM_MESSAGE = """\
You are an expert technical recruiter evaluating resumes for the {position_title}.
Provide accurate, objective evaluations based on the given criteria.

**CRITICAL: You are NOT writing a resume summary. You are SCORING a resume for a job application.**

**CRITICAL FAIRNESS REQUIREMENTS:** Scores must never depend on the candidate's
name, gender, demographic information, institution name, grades, or location.
Evaluate only on demonstrated skills and experience relevant to this role.

**MANDATORY: You MUST always fill ALL categories: {category_keys}**, and provide
evidence for each.

You MUST respond with valid JSON matching the exact structure specified in the
prompt. Do not add or omit fields.
"""

_SCAFFOLD_CRITERIA = """\
You are evaluating a resume for the {position_title}. Analyze the resume data and
provide scores based on these criteria:

**MANDATORY: You MUST always fill ALL categories: {category_keys}.**

## SCORING CRITERIA

{criteria_sections}

## BONUS POINTS (Maximum total: {bonus_max} points)
- TODO: define what earns bonus points for this role

## CRITICAL REQUIREMENTS
1. You MUST respond with ONLY the JSON structure below
2. You MUST fill ALL score categories
3. You MUST provide evidence for each score

Analyze the following resume and provide a JSON response with this EXACT structure:

{{
    "scores": {{
{scores_json}
    }},
    "bonus_points": {{"total": 0, "breakdown": "string"}},
    "deductions": {{"total": 0, "reasons": "string"}},
    "key_strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["improvement1"]
}}

Resume to evaluate:

{{{{ text_content }}}}
"""


def scaffold_role(name: str, roles_dir: Optional[Path] = None) -> Path:
    """Create ``roles/<name>/`` with a basic, runnable role definition.

    Writes ``role.json``, ``criteria.jinja`` and ``system_message.jinja`` with
    placeholder categories. Returns the new role directory. Raises ValueError if
    a role with that name already exists or the name is invalid.
    """
    _validate_role_name(name)
    base_dir = roles_dir or ROLES_DIR
    role_dir = base_dir / name
    if role_dir.exists():
        raise ValueError(f"Role '{name}' already exists at {role_dir}")

    categories = _SCAFFOLD_MANIFEST["categories"]
    category_keys = ", ".join(c["key"] for c in categories)

    criteria_sections = "\n\n".join(
        f"### {c['label']} (0-{c['max']} points)\n- TODO: describe what earns points in this category"
        for c in categories
    )
    scores_json = ",\n".join(
        f'        "{c["key"]}": {{"score": 0, "max": {c["max"]}, "evidence": "string"}}'
        for c in categories
    )

    role_dir.mkdir(parents=True)
    manifest = {**_SCAFFOLD_MANIFEST}
    (role_dir / "role.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (role_dir / "system_message.jinja").write_text(
        _SCAFFOLD_SYSTEM_MESSAGE.format(
            position_title=manifest["position_title"],
            category_keys=category_keys,
        ),
        encoding="utf-8",
    )
    (role_dir / "criteria.jinja").write_text(
        _SCAFFOLD_CRITERIA.format(
            position_title=manifest["position_title"],
            category_keys=category_keys,
            bonus_max=manifest["bonus_max"],
            criteria_sections=criteria_sections,
            scores_json=scores_json,
        ),
        encoding="utf-8",
    )
    return role_dir


def _validate_bonus_rules(name: str, bonus_rules: list) -> None:
    """Validate configured bonus rules for key uniqueness and tier consistency."""
    if not isinstance(bonus_rules, list):
        raise ValueError(f"Role '{name}': 'bonus_rules' must be a list")

    seen_keys = set()
    for rule in bonus_rules:
        if not isinstance(rule, dict):
            raise ValueError(f"Role '{name}': bonus rule must be an object (got {rule!r})")
        key = rule.get("key")
        label = rule.get("label")
        if not key or not label:
            raise ValueError(
                f"Role '{name}': each bonus rule requires 'key' and 'label' (got {rule!r})"
            )
        if key in seen_keys:
            raise ValueError(f"Role '{name}': duplicate bonus rule key '{key}'")
        seen_keys.add(key)

        if "tiers" in rule:
            tiers = rule["tiers"]
            if not isinstance(tiers, list) or not tiers:
                raise ValueError(
                    f"Role '{name}': bonus rule '{key}' must define a non-empty 'tiers' list"
                )
            for tier in tiers:
                min_rank = int(tier["min_rank"])
                max_rank = int(tier["max_rank"])
                points = int(tier["points"])
                if min_rank < 1 or max_rank < min_rank or points < 0:
                    raise ValueError(
                        f"Role '{name}': invalid tier {tier!r} in bonus rule '{key}'"
                    )
        else:
            points = rule.get("points")
            if not isinstance(points, list) or not points or any(int(p) < 0 for p in points):
                raise ValueError(
                    f"Role '{name}': bonus rule '{key}' must define non-negative 'points'"
                )


def load_role(name: str, roles_dir: Optional[Path] = None) -> Role:
    """Load and validate the role named ``name``.

    Raises ValueError with the list of available roles if the role directory,
    its manifest, or its prompt templates are missing or malformed.
    """
    _validate_role_name(name)
    base_dir = roles_dir or ROLES_DIR
    role_dir = base_dir / name
    manifest_path = role_dir / "role.json"
    criteria_path = role_dir / "criteria.jinja"
    system_message_path = role_dir / "system_message.jinja"

    if not manifest_path.is_file():
        available = ", ".join(list_available_roles(base_dir)) or "(none found)"
        raise ValueError(
            f"Unknown role '{name}'. Expected {manifest_path} to exist. "
            f"Available roles: {available}"
        )

    for path in (criteria_path, system_message_path):
        if not path.is_file():
            raise ValueError(f"Role '{name}' is missing required file: {path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Role '{name}' has an invalid role.json: {e}") from e

    raw_categories = manifest.get("categories")
    if not raw_categories or not isinstance(raw_categories, list):
        raise ValueError(
            f"Role '{name}' role.json must define a non-empty 'categories' list"
        )

    categories: List[Category] = []
    seen_keys = set()
    for raw in raw_categories:
        if not isinstance(raw, dict):
            raise ValueError(f"Role '{name}': invalid category entry {raw!r}")
        key = raw.get("key")
        max_score = raw.get("max")
        if not key or max_score is None:
            raise ValueError(
                f"Role '{name}': each category needs a 'key' and 'max' (got {raw!r})"
            )
        max_int = int(max_score)
        if max_int <= 0:
            raise ValueError(
                f"Role '{name}': category '{key}' max must be > 0 (got {max_int})"
            )
        if key in seen_keys:
            raise ValueError(f"Role '{name}': duplicate category key '{key}'")
        seen_keys.add(key)
        categories.append(
            Category(
                key=key,
                label=raw.get("label", key),
                max=max_int,
                icon=raw.get("icon", "•"),
            )
        )

    bonus_max = int(manifest.get("bonus_max", 20))
    if bonus_max < 0:
        raise ValueError(f"Role '{name}': bonus_max must be >= 0 (got {bonus_max})")

    min_final_score = int(manifest.get("min_final_score", 0))
    max_final_score = int(
        manifest.get(
            "max_final_score",
            sum(c.max for c in categories) + bonus_max,
        )
    )
    if min_final_score > max_final_score:
        raise ValueError(
            f"Role '{name}': min_final_score ({min_final_score}) cannot exceed "
            f"max_final_score ({max_final_score})"
        )

    bonus_rules = manifest.get("bonus_rules") or []
    _validate_bonus_rules(name, bonus_rules)

    return Role(
        name=name,
        position_title=manifest.get("position_title", name),
        categories=categories,
        bonus_max=bonus_max,
        min_final_score=min_final_score,
        max_final_score=max_final_score,
        criteria_source=criteria_path.read_text(encoding="utf-8"),
        system_message_source=system_message_path.read_text(encoding="utf-8"),
        bonus_rules=bonus_rules,
    )

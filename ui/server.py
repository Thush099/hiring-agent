"""Lightweight local web UI server for the hiring-agent scoring workbench.

Serves the scorecard in ``ui/index.html`` and exposes a JSON API that runs the real
``compute_evaluation_score`` against a loaded role. This is a decision-support calculator
for reviewing a rubric breakdown — not an automated hiring decision. Standard library only.
Start with ``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roles import ROLES_DIR, load_role  # noqa: E402
from score import compute_evaluation_score  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")


def available_roles() -> list[str]:
    """Return the role names available under the roles directory."""
    if not ROLES_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in ROLES_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


class _CategoryScore:
    """Minimal score record matching the shape ``compute_evaluation_score`` reads."""

    def __init__(self, score: float, max_value: float, evidence: str) -> None:
        self.score = score
        self.max = max_value
        self.evidence = evidence


class _Totalled:
    """Minimal bonus/deduction record exposing a ``total``, read by the scorer."""

    def __init__(self, total: float) -> None:
        self.total = total


class _Evaluation:
    """Minimal evaluation object exposing the fields ``compute_evaluation_score`` reads."""

    def __init__(
        self,
        scores: dict[str, _CategoryScore],
        bonus_points: _Totalled,
        deductions: _Totalled,
    ) -> None:
        self.scores = scores
        self.bonus_points = bonus_points
        self.deductions = deductions


def run_score(
    role_name: str,
    scores: dict[str, Any],
    bonus: float = 0.0,
    deduction: float = 0.0,
) -> dict[str, Any]:
    """Compute the deterministic rubric score and return the breakdown.

    Raises ValueError with a clear message for invalid inputs.
    """
    if not isinstance(role_name, str) or not role_name.strip():
        raise ValueError("role_name must be a non-empty string")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty object mapping category -> score")
    for name, value in (("bonus", bonus), ("deduction", deduction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")

    try:
        role = load_role(role_name.strip())
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError(f"Unknown role {role_name!r}: {exc}") from exc

    known = {category.key for category in role.categories}
    unknown = sorted(set(scores) - known)
    if unknown:
        raise ValueError(f"Unknown score category(ies): {unknown}. Role has: {sorted(known)}")

    eval_scores: dict[str, _CategoryScore] = {}
    for category in role.categories:
        entry = scores.get(category.key)
        if entry is None:
            eval_scores[category.key] = _CategoryScore(0.0, category.max, "")
            continue
        if isinstance(entry, dict):
            raw = entry.get("score", 0.0)
            evidence = str(entry.get("evidence", ""))
            reported_max = entry.get("max", category.max)
        else:
            raw = entry
            evidence = ""
            reported_max = category.max
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Score for '{category.key}' must be a number, got {raw!r}")
        eval_scores[category.key] = _CategoryScore(float(raw), float(reported_max), evidence)

    breakdown = compute_evaluation_score(
        _Evaluation(eval_scores, _Totalled(float(bonus)), _Totalled(float(deduction))),
        role,
    )

    return {
        "role_name": role.name,
        "position_title": role.position_title,
        "categories": [
            {
                "key": c.key,
                "label": c.label,
                "icon": c.icon,
                "raw_score": round(c.raw_score, 4),
                "capped_score": round(c.capped_score, 4),
                "role_max": c.role_max,
                "reported_max": c.reported_max,
                "was_capped": c.was_capped,
                "max_mismatch": c.max_mismatch,
                "evidence": c.evidence,
            }
            for c in breakdown.categories
        ],
        "base_score": round(breakdown.base_score, 4),
        "max_base_score": breakdown.max_base_score,
        "bonus_score": round(breakdown.bonus_score, 4),
        "deduction_score": round(breakdown.deduction_score, 4),
        "unclamped_total": round(breakdown.unclamped_total, 4),
        "total_score": round(breakdown.total_score, 4),
        "min_final_score": breakdown.min_final_score,
        "max_final_score": breakdown.max_final_score,
        "was_clamped_high": breakdown.was_clamped_high,
        "was_clamped_low": breakdown.was_clamped_low,
        "weighted_base_pct": round(breakdown.weighted_base_pct, 4),
        "human_review_required": True,
    }


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "roles": available_roles()})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/score":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_score(
                    role_name=payload.get("role_name", ""),
                    scores=payload.get("scores", {}),
                    bonus=payload.get("bonus", 0.0),
                    deduction=payload.get("deduction", 0.0),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the rubric scoring workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8133)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Rubric scoring workbench running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
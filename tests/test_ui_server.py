"""Tests for the rubric scoring workbench UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_score

ROLE = "software_engineering_intern"
SCORES = {
    "open_source": {"score": 35, "evidence": "strong oss"},
    "self_projects": 20,
    "production": 15,
    "technical_skills": 8,
    "ai_fluency": 7,
}


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/score",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_score_endpoint_runs_real_aggregation(server):
    status, payload = _post(server, {"role_name": ROLE, "scores": SCORES, "bonus": 15})
    assert status == 200
    assert payload["base_score"] == 85.0
    assert payload["total_score"] == payload["base_score"] + payload["bonus_score"]
    assert len(payload["categories"]) == 5
    assert payload["human_review_required"] is True


def test_result_is_json_serializable():
    result = run_score(ROLE, SCORES)
    json.dumps(result)  # must not raise


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert ROLE in health["roles"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "workbench" in html and "does not make hiring decisions" in html


def test_final_clamp_respects_role_bounds():
    # A large deduction drives the total below the role floor; the clamp brings it back.
    zeroed = {key: {"score": 0} for key in SCORES}
    result = run_score(ROLE, zeroed, deduction=1000)
    assert result["was_clamped_low"] is True
    assert result["total_score"] == result["min_final_score"]


def test_bonus_is_clamped_to_role_bonus_max():
    over = {key: {"score": 1000} for key in SCORES}
    result = run_score(ROLE, over, bonus=500)
    assert result["bonus_score"] == 20.0
    assert result["total_score"] == result["max_final_score"]


@pytest.mark.parametrize(
    "args,message",
    [
        (("", {}), "role_name must be a non-empty string"),
        ((ROLE, {}), "scores must be a non-empty object"),
        ((ROLE, {"nope": 1}), "Unknown score category"),
        ((ROLE, {"open_source": "abc"}), "Score for 'open_source' must be a number"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, args, message):
    role_name, scores = args
    status, body = _post(server, {"role_name": role_name, "scores": scores})
    assert status == 400
    assert message in body["error"]
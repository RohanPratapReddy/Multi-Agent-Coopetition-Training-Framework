"""test evaluations (split from test_auth_and_evaluation.py)."""

from __future__ import annotations

from api_support import body, build_settings


def test_p1_evaluation_is_void_with_negative_final_score(client):
    response = client.post(
        "/evaluations",
        json={"peer": "alpha", "episode": 0, "events": [{"rule_id": "P1", "evidence": "tampered with the verifier"}]},
    )
    assert response.status_code == 200, response.text
    card = body(response)
    assert card["final_score"] == -5.0
    assert card["void"] is True
    assert card["explain"]


def test_evaluation_rejects_unknown_rule(client):
    response = client.post(
        "/evaluations", json={"peer": "alpha", "events": [{"rule_id": "ZZ99", "evidence": "x"}]}
    )
    assert response.status_code in (404, 422)


def test_evaluation_validation_errors_are_problem_json(client):
    response = client.post("/evaluations", json={"events": []})
    assert response.status_code == 422
    problem = body(response)
    assert problem["status"] == 422
    assert "peer" in str(problem["extensions"])

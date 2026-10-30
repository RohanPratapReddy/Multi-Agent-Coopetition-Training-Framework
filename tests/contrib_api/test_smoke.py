from __future__ import annotations


def test_every_parameterless_get_route_answers(client):
    for path in (
        "/health", "/meta", "/rules", "/rules/validation", "/rules/P1", "/severities", "/tiers",
        "/scores/scale", "/openapi.json", "/runs", "/peers", "/sandbox/providers", "/sandbox/status",
        "/scorers", "/trainers/capabilities", "/trainers/presets", "/admin/capabilities",
    ):
        response = client.get(path)
        assert response.status_code == 200, (path, response.status_code, response.text[:400])

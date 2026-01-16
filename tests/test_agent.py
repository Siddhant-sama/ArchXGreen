import httpx
import pytest


def _require_fields(data: dict, fields: list[str]):
    missing = [field for field in fields if field not in data]
    assert not missing, f"Missing fields: {missing}"


def test_agent_card(agent):
    """Verify agent card is exposed via .well-known."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json", timeout=10)
    assert response.status_code == 200
    card = response.json()

    _require_fields(card, ["name", "description", "url", "version", "skills", "capabilities"])
    assert card["skills"], "Agent must advertise at least one skill"


def test_card_alias(agent):
    """/card should mirror .well-known/agent-card.json for compatibility."""
    primary = httpx.get(f"{agent}/.well-known/agent-card.json", timeout=10).json()
    alias = httpx.get(f"{agent}/card", timeout=10).json()
    assert alias["name"] == primary["name"]
    assert alias["url"] == primary["url"]


def test_health(agent):
    """Health endpoint exposes task count."""
    response = httpx.get(f"{agent}/health", timeout=10)
    assert response.status_code == 200
    payload = response.json()
    _require_fields(payload, ["status", "total_tasks", "timestamp"])
    assert payload["status"] == "healthy"
    assert payload["total_tasks"] > 0


def test_tasks_listing(agent):
    """Tasks endpoint should return a non-empty list with metadata."""
    response = httpx.get(f"{agent}/tasks?limit=3", timeout=20)
    assert response.status_code == 200
    tasks = response.json()
    assert isinstance(tasks, list)
    assert tasks, "Expected at least one task"
    sample = tasks[0]
    _require_fields(sample, ["task_id", "level", "problem_name", "difficulty_score"])

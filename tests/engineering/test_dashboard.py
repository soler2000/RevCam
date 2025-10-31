from __future__ import annotations

from typing import Any


def _find_metric(metrics: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for metric in metrics:
        if metric.get("key") == key:
            return metric
    raise AssertionError(f"Metric {key!r} not found: {metrics!r}")


def test_engineering_dashboard_reports_zero_drafts_when_none_exist(client):
    client.post(
        "/engineering/work-centres/",
        json={"name": "Assembly"},
    )
    payload = client.get("/engineering/dashboard").json()
    assert payload["name"] == "Engineering Dashboard"
    metric = _find_metric(payload["metrics"], "draft_sops")
    assert metric["value"] == 0
    assert metric["label"] == "SOPs in Draft"
    assert metric["logo"] == "thick-bubble"


def test_engineering_dashboard_counts_sop_drafts(client):
    work_centre = client.post(
        "/engineering/work-centres/",
        json={"name": "Fabrication"},
    ).json()
    client.post(
        "/sop/procedures/",
        json={
            "title": "Safety procedures",
            "reference": "SOP-001",
            "work_centre": work_centre["id"],
            "instructions": "Follow safety steps",
            "is_active": True,
        },
    )
    client.post(
        "/sop/procedures/",
        json={
            "title": "Draft procedure",
            "reference": "SOP-002",
            "work_centre": work_centre["id"],
            "instructions": "Draft instructions",
            "is_active": False,
        },
    )
    payload = client.get("/engineering/dashboard").json()
    metric = _find_metric(payload["metrics"], "draft_sops")
    assert metric["value"] == 1

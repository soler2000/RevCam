from __future__ import annotations

def test_sop_uses_work_centre_reference(client):
    work_centre = client.post(
        "/engineering/work-centres/",
        json={"name": "Inspection"},
    ).json()

    response = client.post(
        "/sop/procedures/",
        json={
            "title": "Final inspection",
            "reference": "SOP-001",
            "work_centre": work_centre["id"],
            "instructions": "Check all fixtures",
            "revision": 2,
        },
    )
    assert response.status_code == 201, response.content
    payload = response.json()
    assert payload["work_centre"] == work_centre["id"]
    assert payload["work_centre_name"] == "Inspection"

    response = client.get(f"/sop/procedures/?work_centre={work_centre['id']}")
    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["title"] == "Final inspection"

    duplicate = client.post(
        "/sop/procedures/",
        json={
            "title": "Another",
            "reference": "SOP-001",
            "work_centre": work_centre["id"],
            "instructions": "N/A",
        },
    )
    assert duplicate.status_code == 400
    assert "reference" in duplicate.json()["detail"].lower()

    missing_wc = client.post(
        "/sop/procedures/",
        json={
            "title": "No WC",
            "reference": "SOP-002",
            "instructions": "N/A",
        },
    )
    assert missing_wc.status_code == 400
    assert "work_centre" in missing_wc.json()["detail"].lower()

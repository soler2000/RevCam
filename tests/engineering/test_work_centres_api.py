from __future__ import annotations

def test_work_centre_crud_flow(client):
    response = client.post(
        "/engineering/work-centres/",
        json={"name": "Assembly", "code": "WC-001", "description": "Assembly line"},
    )
    assert response.status_code == 201, response.content
    created = response.json()
    assert created["name"] == "Assembly"
    assert created["code"] == "WC-001"

    response = client.get("/engineering/work-centres/")
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["name"] == "Assembly"

    work_centre_id = created["id"]
    response = client.patch(
        f"/engineering/work-centres/{work_centre_id}/",
        json={"description": "Updated description"},
    )
    assert response.status_code == 200, response.content
    assert response.json()["description"] == "Updated description"

    response = client.post(
        "/engineering/work-centres/",
        json={"name": "Assembly"},
    )
    assert response.status_code == 400
    assert "name" in response.json()["detail"].lower()

    response = client.post(
        "/engineering/work-centres/",
        json={"name": "Painting", "code": "WC-001"},
    )
    assert response.status_code == 400
    assert "code" in response.json()["detail"].lower()

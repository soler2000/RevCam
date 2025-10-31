from __future__ import annotations

from revcam.frontend import ENGINEERING_NAVIGATION, SopForm, WorkCentreForm


def test_navigation_contains_work_centres_entry():
    labels = [item.label for item in ENGINEERING_NAVIGATION]
    assert "Work Centres" in labels


def test_navigation_includes_engineering_dashboard_first():
    labels = [item.label for item in ENGINEERING_NAVIGATION]
    assert labels[0] == "Engineering Dashboard"


def test_work_centre_form_integration(client):
    form = WorkCentreForm(client)

    success, message = form.create(name="Fabrication", code="FAB")
    assert success is True
    assert "Fabrication" in message

    duplicate, duplicate_message = form.create(name="Fabrication")
    assert duplicate is False
    assert "name" in duplicate_message.lower()

    centres = form.list()
    assert centres and centres[0]["name"] == "Fabrication"

    centre_id = centres[0]["id"]
    updated, update_message = form.update(centre_id, description="Metal fabrication")
    assert updated is True
    assert "updated" in update_message.lower()


def test_sop_form_uses_work_centres(client):
    work_form = WorkCentreForm(client)
    work_form.create(name="QA")
    centres = work_form.list()
    sop_form = SopForm(client)
    available = sop_form.load_work_centres()
    assert any(item["name"] == "QA" for item in available)

    ok, message = sop_form.submit(
        title="Safety checks",
        reference="SOP-QA",
        work_centre=centres[0]["id"],
        instructions="Follow checklist",
    )
    assert ok is True
    assert "SOP-QA" in message

    duplicate, error_message = sop_form.submit(
        title="Duplicate",
        reference="SOP-QA",
        work_centre=centres[0]["id"],
        instructions="Follow checklist",
    )
    assert duplicate is False
    assert "reference" in error_message.lower()

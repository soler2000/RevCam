from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsJsonClient(Protocol):
    def get(self, path: str): ...  # pragma: no cover - protocol signature
    def post(self, path: str, json: Any | None = ...): ...
    def patch(self, path: str, json: Any | None = ...): ...


@dataclass
class WorkCentreForm:
    client: SupportsJsonClient

    def list(self) -> list[dict[str, Any]]:
        response = self.client.get("/engineering/work-centres/")
        payload = response.json()
        return payload.get("results", [])

    def create(self, *, name: str, code: str | None = None, description: str | None = None) -> tuple[bool, str]:
        name = (name or "").strip()
        if not name:
            return False, "Name is required."
        payload: dict[str, Any] = {"name": name}
        if code:
            payload["code"] = code.strip()
        if description:
            payload["description"] = description.strip()
        response = self.client.post("/engineering/work-centres/", json=payload)
        if response.status_code == 201:
            data = response.json()
            return True, f"Work centre '{data['name']}' created successfully."
        data = response.json()
        return False, _format_error_messages(data)

    def update(self, work_centre_id: int, **fields: Any) -> tuple[bool, str]:
        if not fields:
            return False, "No changes to apply."
        response = self.client.patch(
            f"/engineering/work-centres/{work_centre_id}/",
            json=fields,
        )
        if response.status_code == 200:
            data = response.json()
            return True, f"Work centre '{data['name']}' updated successfully."
        return False, _format_error_messages(response.json())


def _format_error_messages(errors: dict[str, Any]) -> str:
    if not errors:
        return "Unable to save work centre."
    details: list[str] = []
    for field, messages in errors.items():
        joined = "; ".join(str(message) for message in (messages if isinstance(messages, list) else [messages]))
        details.append(f"{field}: {joined}")
    return "; ".join(details)

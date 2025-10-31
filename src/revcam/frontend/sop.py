from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .work_centres import SupportsJsonClient


@dataclass
class SopForm:
    client: SupportsJsonClient

    def load_work_centres(self) -> list[dict[str, Any]]:
        response = self.client.get("/engineering/work-centres/")
        data = response.json()
        return data.get("results", [])

    def submit(self, *, title: str, reference: str, work_centre: int, instructions: str, revision: int = 1) -> tuple[bool, str]:
        payload = {
            "title": title.strip(),
            "reference": reference.strip(),
            "work_centre": work_centre,
            "instructions": instructions.strip(),
            "revision": revision,
        }
        response = self.client.post("/sop/procedures/", json=payload)
        if response.status_code == 201:
            data = response.json()
            return True, f"SOP '{data['reference']}' saved for {data['work_centre_name']}."
        errors = response.json()
        return False, _format_errors(errors)


def _format_errors(errors: dict[str, Any]) -> str:
    if not errors:
        return "Unable to save SOP."
    parts: list[str] = []
    for field, messages in errors.items():
        if not isinstance(messages, list):
            messages = [messages]
        joined = "; ".join(str(message) for message in messages)
        parts.append(f"{field}: {joined}")
    return "; ".join(parts)

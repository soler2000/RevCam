from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class NavigationItem:
    label: str
    path: str


ENGINEERING_NAVIGATION: List[NavigationItem] = [
    NavigationItem(label="Engineering Dashboard", path="/engineering/dashboard"),
    NavigationItem(label="Processes", path="/engineering/processes"),
    NavigationItem(label="Work Centres", path="/engineering/work-centres"),
]


def get_navigation_labels(items: Iterable[NavigationItem] | None = None) -> list[str]:
    source = ENGINEERING_NAVIGATION if items is None else list(items)
    return [item.label for item in source]

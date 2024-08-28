from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    id: int
    unity_id: int
    name: str

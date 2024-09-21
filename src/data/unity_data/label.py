from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    """
    Label represents a Unity label.
    """
    id: int  # Internal id
    unity_id: int
    name: str

from dataclasses import dataclass

@dataclass
class Action:
    engine: str
    phase_names: list[str]
    phases: list
    phase_stats = {}


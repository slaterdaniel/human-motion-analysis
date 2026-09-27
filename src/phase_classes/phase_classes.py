from dataclasses import dataclass

@dataclass
class Action:
    engine: str
    phase_names: list[str]
    phases: list
    phase_stats = {}

empty = [None, None, None, None]
summary_data = {
    "": ['Right', 'All', 'Left', 'Optimal'],

    "General Parameters": empty,
    "Cadence": empty,
    "Vertical Oscillation": empty,
    "GCT": empty,
    "    ": empty,

    "Initial Contact Phase": empty,
    "Foot Inclination Angle at IC": empty,
    "Tibial Inclination Angle at IC": empty,
    "Distance Heelstrike to COM at IC": empty,
    "Knee Flexion Angle at IC": empty,
    "Hip Separation Angle at IC": empty,
    " ": empty,

    "Midstance Phase": empty,
    "Ant Pelvic Tilt": empty,
    "Pelvis Rotation": empty,
    "Ankle Dorsiflex at Midstance": empty,
    "Foot Pronation at Midstance": empty,
    "Peak hip adduction angle": empty,
    "Peak hip Internal rotation angle": empty,
    "Contralateral Pelvic drop": empty,
    "Forward lean angle": empty,
    "Knee Window": empty,
    "Knee flexion angle at midstance": empty,
    "  ": empty,

    "Terminal Stance": empty,
    "Hip extension": empty,
    "Ankle supination": empty,
    "   ": empty,

    "Swing Phase": empty,
    "Hip Flexion": empty,
}

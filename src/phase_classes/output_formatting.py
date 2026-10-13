import numpy as np


def get_phase_output_format(phase_name: str, phase_scores: np.ndarray, faulty_reps: np.ndarray, faulty_rep_frames: list) -> str:

    return f"""\n\n-------------------------------------------------------------------------------------------------------------\n\n

{phase_name}:\n\n

Phase Scores:\n
{phase_scores}\n\n

Faulty Reps:\n
{faulty_reps}\n\n

Faulty Rep Frames:\n
{faulty_rep_frames}

\n\n-------------------------------------------------------------------------------------------------------------\n\n"""
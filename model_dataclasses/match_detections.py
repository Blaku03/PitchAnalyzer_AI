from dataclasses import dataclass
from typing import Optional
import numpy as np
import supervision as sv


@dataclass
class MatchDetectionsData:
    """Class to represent yolo detections in a video frame.
    sv.Detections stores the data is np.ndarray format where in a array there all detections for a given frame.
    That's why team is an np.ndarray.
    """

    players_detections: sv.Detections
    ball_detection: sv.Detections
    frame: int
    player_ball_id: int = -1
    team: Optional[np.ndarray] = None

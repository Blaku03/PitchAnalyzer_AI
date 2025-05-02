from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import supervision as sv


@dataclass
class PlayersDetections:
    """Class to represent player detections in a video frame.
    sv.Detections stores the data is np.ndarray format where in a array there all detections for a given frame.
    That's why team is an np.ndarray.
    """

    players_detections: sv.Detections
    ball_detection: sv.Detections
    frame: int
    player_ball_id: int = -1
    team: Optional[np.ndarray] = None

    def to_records(self) -> List[Dict]:
        """Convert the detections in this frame to a list of records, each representing one detection."""
        records = []
        dets = self.players_detections
        frame = self.frame
        team = self.team if self.team is not None else [None] * len(dets)

        # Ensure team is a list of the same length as dets
        class_name_list = (
            dets.data.get("class_name", []) if dets.data is not None else []
        )

        for i in range(len(dets)):
            record = {
                "frame": frame,
                "x_min": dets.xyxy[i, 0],
                "y_min": dets.xyxy[i, 1],
                "x_max": dets.xyxy[i, 2],
                "y_max": dets.xyxy[i, 3],
                "confidence": (
                    dets.confidence[i] if dets.confidence is not None else None
                ),
                "class_id": dets.class_id[i] if dets.class_id is not None else None,
                "class_name": (
                    class_name_list[i] if i < len(class_name_list) else None
                ),
                "tracker_id": (
                    dets.tracker_id[i] if dets.tracker_id is not None else None
                ),
                "team": team[i] if team is not None else None,
            }
            records.append(record)
        return records

    @classmethod
    def to_df(cls, dets_list: List["PlayersDetections"]) -> pd.DataFrame:
        """Convert a list of PlayersDetections to a pandas DataFrame, with one row per detection."""
        all_records = []
        for players_dets in dets_list:
            all_records.extend(players_dets.to_records())
        return pd.DataFrame(all_records)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> List["PlayersDetections"]:
        """Reconstruct a list of PlayersDetections from a pandas DataFrame."""
        players_dets_list = []
        for frame, group in df.groupby("frame"):
            xyxy = group[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            confidence = (
                group["confidence"].to_numpy() if "confidence" in group else None
            )
            class_id = group["class_id"].to_numpy() if "class_id" in group else None
            class_name = (
                group["class_name"].to_numpy().astype("U10")
                if "class_name" in group
                else None
            )
            tracker_id = (
                group["tracker_id"].to_numpy() if "tracker_id" in group else None
            )
            team = (
                group["team"].to_numpy()
                if "team" in group and not group["team"].isna().all()
                else None
            )

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
                data={
                    "class_name": class_name,
                },
            )
            players_dets = cls(
                detections=detections,
                frame=frame,
                team=team,
            )
            players_dets_list.append(players_dets)
        return players_dets_list

from typing import Generator
import numpy as np
from model_dataclasses.player_detection import PlayersDetections
import supervision as sv
import copy


class PlayersAnnotator:
    @staticmethod
    def annotate_frame(
        frame: np.ndarray, players_detections: PlayersDetections
    ) -> np.ndarray:
        """
        Annotate the frame with player detections.

        Args:
            frame (np.ndarray): The input frame.
            players_detections (PlayersDetections): The player detections to annotate.

        Returns:
            np.ndarray: The annotated frame.
        """
        players_detections_copy = copy.deepcopy(players_detections)
        team = players_detections_copy.team
        if team is None:
            raise ValueError("Team array is None, cannot filter by team")
        mask = team == 2
        players_detections_copy.detections.class_id[mask] = 4
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(
                [
                    "#FF0000",
                    "#00BFFF",
                    "#FF1493",
                    "#FFFFFF",
                    "#00FF00",
                    "#FFFF00",
                    "#FFA500",
                    "#0000FF",
                ]
            )
        )
        return ellipse_annotator.annotate(
            scene=frame,
            detections=players_detections_copy.detections,
        )

    @classmethod
    def annotate_video(
        cls,
        frame_generator: Generator,
        players_detections_generator: Generator[PlayersDetections, None, None],
    ) -> Generator:
        for frame, players_detections in zip(
            frame_generator, players_detections_generator
        ):
            yield cls.annotate_frame(frame, players_detections)

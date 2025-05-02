from typing import Generator
import numpy as np
from model_dataclasses.players_detections import PlayersDetections
import supervision as sv
import copy
import pdb


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

        # Assign a new class id to the other team
        players_detections_copy = copy.deepcopy(players_detections)
        team = players_detections_copy.team
        if team is None:
            raise ValueError("Team array is None, cannot filter by team")
        mask = team == 2
        players_detections_copy.players_detections.class_id[mask] = 4

        players_annotator = sv.EllipseAnnotator(
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
        ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17
        )
        closest_player_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#7CB342"), base=20, height=12
        )

        frame = players_annotator.annotate(
            scene=frame,
            detections=players_detections_copy.players_detections,
        )
        frame = ball_annotator.annotate(
            scene=frame, detections=players_detections_copy.ball_detection
        )

        if players_detections_copy.player_ball_id != -1:
            frame = closest_player_annotator.annotate(
                scene=frame, 
                detections=players_detections_copy.players_detections[players_detections_copy.player_ball_id])

        return frame

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

from typing import Generator
import numpy as np
from annotators.base_annotator import BaseAnnotator
from model_dataclasses.match_detections import MatchDetectionsData
import supervision as sv
import copy


class GameEntitiesAnnotator(BaseAnnotator):
    """Annotates players, referees, and ball on video frames."""

    # Define color palette once as a class constant
    COLOR_PALETTE = sv.ColorPalette.from_hex(
        [
            "#FF0000",  # Team 1
            "#00BFFF",  # Goalkeeper
            "#FF1493",  # Referee
            "#FFFFFF",  # Other
            "#00FF00",  # Team 2
            "#FFFF00",  # Additional role 1
            "#FFA500",  # Ball
            "#0000FF",  # Additional role 2
        ]
    )

    @staticmethod
    def annotate_frame(
        frame: np.ndarray, data: MatchDetectionsData, **kwargs
    ) -> np.ndarray:
        """
        Annotate a frame with player detections.

        Args:
            frame: Input video frame
            data: Player detection data with team assignments
            **kwargs: Additional arguments (not used currently)

        Returns:
            np.ndarray: Annotated frame with player markings

        Raises:
            ValueError: If team information is missing
        """
        # Create a copy to avoid modifying the original data
        match_detections_copy = copy.deepcopy(data)

        # Verify team data exists
        team = match_detections_copy.team
        if team is None:
            raise ValueError("Team array is None, cannot filter by team")

        # Assign different class IDs for visualization
        team2_mask = team == 2
        match_detections_copy.players_detections.class_id[team2_mask] = 4

        # Create annotators for different elements
        players_annotator = sv.EllipseAnnotator(
            color=GameEntitiesAnnotator.COLOR_PALETTE
        )

        players_label_annotator = sv.LabelAnnotator(
            color=GameEntitiesAnnotator.COLOR_PALETTE,
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )

        ball_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17
        )

        closest_player_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#7CB342"), base=20, height=12
        )

        # Annotate players
        annotated_frame = players_annotator.annotate(
            scene=frame,
            detections=match_detections_copy.players_detections,
        )

        # Add player ID labels
        # labels = [
        #     f"#{tracker_id}"
        #     for tracker_id in players_data.players_detections.tracker_id
        # ]
        # annotated_frame = players_label_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=players_data.players_detections,
        #     labels=labels,
        # )

        # Annotate ball
        annotated_frame = ball_annotator.annotate(
            scene=annotated_frame, detections=match_detections_copy.ball_detection
        )

        # Highlight player closest to ball
        if match_detections_copy.player_ball_id != -1:
            annotated_frame = closest_player_annotator.annotate(
                scene=annotated_frame,
                detections=match_detections_copy.players_detections[
                    match_detections_copy.player_ball_id
                ],
            )

        return annotated_frame

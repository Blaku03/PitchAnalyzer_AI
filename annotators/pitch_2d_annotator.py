from typing import Generator
import numpy as np
from annotators.base_annotator import BaseAnnotator
from utils.pitch_utils import draw_pitch, draw_points_on_pitch
from mapping_2d.soccer_field import SoccerPitchConfiguration
from model_dataclasses.match_detections import MatchDetectionsData
import supervision as sv


class Pitch2DAnnotator(BaseAnnotator):
    """Annotates a 2D top-down view of the soccer pitch with player positions."""

    # Define color mapping as class constant
    COLOR_MAP = {
        "team1": "#FF0000",
        "team2": "#00FF00",
        "referee": "#FFFF00",
        "goalkeeper": "#00BFFF",
        "ball": "#FFA500",
    }

    @staticmethod
    def color_pitch_points(
        xy_points: np.ndarray, color: str, pitch_img: np.ndarray
    ) -> np.ndarray:
        """
        Draw colored points on a 2D pitch representation.

        Args:
            xy_points: Array of (x,y) coordinates to plot
            color: Hex color string for the points
            pitch_img: Base pitch image to draw on

        Returns:
            np.ndarray: Image with added colored points
        """
        if len(xy_points) == 0:
            return pitch_img

        return draw_points_on_pitch(
            config=SoccerPitchConfiguration(),
            xy=xy_points,
            face_color=sv.Color.from_hex(color),
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch_img,
        )

    @staticmethod
    def annotate_frame(
        mapped_2d_points: np.array, match_detections: MatchDetectionsData, **kwargs
    ) -> np.ndarray:
        """
        Create a 2D top-down view of the pitch with player positions.

        Args:
            frame: Not used in this annotator
            data: Tuple containing (xy_points, players_detections)
            **kwargs: Additional arguments (not used currently)

        Returns:
            np.ndarray: 2D pitch image with player positions
        """
        # Create base 2D pitch
        field_2d_img = draw_pitch(SoccerPitchConfiguration())

        # Draw different types of points with different colors
        # Referees
        referee_mask = (
            match_detections.players_detections.data["class_name"] == "referee"
        )
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            xy_points=mapped_2d_points[referee_mask],
            color=Pitch2DAnnotator.COLOR_MAP["referee"],
            pitch_img=field_2d_img,
        )

        # Goalkeepers
        goalkeeper_mask = (
            match_detections.players_detections.data["class_name"] == "goalkeeper"
        )
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            xy_points=mapped_2d_points[goalkeeper_mask],
            color=Pitch2DAnnotator.COLOR_MAP["goalkeeper"],
            pitch_img=field_2d_img,
        )

        # Ball
        ball_mask = match_detections.players_detections.data["class_name"] == "ball"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            xy_points=mapped_2d_points[ball_mask],
            color=Pitch2DAnnotator.COLOR_MAP["ball"],
            pitch_img=field_2d_img,
        )

        # Team 1
        team1_mask = match_detections.team == 1
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            xy_points=mapped_2d_points[team1_mask],
            color=Pitch2DAnnotator.COLOR_MAP["team1"],
            pitch_img=field_2d_img,
        )

        # Team 2
        team2_mask = match_detections.team == 2
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            xy_points=mapped_2d_points[team2_mask],
            color=Pitch2DAnnotator.COLOR_MAP["team2"],
            pitch_img=field_2d_img,
        )

        return field_2d_img

    @classmethod
    def annotate_video(
        cls,
        mapped_2d_points_generator: Generator[sv.KeyPoints, None, None],
        match_detections_generator: Generator[MatchDetectionsData, None, None],
    ) -> Generator:
        """
        Generate annotated 2D pitch views for a video.

        Args:
            frame_generator: Not used in this annotator
            xy_points_generator: Generator yielding player coordinates
            players_detections_generator: Generator yielding player detections

        Yields:
            np.ndarray: 2D pitch annotations
        """
        for mapped_2d_points, match_detections in zip(
            mapped_2d_points_generator, match_detections_generator
        ):
            yield cls.annotate_frame(
                mapped_2d_points=mapped_2d_points.xy[0], match_detections=match_detections
            )

from typing import Generator
import numpy as np
from utils.pitch_utils import draw_pitch, draw_points_on_pitch
from mapping_2d.soccer_field import SoccerPitchConfiguration
from model_dataclasses.players_detections import PlayersDetections
import supervision as sv


class Pitch2DAnnotator:
    @staticmethod
    def color_pitch_points(
        masked_xy_points: np.array, color: str, pitch_img: np.ndarray
    ) -> np.array:
        return draw_points_on_pitch(
            config=SoccerPitchConfiguration(),
            xy=masked_xy_points,
            face_color=sv.Color.from_hex(color),
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch_img,
        )

    @staticmethod
    def annotate_frame(
        xy_points: np.array,
        players_detections: PlayersDetections,
    ) -> np.ndarray:
        field_2d_img = draw_pitch(SoccerPitchConfiguration())

        # Draw referees
        referee_mask = (
            players_detections.players_detections.data["class_name"] == "referee"
        )
        referee_color = "#FFFF00"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            masked_xy_points=xy_points[referee_mask],
            color=referee_color,
            pitch_img=field_2d_img,
        )

        goalkeeper_mask = (
            players_detections.players_detections.data["class_name"] == "goalkeeper"
        )
        goalkeeper_color = "#00BFFF"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            masked_xy_points=xy_points[goalkeeper_mask],
            color=goalkeeper_color,
            pitch_img=field_2d_img,
        )

        ball_mask = players_detections.players_detections.data["class_name"] == "ball"
        ball_color = "#FFA500"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            masked_xy_points=xy_points[ball_mask],
            color=ball_color,
            pitch_img=field_2d_img,
        )

        team1_mask = players_detections.team == 1
        team1_color = "#FF0000"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            masked_xy_points=xy_points[team1_mask],
            color=team1_color,
            pitch_img=field_2d_img,
        )

        team2_mask = players_detections.team == 2
        team2_color = "#00FF00"
        field_2d_img = Pitch2DAnnotator.color_pitch_points(
            masked_xy_points=xy_points[team2_mask],
            color=team2_color,
            pitch_img=field_2d_img,
        )

        return field_2d_img

    @classmethod
    def annotate_video(
        cls,
        xy_points_generator: Generator,
        players_detections_generator: Generator[PlayersDetections, None, None],
    ) -> Generator:
        for xy_points, players_detections in zip(
            xy_points_generator, players_detections_generator
        ):
            yield cls.annotate_frame(xy_points.xy[0], players_detections)

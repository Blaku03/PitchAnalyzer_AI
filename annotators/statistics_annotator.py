import supervision as sv
from annotators.base_annotator import BaseAnnotator
from model_dataclasses.statistics_dataclass import StatisticsDataclass
from utils.video_utils import add_text_to_image
import numpy as np
from typing import Generator


class StatisticsAnnotator(BaseAnnotator):
    """Annotates statistical information on video frames."""

    @staticmethod
    def annotate_frame(
        frame: np.ndarray, data: StatisticsDataclass, **kwargs
    ) -> np.ndarray:
        """
        Annotate a frame with statistical information.

        Args:
            frame: Input video frame
            data: Tuple containing ball possession percentages (team1, team2)
            **kwargs: Additional arguments (not used currently)

        Returns:
            np.ndarray: Annotated frame with statistics
        """
        # Create stats panel
        rect = sv.Rect(x=1400, y=45, width=500, height=170)

        # Draw white background for better readability
        scene = sv.draw_filled_rectangle(
            scene=frame.copy(), rect=rect, color=sv.Color.WHITE
        )

        # Format and add text
        team1_ball_possession = data.team1_ball_possession
        team2_ball_possession = data.team2_ball_possession
        text = f"Ball possession: \nTeam1: {team1_ball_possession}%\nTeam2: {team2_ball_possession}%"

        scene = add_text_to_image(
            image_rgb=scene,
            label=text,
            font_thickness=3,
            font_scale=1.5,
            top_left_xy=(1450, 50),
            font_color_rgb=(0, 0, 0),
        )

        return scene

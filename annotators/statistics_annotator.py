import supervision as sv
from utils.video_utils import add_text_to_image
import numpy as np
from typing import Generator

class StatisticsAnnotator:
    @staticmethod
    def annotate_frame(
        frame: np.ndarray, ball_possesion: tuple[float, float]
    ) -> np.ndarray:
        rect = sv.Rect(x=1400,y=45,width=500,height=170)

        scene = sv.draw_filled_rectangle(scene=frame.copy(),rect=rect,color=sv.Color.WHITE)
        
        text = f"Ball possession: \nTeam1: {ball_possesion[0]}%\nTeam2: {ball_possesion[1]}%"
        scene = add_text_to_image(
            image_rgb=scene,
            label=text,
            font_thickness=3,
            font_scale=1.5,
            top_left_xy=(1450, 50),
            font_color_rgb=(0, 0, 0)
        )

        return scene

    @classmethod
    def annotate_video(
        cls,
        frame_generator: Generator,
        ball_possesion_generator: Generator[tuple[float, float], None, None],
    ) -> Generator:
        for frame, ball_possesion in zip(
            frame_generator, ball_possesion_generator
        ):
            yield cls.annotate_frame(frame, ball_possesion)
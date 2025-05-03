from typing import Generator
import numpy as np
import supervision as sv
from annotators.base_annotator import BaseAnnotator
from mapping_2d.soccer_field import SoccerPitchConfiguration


class PitchAnnotator(BaseAnnotator):
    """Annotates soccer pitch keypoints and edges on video frames."""

    @staticmethod
    def annotate_frame(
        frame: np.ndarray, data: tuple[sv.KeyPoints, sv.KeyPoints], **kwargs
    ) -> np.ndarray:
        """
        Annotate a frame with pitch keypoints and edges.

        Args:
            frame: Input video frame
            data: tuple containing keypoints and edges
            **kwargs: Additional arguments (not used currently)

        Returns:
            np.ndarray: Annotated frame with pitch markings
        """
        pitch_keypoints, pitch_edges = data
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex("#FF1493"), radius=8
        )
        annotated_frame = vertex_annotator.annotate(
            scene=frame,
            key_points=pitch_keypoints,
        )

        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex("#00BFFF"),
            thickness=2,
            edges=SoccerPitchConfiguration().edges,
        )
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame,
            key_points=pitch_edges,
        )

        return annotated_frame

from typing import Generator
import numpy as np
import supervision as sv
from mapping_2d.view_trans import ViewTransformer
from mapping_2d.soccer_field import SoccerPitchConfiguration


class FieldAnnotator:
    @staticmethod
    def annotate_frame_keypoints(
        frame: np.ndarray, field_keypoints: sv.KeyPoints
    ) -> np.ndarray:
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex("#FF1493"), radius=8
        )
        return vertex_annotator.annotate(
            scene=frame,
            key_points=field_keypoints,
        )

    @staticmethod
    def annotate_frame_edges(
        frame: np.ndarray, field_edges: sv.KeyPoints
    ) -> np.ndarray:
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex("#00BFFF"),
            thickness=2,
            edges=SoccerPitchConfiguration().edges,
        )

        return edge_annotator.annotate(
            scene=frame,
            key_points=field_edges,
        )

    @classmethod
    def annotate_video(
        cls,
        frame_generator: Generator,
        field_detections_generator: Generator[sv.KeyPoints, None, None],
    ) -> Generator:
        for frame, field_detections in zip(frame_generator, field_detections_generator):
            yield cls.annotate_frame_keypoints(frame, field_detections)

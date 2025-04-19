from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import numpy as np
from mapping_2d.soccer_field import SoccerPitchConfiguration
from mapping_2d.view_trans import ViewTransformer


class FieldTracker:
    def __init__(self, model_path: Path, filter_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.filter_threshold = filter_threshold

    def get_field_keypoints(self, frame: np.ndarray) -> tuple[sv.KeyPoints, np.array]:
        """
        Detect field keypoints in a video stream and return results as a generator.
        Args:
            frame_gen (generator): A generator yielding video frames.

        Returns:
            generator: A generator yielding KeyPoints with detection results.
        """
        results = self.model.predict(frame)[0]
        keypoints_supervision = sv.KeyPoints.from_ultralytics(results)
        confident_mask = keypoints_supervision.confidence[0] > self.filter_threshold

        filtered_xy = keypoints_supervision.xy[0][confident_mask]
        filtered_confidence = keypoints_supervision.confidence[0][confident_mask]

        filtered_keypoints = sv.KeyPoints(
            xy=filtered_xy[np.newaxis, ...],
            confidence=filtered_confidence[np.newaxis, ...],
        )

        return filtered_keypoints, confident_mask

    def get_field_edges(self, frame: np.ndarray) -> sv.KeyPoints:
        raw_soccer_verticies = np.array(SoccerPitchConfiguration().vertices)
        field_keypoint, confident_mask = self.get_field_keypoints(frame)
        filtered_soccer_verticies = raw_soccer_verticies[confident_mask]

        v_transformer = ViewTransformer(
            source=filtered_soccer_verticies, target=field_keypoint.xy[0]
        )
        transformed_keypoints = v_transformer.transform_points(
            points=raw_soccer_verticies
        )

        return sv.KeyPoints(xy=transformed_keypoints[np.newaxis, ...])

    def map_points_2d(self, frame: np.ndarray, points: np.array) -> sv.KeyPoints:
        raw_soccer_verticies = np.array(SoccerPitchConfiguration().vertices)

        field_keypoint, confident_mask = self.get_field_keypoints(frame)
        filtered_soccer_verticies = raw_soccer_verticies[confident_mask]

        v_transformer = ViewTransformer(
            source=field_keypoint.xy[0], target=filtered_soccer_verticies
        )
        transformed_keypoints = v_transformer.transform_points(points=points)

        return sv.KeyPoints(xy=transformed_keypoints[np.newaxis, ...])

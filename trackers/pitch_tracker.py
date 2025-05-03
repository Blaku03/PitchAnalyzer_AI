from pathlib import Path
from typing import Generator
from ultralytics import YOLO
import supervision as sv
import numpy as np
from mapping_2d.soccer_field import SoccerPitchConfiguration
from mapping_2d.view_trans import ViewTransformer
from model_dataclasses.match_detections import MatchDetectionsData


class PitchTracker:
    def __init__(self, model_path: Path, filter_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.filter_threshold = filter_threshold

    def get_pitch_keypoints(self, frame: np.ndarray) -> tuple[sv.KeyPoints, np.array]:
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

    def get_pitch_edges(self, frame: np.ndarray) -> sv.KeyPoints:
        raw_soccer_verticies = np.array(SoccerPitchConfiguration().vertices)
        field_keypoint, confident_mask = self.get_pitch_keypoints(frame)
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

        field_keypoint, confident_mask = self.get_pitch_keypoints(frame)
        filtered_soccer_verticies = raw_soccer_verticies[confident_mask]

        v_transformer = ViewTransformer(
            source=field_keypoint.xy[0], target=filtered_soccer_verticies
        )
        transformed_keypoints = v_transformer.transform_points(points=points)

        return sv.KeyPoints(xy=transformed_keypoints[np.newaxis, ...])

    def map_players_tracks_2d_generator(
        self,
        frame_generator: Generator,
        match_detections_generator: Generator[MatchDetectionsData, None, None],
    ) -> Generator[sv.KeyPoints, None, None]:
        """
        Lazily map each (frame, points) pair to KeyPoints.
        Stops when either `frames` or `points` is exhausted.
        """
        for frame, match_detections in zip(frame_generator, match_detections_generator):
            bottom_players_boxes = (
                match_detections.players_detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
            )
            yield self.map_points_2d(frame, bottom_players_boxes)

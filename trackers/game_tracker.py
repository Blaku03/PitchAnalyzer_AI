import random
import itertools
from pathlib import Path
from typing import Generator, List, Tuple
import numpy as np
from ultralytics import YOLO
import supervision as sv
from norfair import Tracker, Detection
from assigners.team_assigner import TeamAssigner
from model_dataclasses.players_detections import PlayersDetections


class GameTracker:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)
        self.conf = 0.1
        self.player_tracker = None
        self.ball_tracker = None
        self.team_assigner = TeamAssigner()

    def _initialize_trackers(self):
        """Initialize player and ball trackers with appropriate configurations"""
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            minimum_matching_threshold=0.4,
            lost_track_buffer=60,
        )

        # Ball tracker configuration
        def distance_function(detection, track):
            return np.linalg.norm(detection.points - track.estimate)

        self.ball_tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=50,
            initialization_delay=0,
            hit_counter_max=100,
        )

    def _detect_objects_in_frame(
        self, frame_generator: Generator, batch_size: int = 20
    ) -> Generator:
        """Batch process frames for object detection"""

        def process_batch(batch):
            yield from self.model.predict(batch, conf=self.conf, stream=True)

        batch = []
        for frame in frame_generator:
            batch.append(frame)
            if len(batch) == batch_size:
                yield from process_batch(batch)
                batch = []

        if batch:  # Process remaining frames
            yield from process_batch(batch)

    def _reservoir_sample(
        self, frame_generator: Generator, sample_size: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Reservoir sampling implementation for frame sampling"""
        reservoir = []
        for i, frame in enumerate(frame_generator, start=1):
            if len(reservoir) < sample_size:
                reservoir.append((frame, i))
            else:
                j = random.randint(1, i)
                if j <= sample_size:
                    reservoir[j - 1] = (frame, i)
        return zip(*reservoir) if reservoir else ([], [])

    def get_sample_frames(
        self, frame_generator: Generator, samples_num: int = 10
    ) -> Tuple[List[np.ndarray], List[PlayersDetections]]:
        """Sample frames and return detections"""
        sampled_frames, frame_indices = self._reservoir_sample(
            frame_generator, samples_num
        )

        detections = []
        for result in self._detect_objects_in_frame(iter(sampled_frames)):
            detections.append(sv.Detections.from_ultralytics(result))

        return list(sampled_frames), detections

    def _process_ball_detections(self, ball_dets: sv.Detections) -> sv.Detections:
        """Convert and track ball detections using Norfair tracker"""

        # Convert to Norfair detections
        norfair_dets = [
            Detection(
                points=np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]),
                scores=np.array([conf]),
            )
            for bbox, conf in zip(ball_dets.xyxy, ball_dets.confidence)
        ]

        tracked_balls = self.ball_tracker.update(norfair_dets)
        if not tracked_balls:
            return sv.Detections.empty()

        # Convert back to sv.Detections format
        ball_size = 15
        center = tracked_balls[0].estimate[0]
        xyxy = np.array(
            [
                [
                    center[0] - ball_size / 2,
                    center[1] - ball_size / 2,
                    center[0] + ball_size / 2,
                    center[1] + ball_size / 2,
                ]
            ]
        )

        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.5]),
            class_id=np.array([0]),
            tracker_id=np.array([tracked_balls[0].id]),
            data={"class_name": np.array(["ball"])},
        )

    def _process_frame_detections(
        self, detections: sv.Detections
    ) -> Tuple[sv.Detections, sv.Detections]:
        """Separate and process player and ball detections"""
        player_dets = detections[detections.data["class_name"] != "ball"]
        ball_dets = detections[detections.data["class_name"] == "ball"]

        tracked_players = self.player_tracker.update_with_detections(player_dets)
        tracked_ball = self._process_ball_detections(ball_dets)

        return tracked_players, tracked_ball

    def get_detections_from_frames(
        self, frame_generator: Generator
    ) -> Generator[PlayersDetections, None, None]:
        """Main processing pipeline for video frames"""
        self._initialize_trackers()

        # Split generator for parallel processing
        frame_streams = itertools.tee(frame_generator, 3)
        main_frame_stream, detection_stream, training_stream = frame_streams

        # Train team assigner
        sample_frames, sample_detections = self.get_sample_frames(training_stream)
        self.team_assigner.initialize_assigner(sample_frames, sample_detections)

        # Process detection stream
        detection_results = self._detect_objects_in_frame(detection_stream)

        for frame_num, (frame, result) in enumerate(
            zip(main_frame_stream, detection_results)
        ):
            detections = sv.Detections.from_ultralytics(result)
            players, ball = self._process_frame_detections(detections)

            player_detection = PlayersDetections(
                players_detections=players,
                ball_detection=ball,
                frame=frame_num,
                team=self.team_assigner.get_players_teams(frame, players),
            )

            yield player_detection

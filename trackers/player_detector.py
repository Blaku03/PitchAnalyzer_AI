import random
from ultralytics import YOLO
import itertools
import supervision as sv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, List, Tuple

from assigners.team_assigner import TeamAssigner
from model_dataclasses.player_detection import PlayersDetections


class PlayerDetector:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)
        self.conf = 0.1

    def _detect_objects_in_frame(
        self, frame_generator: Generator, batch_size=20
    ) -> Generator:
        """
        Detect objects in a sequence of frames using batch processing.
        Args:
            frame_gen (generator): A generator yielding video frames.
            batch_size (int): Number of frames per batch.

        Yields:
            generator: A generator yielding detection results for each frame.
        """

        def process_batch(batch):
            results = self.model.predict(batch, conf=self.conf, stream=True)
            for result in results:
                yield result

        batch = []
        for frame in frame_generator:
            batch.append(frame)
            if len(batch) == batch_size:
                yield from process_batch(batch)
                batch = []

        # Handle any remaining frames
        if batch:
            yield from process_batch(batch)

    def get_sample_frames(
        self, frame_generator: Generator, samples_num: int = 10
    ) -> Tuple[List[np.ndarray], List[PlayersDetections]]:
        """
        Sample N random frames from a video stream
        (using reservoir sampling [https://en.wikipedia.org/wiki/Reservoir_sampling]),
        then detect & track those frames.

        Args:
            frame_generator: A generator yielding video frames.
            samples_num:     Number of frames to sample.

        Returns:
            sampled_frames:            List of the sampled frame images.
            sampled_players_detections: List of PlayersDetections for those frames.
        """
        # Reservoir‐sample raw frames
        reservoir: List[Tuple[np.ndarray, int]] = []
        for i, frame in enumerate(frame_generator, start=1):
            if len(reservoir) < samples_num:
                reservoir.append((frame, i))
            else:
                j = random.randint(1, i)
                if j <= samples_num:
                    reservoir[j - 1] = (frame, i)

        sampled_frames, frame_indices = zip(*reservoir)

        # Run detection+tracking on those sampled frames
        det_gen = self._detect_objects_in_frame(iter(sampled_frames))

        sampled_players_detections: List[PlayersDetections] = []
        for frame, raw_dets, idx in zip(sampled_frames, det_gen, frame_indices):
            sup = sv.Detections.from_ultralytics(raw_dets)
            sampled_players_detections.append(sup)

        return list(sampled_frames), list(sampled_players_detections)

    def get_detections_from_frames(
        self, frame_generator: Generator
    ) -> Generator[PlayersDetections, None, None]:
        """
        Detect objects in a video stream and return results as a DataFrame.
        Args:
            frame_gen (generator): A generator yielding video frames.

        Returns:
            generator: A generator yielding DataFrames with detection results.
        """
        player_tracker = sv.ByteTrack(
            lost_track_buffer=60, minimum_matching_threshold=0.4
        )
        ball_tracker = sv.ByteTrack(
                track_thresh=0.05,    # Very low to catch weak ball detections
                match_thresh=0.25,    # Loose IoU matching for ball
                track_buffer=100,     # Long buffer to maintain ball tracks
                frame_rate=30         # Adjust based on your video's frame rate
            )
        team_assigner = TeamAssigner()

        # Duplicate the generator
        frame_gen1, frame_gen2, frame_gen3 = itertools.tee(frame_generator, 3)
        loop_frame_generator = frame_gen1
        frame_generator_detector = frame_gen2
        assigner_training_generator = frame_gen3

        detections_in_frame_generator = self._detect_objects_in_frame(
            frame_generator_detector
        )

        # get sample frames for team assigner training
        sample_frames, sample_player_detections = self.get_sample_frames(
            assigner_training_generator, samples_num=10
        )
        team_assigner.initialize_assigner(sample_frames, sample_player_detections)

        for frame_num, detections in enumerate(detections_in_frame_generator):
            current_frame = next(loop_frame_generator)

            detection_supervision = sv.Detections.from_ultralytics(detections)

            player_dets = detection_supervision[
                detection_supervision.data["class_name"] != "ball"
            ]
            ball_dets = detection_supervision[
                detection_supervision.data["class_name"] == "ball"
            ]

            # Update trackers
            tracked_players = player_tracker.update_with_detections(player_dets)
            tracked_ball = ball_tracker.update_with_detections(ball_dets)
            print(ball_dets)
            print(tracked_ball)
            current_frame_players_detections = PlayersDetections(
                tracked_players, ball_dets, frame_num
            )

            # Assign team to players
            teams_arr = team_assigner.get_players_teams(
                current_frame, current_frame_players_detections
            )
            current_frame_players_detections.team = teams_arr
            yield current_frame_players_detections

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

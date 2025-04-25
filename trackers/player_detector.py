from ultralytics import YOLO
import itertools
import supervision as sv
import pandas as pd
from pathlib import Path
from typing import Generator

from assigners.team_assigner import TeamAssigner
from model_dataclasses.player_detection import PlayersDetections


class PlayerDetector:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)

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
            results = self.model.predict(batch, conf=0.1, stream=True)
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
        tracker = sv.ByteTrack()
        team_assigner = TeamAssigner()
        all_frames_players_detections: list[PlayersDetections] = []

        # Duplicate the generator
        frame_gen1, frame_gen2 = itertools.tee(frame_generator)
        loop_frame_generator = frame_gen1
        frame_generator_detector = frame_gen2

        detections_in_frame_generator = self._detect_objects_in_frame(
            frame_generator_detector
        )

        for frame_num, detections in enumerate(detections_in_frame_generator):
            current_frame = next(loop_frame_generator)

            detection_supervision = sv.Detections.from_ultralytics(detections)
            detection_with_tracks = tracker.update_with_detections(
                detection_supervision
            )
            current_frame_players_detections = PlayersDetections(
                detection_with_tracks, frame_num
            )

            if frame_num == 0:
                # Initialize the TeamAssigner with the first frame
                team_assigner.initialize_assigner(
                    current_frame, current_frame_players_detections
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

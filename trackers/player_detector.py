import random
from ultralytics import YOLO
import itertools
import supervision as sv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, List, Tuple
from norfair import Tracker, Detection
from assigners.team_assigner import TeamAssigner
from model_dataclasses.player_detection import PlayersDetections
import pdb

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
        Detect objects in a video stream and return results with player and ball detections.
        Args:
            frame_generator (Generator): A generator yielding video frames.
    
        Returns:
            Generator: A generator yielding PlayersDetections with player and ball info.
        """
        # Player tracker using ByteTrack
        player_tracker = sv.ByteTrack(
            track_activation_threshold=0.4,  # Threshold for player tracks
            minimum_matching_threshold=0.4,  # Matching threshold for players
            lost_track_buffer=60            # Buffer for player tracks
        )
        team_assigner = TeamAssigner()
    
        # Norfair tracker for the ball
        def distance_function(detection, track):
            det_center = detection.points  # Single 2D point (center of bbox)
            track_center = track.estimate
            return np.linalg.norm(det_center - track_center)
    
        ball_tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=50,  # Adjust based on ball speed and frame rate
            initialization_delay=0,  # Start tracking immediately
            hit_counter_max=100     # Keep track for 100 frames without detection
        )
    
        # Duplicate the generator for frame processing
        frame_gen1, frame_gen2, frame_gen3 = itertools.tee(frame_generator, 3)
        loop_frame_generator = frame_gen1
        frame_generator_detector = frame_gen2
        assigner_training_generator = frame_gen3
    
        # Get detections from frames
        detections_in_frame_generator = self._detect_objects_in_frame(
            frame_generator_detector
        )
    
        # Train team assigner with sample frames
        sample_frames, sample_player_detections = self.get_sample_frames(
            assigner_training_generator, samples_num=10
        )
        team_assigner.initialize_assigner(sample_frames, sample_player_detections)
    
        # Process each frame
        for frame_num, detections in enumerate(detections_in_frame_generator):
            current_frame = next(loop_frame_generator)
    
            # Convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detections)
    
            # Separate player and ball detections
            player_dets = detection_supervision[
                detection_supervision.data["class_name"] != "ball"
            ]
            ball_dets = detection_supervision[
                detection_supervision.data["class_name"] == "ball"
            ]
    
            # Update player tracker
            tracked_players = player_tracker.update_with_detections(player_dets)
    
            # Convert ball detections to Norfair format (single 2D point: center of bbox)
            norfair_detections = [
                Detection(
                    points=np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]),  # Center: [(x_min + x_max)/2, (y_min + y_max)/2]
                    scores=np.array([conf])  # Confidence score
                )
                for bbox, conf in zip(ball_dets.xyxy, ball_dets.confidence)
            ]
    
            # Update ball tracker with Norfair
            tracked_balls = ball_tracker.update(detections=norfair_detections)
            ball_detections = sv.Detections.empty()
            
            # Convert tracked ball to sv.Detections
            if tracked_balls:
                # Assume one ball; get center point
                center = tracked_balls[0].estimate[0]  # [x_center, y_center]
                ball_size = 15  # Assume ball is ~15 pixels wide/tall; adjust as needed
                xyxy = np.array([[
                    center[0] - ball_size / 2,  # x_min
                    center[1] - ball_size / 2,  # y_min
                    center[0] + ball_size / 2,  # x_max
                    center[1] + ball_size / 2   # y_max
                ]])
                # Create sv.Detections object
                ball_detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.array([0.5]),  # Placeholder confidence; adjust as needed
                    class_id=np.array([0]),      # Ball class ID
                    tracker_id=np.array([tracked_balls[0].id]),  # Norfair track ID
                    data={"class_name": np.array(["ball"])}
                )
    
            # Create PlayersDetections object
            current_frame_players_detections = PlayersDetections(
                tracked_players, ball_detections, frame_num
            )
    
            # Assign teams to players
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

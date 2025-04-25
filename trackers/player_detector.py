from ultralytics import YOLO
import itertools
import pdb
import cv2
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator

from assigners.team_assigner import TeamAssigner
from model_dataclasses.player_detection import PlayersDetections
from utils.bbox_utils import get_bbox_width, get_center_of_bbox


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
            list: List of detections per frame (one list per frame).
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

    def get_detections_from_frames(self, frame_generator: Generator) -> pd.DataFrame:
        """
        Detect objects in a video stream and return results as a DataFrame.
        Args:
            frame_gen (generator): A generator yielding video frames.

        Returns:
            pd.DataFrame: DataFrame containing detection results.
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

            all_frames_players_detections.append(current_frame_players_detections)
            break

        return all_frames_players_detections

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

    def get_object_tracks(self, frame_gen, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as stub_file:
                tracks = pickle.load(stub_file)
            return tracks

        detections_frame = self.detect_frames(frame_gen)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detections in enumerate(detections_frame):
            class_names = detections.names
            class_names_inverted = {
                v: k for k, v in class_names.items()
            }  # Invert mapping

            # Convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detections)

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_names_inverted[
                        "player"
                    ]

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for idx in range(len(detection_with_tracks)):
                bbox = detection_with_tracks.xyxy[idx].tolist()
                class_id = detection_with_tracks.class_id[idx]
                track_id = detection_with_tracks.tracker_id[idx]

                if class_id == class_names_inverted["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_names_inverted["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_names_inverted["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            print("--------------------")
            print(f"Frame {frame_num}:")
            print(detection_with_tracks)

            if stub_path:
                with open(stub_path, "wb") as stub_file:
                    pickle.dump(tracks, stub_file)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(width, int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, frame_gen, tracks):
        for frame_num, frame in enumerate(frame_gen):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (255, 0, 0))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            yield frame

from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import sys
import os

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frame_gen, batch_size=20):
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
        for frame in frame_gen:
            batch.append(frame)
            if len(batch) == batch_size:
                yield from process_batch(batch)
                batch = []

        # Handle any remaining frames
        if batch:
            yield from process_batch(batch)

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

    def draw_ellipse(self, frame, bbox, color, track_id):
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

        return frame

    def draw_annotations(self, frame_gen, tracks):
        for frame_num, frame in enumerate(frame_gen):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            yield frame

import supervision as sv
from utils.bbox_utils import measure_distance, get_bottom_center_of_boxes
import numpy as np


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 100

    def get_distances_to_ball(
        self, ball_detection: sv.Detections, players_detections: sv.Detections
    ) -> np.array:
        n_detections = len(players_detections)
        if ball_detection.is_empty():
            return np.empty(n_detections)

        is_referee_mask = players_detections.data["class_name"] == "referee"
        ball_position = get_bottom_center_of_boxes(ball_detection)[0]
        players_positions = get_bottom_center_of_boxes(players_detections)

        players_distance = np.full(n_detections, None, dtype=object)

        for i in range(n_detections):
            if is_referee_mask[i]:
                continue
            player_position = players_positions[i]
            distance = measure_distance(player_position, ball_position)
            players_distance[i] = distance

        return players_distance

    def assign_player_to_ball(
        self, ball_detection: sv.Detections, players_detections: sv.Detections
    ) -> int:
        if ball_detection.is_empty():
            return -1

        players_distance = self.get_distances_to_ball(
            ball_detection, players_detections
        )
        smallest_distance = -1
        smallest_distance_index = -1
        for player_id, distance in enumerate(players_distance):
            if distance is not None and distance < self.max_player_ball_distance:
                if smallest_distance == -1 or distance < smallest_distance:
                    smallest_distance = distance
                    smallest_distance_index = player_id
        return smallest_distance_index

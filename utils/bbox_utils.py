from typing import Tuple
import supervision as sv
import numpy as np

def get_bottom_center_of_boxes(boxes: sv.Detections) -> np.array:
    return boxes.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.
    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).
        Returns:
            float: The Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

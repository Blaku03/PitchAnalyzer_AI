from typing import Tuple


def get_center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Calculate the center of a bounding box.
    Args:
        bbox (Tuple[float, float, float, float]): The bounding box in the format (x_min, y_min, x_max, y_max).
    Returns:
        Tuple[int, int]: The center coordinates (x, y) of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def get_bbox_width(bbox: Tuple[float, float, float, float]) -> int:
    """
    Calculate the width of a bounding box.
    Args:
        bbox (Tuple[float, float, float, float]): The bounding box in the format (x_min, y_min, x_max, y_max).
    Returns:
        int: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return int(x2 - x1)


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


def measure_xy_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Measure the distance between two points in x and y directions.
    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).
    Returns:
        Tuple[float, float]: The distance in x and y directions (dx, dy).
    """

    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Calculate the foot position of a player based on the bounding box.
    Args:
        bbox (Tuple[float, float, float, float]): The bounding box in the format (x_min, y_min, x_max, y_max).
    Returns:
        Tuple[int, int]: The foot position (x, y) of the player.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

import numpy as np
import cv2
from mapping_2d.soccer_field import SoccerPitchConfiguration


def map_field_points(field_points: np.ndarray, plane_points: np.ndarray) -> np.ndarray:
    """
    Map points using a homography matrix.
    """

    field_points = field_points.astype(np.float32)
    plane_points = plane_points.astype(np.float32)

    homography_matrix, _ = cv2.findHomography(plane_points, field_points)

    plane_points_for_transform = (
        np.array(SoccerPitchConfiguration().vertices)
        .reshape(-1, 1, 2)
        .astype(np.float32)
    )
    mapped_points = cv2.perspectiveTransform(
        plane_points_for_transform, homography_matrix
    )
    return mapped_points.reshape(-1, 2).astype(np.float32)

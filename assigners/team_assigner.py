import pdb
from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans

from model_dataclasses.player_detection import PlayersDetections


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def _get_clustering_model(self, image: np.array):
        """
        Get the K-means clustering model for the given image.
        Args:
            image: The image to be clustered.
        Returns:
            KMeans: The K-means clustering model.
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def _get_player_color(self, frame: np.array, bbox: np.array):
        """
        Get the color of the player in the given bounding box.
        Args:
            frame: The current video frame.
            bbox: The bounding box of the player in the format (x_min, y_min, x_max, y_max).
            Returns:
                The color of the player in the bounding box.
        """
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get Clustering model
        kmeans = self._get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Get the player cluster
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def initialize_assigner(
        self, frame: np.array, frame_player_detections: PlayersDetections
    ):
        """
        Initialize the team assigner with the first frame and player detections.
        This method uses K-means clustering to determine the team colors based on the player detections.
        Args:
            frame: The current video frame.
            frame_player_detections: The player detection object containing the bounding box and other information.
        """
        player_colors = []

        for idx in range(len(frame_player_detections.detections)):
            detection = frame_player_detections.detections[idx]
            if detection.data["class_name"] != "player":
                continue
            player_color = self._get_player_color(frame, detection.xyxy.flatten())
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_players_teams(
        self, frame: np.array, frame_player_detections: PlayersDetections
    ) -> np.array:
        """
        Get the team of the player based on the bounding box and K-means clustering.
        Args:
            frame: The current video frame.
            frame_player_detections: The player detection object containing the bounding box and other information.
        Returns:
            np.array: List of team IDs for the players in the current frame.
        """

        n_detections = len(frame_player_detections.detections)
        players_teams = np.full(n_detections, None, dtype=object)

        for idx in range(n_detections):
            player_detection = frame_player_detections.detections[idx]
            if player_detection.data["class_name"] != "player":
                continue

            # Check if the player ID is already assigned a team
            if player_detection.tracker_id[0] in self.player_team_dict:
                players_teams[idx] = self.player_team_dict[
                    player_detection.tracker_id[0]
                ]
                continue
            player_color = self._get_player_color(
                frame, player_detection.xyxy.flatten()
            )
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1
            self.player_team_dict[player_detection.tracker_id[0]] = team_id
            players_teams[idx] = team_id
        return players_teams

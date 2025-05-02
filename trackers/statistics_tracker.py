from model_dataclasses.players_detections import PlayersDetections
from typing import Generator

class StatisticsTracker:
    def __init__(self):
        self.frames_with_ball = 0
        self.team1_ball_possession = 0

    def _calculate_ball_possession(self):
        # Round up to the 2 decimal place
        team1_possession = round(
            (self.team1_ball_possession / self.frames_with_ball) * 100, 2
        )
        team2_possession = round(100 - team1_possession, 2)

        return team1_possession, team2_possession

    def update_with_detections(
        self, players_tracks: PlayersDetections
    ) -> tuple[float, float]:
        if players_tracks.player_ball_id == -1:
            return self._calculate_ball_possession()

        self.frames_with_ball += 1
        if players_tracks.team[players_tracks.player_ball_id] == 1:
            self.team1_ball_possession += 1
        if players_tracks.team[players_tracks.player_ball_id] == None:
            raise ValueError("Team ID is None")
        return self._calculate_ball_possession()

    def get_statictics_generator(self, players_tracks_generator: Generator[PlayersDetections, None, None],
    ) -> Generator[tuple[float, float], None, None]:
        for players_tracks in players_tracks_generator:
            yield self.update_with_detections(players_tracks)
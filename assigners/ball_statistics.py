import pdb
from model_dataclasses.player_detection import PlayersDetections


class BallStatictis:
    def stats(self, players_tracks: PlayersDetections):
        print(f"Number of balls: {len(players_tracks.ball_detection)}")

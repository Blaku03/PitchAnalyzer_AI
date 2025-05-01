import pdb
from model_dataclasses.player_detection import PlayersDetections


class BallStatictis:
    def stats(self, players_tracks: PlayersDetections):
        # pdb.set_trace()
        if players_tracks.ball_detection is not None:
            print(f"Number of balls: {len(players_tracks.ball_detection)}")
        else:
            print("No balls")

from model_dataclasses.match_detections import MatchDetectionsData
from model_dataclasses.statistics_dataclass import StatisticsDataclass
from typing import Generator


class StatisticsTracker:
    def __init__(self):
        self.frames_with_ball = 0
        self.team1_ball_possession = 0

    def _calculate_ball_possession(self) -> tuple[float, float]:
        if self.frames_with_ball == 0:
            return 0, 0

        # Round up to the 2 decimal place
        team1_possession = round(
            (self.team1_ball_possession / self.frames_with_ball) * 100, 2
        )
        team2_possession = round(100 - team1_possession, 2)

        return team1_possession, team2_possession

    def update_with_detections(
        self, match_detections: MatchDetectionsData
    ) -> StatisticsDataclass:
        if match_detections.player_ball_id == -1:
            team1_possession, team2_possession = self._calculate_ball_possession()
            return StatisticsDataclass(
                team1_ball_possession=team1_possession,
                team2_ball_possession=team2_possession,
            )

        self.frames_with_ball += 1

        # Get team of player with the ball
        player_team = match_detections.team[match_detections.player_ball_id]

        if player_team is None:
            raise ValueError("Team ID is None for player with ball")
        if player_team == 1:
            self.team1_ball_possession += 1

        # Calculate updated possession statistics
        team1_possession, team2_possession = self._calculate_ball_possession()
        return StatisticsDataclass(
            team1_ball_possession=team1_possession,
            team2_ball_possession=team2_possession,
        )

    def get_statictics_generator(
        self,
        match_detections_generator: Generator[MatchDetectionsData, None, None],
    ) -> Generator[StatisticsDataclass, None, None]:
        for match_detection in match_detections_generator:
            yield self.update_with_detections(match_detection)

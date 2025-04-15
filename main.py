from utils import read_video, save_video
import os
from trackers import Tracker
import itertools
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def main():
    # Example usage of read_video and save_video functions
    video_path = "sample_data/08fd33_4.mp4"
    output_path = "output_videos/output_video.avi"
    model_path = "./models/v1_1large/v1_1.pt"

    tracker = Tracker(model_path)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create generator of video frames
    frame_generator = read_video(video_path)

    tracks = tracker.get_object_tracks(
        frame_generator, read_from_stub=False, stub_path="stubs/track_stubs.pkl"
    )

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    frame_generator = read_video(video_path)

    # Assign team colors to players
    team_assigner = TeamAssigner()
    frame = next(frame_generator)
    player_detections = tracks["players"][0]
    team_assigner.assign_team_color(frame, player_detections)
    player_ball_assigner = PlayerBallAssigner()

    frame_generator = read_video(video_path)

    # Assign team to players
    for frame_num, player_track in enumerate(tracks["players"]):
        frame = next(frame_generator)
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_ball_assigner.assign_ball_to_player(
            player_track, ball_bbox
        )
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

    # save cropped image of a player
    # for track_id, player in tracks["players"][0].items():
    #     frame = next(gen_copy)
    #     bbox = player["bbox"]

    #     cropped_image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

    #     cv2.imwrite(f"output_videos/cropped_player_{track_id}.jpg", cropped_image)
    #     break

    # FIXME: We need to have another generator here because we used up the first one for tracks
    frame_generator = read_video(video_path)

    # Draw output
    output_video_frames = tracker.draw_annotations(frame_generator, tracks)

    # Save video frames directly from the generator
    frame_count = save_video(output_video_frames, output_path)

    print(f"Saved {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()

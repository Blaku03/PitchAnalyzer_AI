from utils import read_video, save_video
import os
from trackers import Tracker


def main():
    # Example usage of read_video and save_video functions
    video_path = "sample_data/08fd33_4.mp4"
    output_path = "output_video/output_video.avi"

    tracker = Tracker("models/weights/best.pt")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create generator of video frames
    frame_generator = read_video(video_path)

    tracks = tracker.get_object_tracks(
        frame_generator, read_from_stub=False, stub_path="stubs/track_stubs.pkl"
    )

    # FIXME: We need to have another generator here because we used up the first one for tracks
    frame_generator = read_video(video_path)

    # Draw output
    output_video_frames = tracker.draw_annotations(frame_generator, tracks)

    # Save video frames directly from the generator
    frame_count = save_video(output_video_frames, output_path)

    print(f"Saved {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()

import cv2
from pathlib import Path
from typing import Generator


def read_video(video_path: Path) -> Generator:
    """
    Read a video file and yield its frames one by one.

    Args:
        video_path (str): Path to the video file.

    Yields:
        numpy.ndarray: The next frame in the video.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file {video_path} not found.")

    video = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb
    finally:
        video.release()


def save_video(frame_generator: Generator, output_path: Path, fps: int = 30) -> int:
    """
    Save video frames to a file.

    Args:
        frame_generator (generator): A generator yielding video frames in RGB format.
        output_path (Path): Path to save the output video.
        fps (int): Frames per second for the output video.

    Returns:
        int: Number of frames written.
    """
    writer = None
    frame_count = 0

    for frame in frame_generator:
        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        frame_count += 1

    if writer is not None:
        writer.release()

    return frame_count

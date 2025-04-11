import cv2


def read_video(video_path):
    """
    Read a video file and yield its frames one by one.

    Args:
        video_path (str): Path to the video file.

    Yields:
        numpy.ndarray: The next frame in the video.
    """
    video = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            yield frame
    finally:
        video.release()


def save_video(frame_generator, output_path, fps=30):
    """
    Save video frames to a file.

    Args:
        frame_generator (generator): A generator yielding video frames.
        output_path (str): Path to save the output video.
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
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        writer.write(frame)
        frame_count += 1

    if writer is not None:
        writer.release()

    return frame_count

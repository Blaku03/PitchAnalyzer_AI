import cv2
from pathlib import Path
from typing import Generator
import numpy as np


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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        frame_count += 1

    if writer is not None:
        writer.release()

    return frame_count


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
    bg_color_rgb: tuple | None = None,
    outline_color_rgb: tuple | None = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb

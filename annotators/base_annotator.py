from typing import Any, Generator


class BaseAnnotator:
    """Base class for all annotators with common functionality."""

    @classmethod
    def annotate_video(
        cls,
        frame_generator: Generator,
        data_generator: Generator[Any, None, None],
        **kwargs
    ) -> Generator:
        """
        Generate annotated video frames by applying the annotate_frame method to each frame.

        Args:
            frame_generator: Generator yielding video frames
            data_generator: Generator yielding data to annotate frames with
            **kwargs: Additional arguments to pass to annotate_frame

        Yields:
            np.ndarray: Annotated frames
        """
        for frame, data in zip(frame_generator, data_generator):
            yield cls.annotate_frame(frame=frame, data=data, **kwargs)

from typing import Tuple
import numpy as np


def show_image(frame: np.array, title=None, size: Tuple[int, int] = (12, 12)):
    """
    Displays an image in a Jupyter notebook.

    Args:
        img (numpy.ndarray): The image to display.
        title (str, optional): The title of the image. Defaults to None.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=size)
    plt.imshow(frame)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

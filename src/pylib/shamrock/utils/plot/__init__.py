"""
Shamrock plot utility functions.
"""

import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import shamrock.sys
from PIL import Image


def show_image_sequence(glob_str, render_gif=True, dpi=200, interval=50, repeat_delay=10):
    """
    Create a matplotlib animation from a sequence of image files.

    Parameters
    ----------
    glob_str : str
        Glob pattern matching image files.
    render_gif : bool, optional
        Whether to render the animation.
    dpi : int, optional
        Dots per inch for the figure.
    interval : int, optional
        Delay between frames in milliseconds.
    repeat_delay : int, optional
        Delay before repeating the animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        Animation object on rank 0, otherwise None.
    """

    if not render_gif:
        return None

    if shamrock.sys.world_rank() != 0:
        return None

    files = sorted(glob.glob(glob_str))

    image_array = []
    for my_file in files:
        with Image.open(my_file) as image:
            image_array.append(image.copy())

    if not image_array:
        raise FileNotFoundError(f"No images found for glob pattern: {glob_str}")

    pixel_x, pixel_y = image_array[0].size

    # Create the figure and axes objects
    # Remove axes, ticks, and frame & set aspect ratio
    fig = plt.figure(dpi=dpi)
    plt.gca().set_position((0, 0, 1, 1))
    plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
    plt.axis("off")

    # Set the initial image with correct aspect ratio
    im = plt.imshow(image_array[0], animated=True, aspect="auto")

    def update(i):
        im.set_array(image_array[i])
        return (im,)

    # Create the animation object
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_array),
        interval=interval,
        blit=True,
        repeat_delay=repeat_delay,
    )

    return ani

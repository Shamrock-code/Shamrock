"""
Set the matplotlib style for shamrock (doc and standard plots).
"""


def set_shamrock_mpl_style():
    """
    Set the matplotlib style for shamrock (doc and standard plots).
    """
    try:
        import matplotlib as mpl  # pylint: disable=C0415
    except ImportError as e:
        raise ImportError(
            "matplotlib is required to use the shamrock matplotlib style. "
            "Please install it using 'pip install matplotlib'."
        ) from e

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "axes.grid.which": "major",  # grid only on major ticks
        }
    )

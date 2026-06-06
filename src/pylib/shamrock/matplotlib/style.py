"""
Set the matplotlib style for shamrock (doc and standard plots).
"""

import matplotlib as mpl


def set_shamrock_mpl_style():
    """
    Set the matplotlib style for shamrock (doc and standard plots).
    """

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
            "axes.facecolor": "#f2f2f2",
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
        }
    )

"""
Helpers for building standardized benchmark plots: several series on a
log-log scale, each ending in a non-overlapping value callout, with a
legend row underneath. Used across Shamrock's benchmark examples.
"""

import matplotlib.pyplot as plt

__all__ = ["add_end_labels", "make_std_bench_plot"]


def add_end_labels(ax_main, ax_annot, entries, x_pad=0.05, min_gap_px=25, fontsize=9):
    """
    Place non-overlapping value callouts in a dedicated annotation axes.

    `ax_annot` must share its y-axis with `ax_main` (i.e. have been created
    with ``sharey=ax_main``), so a "data" y-coordinate means the same thing
    in either axes. Each callout is linked back to its line's last data
    point in `ax_main` with a leader line.

    Parameters
    ----------
    ax_main : matplotlib.axes.Axes
        The axes holding the plotted lines.
    ax_annot : matplotlib.axes.Axes
        The (typically narrow, spine-less) axes the callout text is drawn
        into. Must share its y-axis with `ax_main`.
    entries : list of (float, float, str, color)
        One ``(x, y, text, color)`` tuple per callout, where ``(x, y)`` is
        the data point the leader line points to.
    x_pad : float, optional
        Horizontal position of the callout text, in `ax_annot` axes-fraction
        coordinates.
    min_gap_px : float, optional
        Minimum vertical spacing between callouts, in display pixels.
    fontsize : float, optional
        Font size of the callout text.
    """
    if not entries:
        return

    # Sort by data y-value and convert to display (pixel) coordinates so
    # spacing can be reasoned about independently of the (log) data scale
    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    disp_y = [ax_main.transData.transform((0, entries[i][1]))[1] for i in order]

    # Group overlapping labels into clusters and spread each cluster
    # symmetrically around the mean of its members' true positions, rather
    # than cascading everything upward when things get crammed
    clusters = []  # each: {"center": mean y, "count": n}
    for y in disp_y:
        clusters.append({"center": y, "count": 1})
        while len(clusters) >= 2:
            a, b = clusters[-2], clusters[-1]
            span_a = (a["count"] - 1) * min_gap_px
            span_b = (b["count"] - 1) * min_gap_px
            top_a = a["center"] + span_a / 2
            bot_b = b["center"] - span_b / 2
            if bot_b - top_a < min_gap_px:
                count = a["count"] + b["count"]
                center = (a["center"] * a["count"] + b["center"] * b["count"]) / count
                clusters[-2:] = [{"center": center, "count": count}]
            else:
                break

    disp_y = []
    for c in clusters:
        span = (c["count"] - 1) * min_gap_px
        start = c["center"] - span / 2
        disp_y.extend(start + k * min_gap_px for k in range(c["count"]))

    # ax_annot shares its y-axis with ax_main, so a "data" y-coordinate
    # means the same thing in either axes
    inv = ax_main.transData.inverted()
    for idx, y_disp in zip(order, disp_y):
        x_data, y_data, text, color = entries[idx]
        label_y_data = inv.transform((0, y_disp))[1]
        # mirror the bend when the label lands below its point, otherwise the
        # corner ends up on the wrong side and the leader line doubles back
        angle_b = 60 if label_y_data >= y_data else -60
        ax_annot.annotate(
            text,
            xy=(x_data, y_data),
            xycoords=ax_main.transData,
            xytext=(x_pad, label_y_data),
            textcoords=("axes fraction", "data"),
            color=color,
            fontsize=fontsize,
            va="center",
            ha="left",
            annotation_clip=False,
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.8,
                shrinkA=0,
                shrinkB=2,
                connectionstyle=f"angle,angleA=0,angleB={angle_b},rad=10",
            ),
        )


def make_std_bench_plot(
    plot_data,
    xlabel,
    ylabel,
    title,
    end_label_fmt=lambda y: f"{y:.2f}",
    before_plot_func=None,
    dpi=250,
    figsize=(8, 6),
    min_gap_px=75,
    legend_ncol=2,
):
    """
    Build a standardized benchmark plot.

    Layout: a log-log plot on the left (5/6 of the width), a value-callout
    panel on the right (1/6), and a legend row spanning the full width
    underneath.

    Parameters
    ----------
    plot_data : dict
        Maps a series key to a dict with keys ``x``, ``y``, ``color``,
        ``label``, ``linestyle`` and ``marker``, one entry per line to plot.
    xlabel, ylabel, title : str
        Axis labels and title for the main plot.
    end_label_fmt : callable, optional
        Formats a series' last y-value into its callout text.
    before_plot_func : callable, optional
        Called as ``before_plot_func(ax_main)`` before the series in
        `plot_data` are plotted, e.g. to draw reference lines underneath
        them.
    dpi, figsize : optional
        Passed to `matplotlib.pyplot.figure`.
    min_gap_px : float, optional
        Minimum vertical spacing between callouts, in display pixels; see
        `add_end_labels`.
    legend_ncol : int, optional
        Number of columns in the legend row.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The created figure and its main (plot) axes.
    """
    # Layout: 75%/25% split between the plot and its annotation panel on
    # top, with a legend row spanning the full width underneath
    fig = plt.figure(dpi=dpi, figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[5, 1], height_ratios=[5, 1.5], hspace=0.35, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_annot = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_legend = fig.add_subplot(gs[1, :])
    ax_annot.axis("off")
    ax_legend.axis("off")

    # finalize the outer figure margins before anything layout-dependent (the
    # end-label declutter math) reads axes geometry off of ax_main
    fig.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.06)

    if before_plot_func is not None:
        before_plot_func(ax_main)

    end_labels = []
    for d in plot_data.values():
        ax_main.plot(
            d["x"],
            d["y"],
            d["linestyle"],
            color=d["color"],
            label=d["label"],
            marker=d["marker"],
        )
        end_labels.append((d["x"][-1], d["y"][-1], end_label_fmt(d["y"][-1]), d["color"]))

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")

    ax_main.grid(True)

    add_end_labels(ax_main, ax_annot, end_labels, min_gap_px=min_gap_px)

    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", ncol=legend_ncol, fontsize=10)

    return fig, ax_main

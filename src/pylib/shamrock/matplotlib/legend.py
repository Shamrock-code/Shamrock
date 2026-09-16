"""
Legend helpers for plots with a family of curves colored via a colormap
sweep (e.g. ``cmap(i / N)``), where a single gradient-swatch legend entry
is preferable to one label per curve.
"""

__all__ = []

try:
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

if _HAS_MATPLOTLIB:
    __all__ = ["HandlerColormapLine", "add_cmap_legend_entry"]

    class HandlerColormapLine(HandlerBase):
        """Legend handler that draws a horizontal colormap gradient swatch."""

        def __init__(self, cmap, num_stripes=8):
            self.cmap = cmap
            self.num_stripes = num_stripes
            super().__init__()

        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
            y_center = height / 2 - ydescent
            stripe_lw = height * 0.6
            stripes = []
            for i in range(self.num_stripes):
                s = Line2D(
                    [
                        xdescent + i * width / self.num_stripes,
                        xdescent + (i + 1) * width / self.num_stripes,
                    ],
                    [y_center, y_center],
                    color=self.cmap(i / (self.num_stripes - 1)),
                    lw=stripe_lw,
                    solid_capstyle="butt",
                    transform=trans,
                )
                stripes.append(s)
            return stripes

    def add_cmap_legend_entry(
        ax, cmap, label, num_stripes=8, extra_handles=None, extra_labels=None, **legend_kwargs
    ):
        """Add a colormap-gradient swatch entry to ax's legend, optionally
        combined with normal labeled handles (extra_handles/extra_labels)."""
        proxy = Line2D([0], [0], color="none")
        handles = [proxy]
        labels = [label]
        if extra_handles:
            handles = extra_handles + handles
            labels = extra_labels + labels

        handler_map = {proxy: HandlerColormapLine(cmap, num_stripes=num_stripes)}
        legend_kwargs.setdefault("handlelength", 3)
        legend_kwargs.setdefault("handleheight", 1)
        legend_kwargs.setdefault("fontsize", 9)

        return ax.legend(handles, labels, handler_map=handler_map, **legend_kwargs)

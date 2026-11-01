from .legend import _HAS_MATPLOTLIB
from .style import set_shamrock_mpl_style

if _HAS_MATPLOTLIB:
    from .legend import HandlerColormapLine, add_cmap_legend_entry

import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Iterable, List, Optional, Tuple, Union

def plot_qbit_space(
    # --- general parameters ---
    n: int,
    tensor_bs_facecolor: Optional[str] = "white",   
    tensor_bs_edgecolor: Optional[str] = "black",
    tensor_bs_size: Optional[Tuple[float, float]] = None,
    tensor_bs_labels: Optional[Iterable[Union[str, int]]] = None,
    tensor_bs_visible: int = 0,
    font_size: Optional[int] = 2,
    font_color: Optional[str] = "white",
    edge_visible: int = 0,
    edge_alpha: float = 0.8,
    edge_linewidth: float = 0.3,
    edge_color: str = "lightgray",
    horizontal_spacing: float = 1.0,
    vertical_spacing: float = 0.5,
    title: Optional[str] = None,
    
    # --- probability coloring controls ---
    prob: Optional[Union[dict, Iterable, callable]] = None,  # dict/iter/callable giving values per subset
    prob_component: str = "abs",   # "abs" | "real" | "imag" | "phase"
    cmap: str = "gray_r",
    vmin: Optional[float] = None,  # None => infer from data
    vmax: Optional[float] = None,  # None => infer from data
    show_colorbar: bool = True,
    colorbar_pad: float = 0.12,
    colorbar_label: Optional[str] = None,
    
    # --- allow for allow external figure/axes to compose layouts
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if n < 0:
        raise ValueError("number of qbits n must be non-negative")

    # populate labels with integers from 0 to n if not otherwise specified
    if tensor_bs_labels is None:
        labels: List[Union[str, int]] = list(range(1, n + 1))
    else:
        labels = list(tensor_bs_labels)
        if len(labels) < n:
            labels += list(range(len(labels)+1, n+1))
        elif len(labels) > n:
            labels = labels[:n]

    # subsets by level
    levels = [list(itertools.combinations(range(n), k)) for k in range(n + 1)]

    # positions according to level
    positions = {}
    for k, level in enumerate(levels):
        m = len(level)
        total_width = (m - 1) * horizontal_spacing
        x_start = -total_width / 2
        xs = [x_start + i * horizontal_spacing for i in range(m)]
        y = -k * vertical_spacing
        for x, subset in zip(xs, level):
            positions[subset] = (x, y)

    # size at which tensor basis is displayed
    if tensor_bs_size is None:
        ew = horizontal_spacing * 0.6
        eh = vertical_spacing * 0.45
    else:
        ew, eh = tensor_bs_size

    # font size
    if font_size is None:
        font_size = max(7, 13 - max(0, n - 3))

    # currently, these are hardcoded fixes to adjust the figure size
    # note that fig/ax only created if not provided (so we can compose grids outside)
    created_fig = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(3, (n + 1) * 1.2)))
        created_fig = True

    # --- edges ---
    if edge_visible == 1:
        for k in range(n):
            for subset in levels[k + 1]:
                for i in subset:
                    parent = tuple(j for j in subset if j != i)
                    x1, y1 = positions[parent]
                    x2, y2 = positions[subset]
                    ax.plot([x1, x2], [y1, y2],
                            color=edge_color, alpha=edge_alpha,
                            linewidth=edge_linewidth)

    
    # --- compute values per basis (can be real or complex) ---
    # --- coloring from complex dict with L1 normalization over magnitudes ---
    all_subsets = [s for level in levels for s in level]

    # get raw complex values per subset (missings are set to 0)
    z_raw = [prob.get(s, 0.0) if isinstance(prob, dict) else 0.0 for s in all_subsets]

    # L1-normalize by magnitudes so sum(|z|) == 1
    sum_abs = sum(abs(z) for z in z_raw)
    if sum_abs > 0:
        z = [complex(zj) / sum_abs for zj in z_raw]
    else:
        z = [0j for _ in z_raw]

    # select what to display from the normalized z
    # options: "abs" | "real" | "imag" | "phase"
    prob_component = prob_component if 'prob_component' in locals() else "abs"

    if prob_component == "abs":
        vals = [abs(zj) for zj in z]          # in [0, 1], sums to 1
        vmin, vmax = 0.0, 1.0                 # fixed scale
    elif prob_component == "real":
        vals = [zj.real for zj in z]
        vmin, vmax = -1.0, 1.0                # fixed symmetric scale
    elif prob_component == "imag":
        vals = [zj.imag for zj in z]
        vmin, vmax = -1.0, 1.0
    elif prob_component == "phase":
        vals = [math.atan2(zj.imag, zj.real) for zj in z]
        vmin, vmax = -math.pi, math.pi
    else:
        raise ValueError("prob_component must be 'abs', 'real', 'imag', or 'phase'")

    # map to colors 
    if prob_component == "abs":
        cmap_name = cmap if 'cmap' in locals() else "gray_r"
    elif prob_component == "real":
        cmap_name = "plasma"
    elif prob_component == "imag":
        cmap_name = "plasma"
    elif prob_component == "phase":
        cmap_name = "plasma"
    else:
        raise ValueError("prob_component must be 'abs', 'real', 'imag', or 'phase'")
        
    #cmap_obj = cm.get_cmap(cmap_name)
    cmap_obj = mpl.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    facecolors = {s: cmap_obj(norm(v)) for s, v in zip(all_subsets, vals)}

    # --- plot nodes as rectangles ---
    for subset, (x, y) in positions.items():
        label = "∅" if len(subset) == 0 else "{" + ", ".join(str(labels[i]) for i in subset) + "}"
        rect = Rectangle(
            (x - ew/2, y - eh/2), ew, eh,
            facecolor=facecolors[subset],
            edgecolor=tensor_bs_edgecolor,
            linewidth=edge_linewidth
        )
        ax.add_patch(rect)
        
        if tensor_bs_visible == 1:
            ax.text(x, y, label, ha="center", va="center",
                fontsize=font_size, color=font_color)

    # format figure -- limits, aspect, title
    all_x = [pos[0] for pos in positions.values()]
    ax.set_xlim(min(all_x)-ew, max(all_x)+ew)
    ax.set_ylim(-n * vertical_spacing - eh, eh)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        base = ", ".join(map(str, labels))
        title = f"Tensor state for ({{{base}}}) q-bits"
    ax.set_title(title, fontsize=14)
    
    if show_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=colorbar_pad)
        if colorbar_label is None:
            if prob_component == "abs":
                label = "|z|"
            elif prob_component == "real":
                label = "Re(z)"
            elif prob_component == "imag":
                label = "Im(z)"
            else:
                label = "arg(z) [rad]"
        else:
            label = colorbar_label
        cbar.set_label(label, rotation=90)

    return fig, ax
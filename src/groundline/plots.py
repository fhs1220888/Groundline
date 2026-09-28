"""Small matplotlib helpers that give every evidence figure the same look."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated reference palette (see README > Figures).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"
WARNING = "#fab219"
PHASE_FILL = {
    "pre_test": "#f1f0ec",
    "startup": "#e6eefa",
    "mainstage": "#e8f5ee",
    "shutdown": "#fbede6",
    "post_test": "#f1f0ec",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
    }
)


def new_fig(rows: int = 1, height: float = 2.8, width: float = 9.0, sharex: bool = True):
    fig, axes = plt.subplots(rows, 1, figsize=(width, height * rows), sharex=sharex, squeeze=False)
    return fig, [a[0] for a in axes]


def shade_phases(ax, phases: list[dict] | None, label: bool = False, xlim: tuple[float, float] | None = None) -> None:
    if not phases:
        return
    for p in phases:
        ax.axvspan(p["t_start"], p["t_end"], color=PHASE_FILL.get(p["name"], "#f1f0ec"), zorder=0, lw=0)
        a, b = p["t_start"], p["t_end"]
        if xlim:
            a, b = max(a, xlim[0]), min(b, xlim[1])
        if label and b > a:
            ax.text(
                (a + b) / 2,
                1.0,
                p["name"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_2,
            )


def to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

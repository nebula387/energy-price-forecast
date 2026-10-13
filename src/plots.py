"""Единый стиль графиков проекта и разметка событий войны."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.events import events_frame

PRE = "#2a78d6"     # до кризиса
CRISIS = "#eb6834"  # кризис 2026
THIRD = "#1baf7a"
MUTED = "#a8a69f"   # прошлые годы / фон
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"


def setup() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 4.2),
        "figure.dpi": 110,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def mark_events(ax, label: bool = True, numbered: bool = True) -> None:
    """Вертикальные метки событий войны 2026 года.

    numbered=True — только номера 1..N (для длинных периодов, где подписи наложатся),
    расшифровка — в events_frame(); numbered=False — полные подписи лесенкой (для крупного плана).
    """
    for i, ev in events_frame().iterrows():
        ax.axvline(ev["date"], color=CRISIS, linewidth=0.8, alpha=0.7, zorder=1)
        if not label:
            continue
        text = f" {i + 1}" if numbered else f" {i + 1}. {ev['label']}"
        y = 0.98 if numbered else 0.98 - 0.08 * i
        ax.text(ev["date"], y, text, transform=ax.get_xaxis_transform(),
                fontsize=8 if numbered else 7.5, color=INK_2, va="top", ha="left")


def date_axis(ax, fmt: str = "%b %Y") -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

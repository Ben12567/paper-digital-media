from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "manuscript" / "origin_data" / "figure6_ablation_analysis.csv"
OUT_DIR = ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = OUT_DIR / "figure6_ablation_analysis_ab"

NAVY = "#173F70"
TEAL = "#2A8581"
PURPLE = "#765AA3"
ORANGE = "#D27B32"
CORAL = "#C85F61"
GOLD = "#B98B2E"
INK = "#1F2937"
GRID = "#DDE3EA"

with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
    ROWS = list(csv.DictReader(handle))

ABS_NAMES = ["Full model\n(EAI-CO)"] + [row["short_label"] for row in ROWS[1:]]
ABS_VALUES = np.array([float(row["composite_reward"]) for row in ROWS])
ABS_COLORS = [NAVY, TEAL, PURPLE, ORANGE, CORAL]

DROP_ROWS = sorted(ROWS[1:], key=lambda row: abs(float(row["reward_drop"])), reverse=True)
DROP_NAMES = [row["mechanism"].replace(" preservation", "").replace(" modeling", "").replace(" penalty", "").title() for row in DROP_ROWS]
DROP_VALUES = np.array([abs(float(row["reward_drop"])) for row in DROP_ROWS])
DROP_COLORS = [CORAL, ORANGE, PURPLE, TEAL]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.edgecolor": "#98A2B3",
            "axes.linewidth": 0.75,
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def label_panel(ax: plt.Axes, label: str, subtitle: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=11.8, fontweight="bold", va="top", color="#111827")
    ax.text(0.00, 1.045, subtitle, transform=ax.transAxes, fontsize=9.4, fontweight="bold", va="top", color=INK)


def style_chart(ax: plt.Axes, axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    ax.grid(axis=axis, color=GRID, lw=0.72, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)


def draw_absolute(ax: plt.Axes) -> None:
    x = np.arange(len(ABS_VALUES))
    bars = ax.bar(
        x,
        ABS_VALUES - 0.70,
        bottom=0.70,
        width=0.57,
        color=ABS_COLORS,
        edgecolor=[GOLD, TEAL, PURPLE, ORANGE, CORAL],
        linewidth=[1.55, 0.70, 0.70, 0.70, 0.70],
        zorder=3,
    )
    ax.set_ylim(0.70, 0.826)
    ax.set_yticks([0.70, 0.73, 0.76, 0.79, 0.82])
    ax.set_ylabel("Composite reward (R)", labelpad=8)
    ax.set_xticks(x, ABS_NAMES)
    ax.tick_params(axis="x", pad=6)
    style_chart(ax, "y")
    for index, (bar, value) in enumerate(zip(bars, ABS_VALUES)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.35,
            color=NAVY if index == 0 else "#344054",
            fontweight="bold" if index == 0 else "normal",
        )
    ax.plot([-0.29, 0.29], [ABS_VALUES[0], ABS_VALUES[0]], color=GOLD, lw=1.25, zorder=5)
    label_panel(ax, "(a)", "Absolute performance")


def draw_drop(ax: plt.Axes) -> None:
    y = np.arange(len(DROP_VALUES))
    ax.hlines(y, 0, DROP_VALUES, color=DROP_COLORS, lw=4.6, zorder=3, capstyle="round")
    ax.scatter(DROP_VALUES, y, s=88, facecolor="white", edgecolor=DROP_COLORS, linewidth=1.8, zorder=4)
    ax.scatter(DROP_VALUES, y, s=24, facecolor=DROP_COLORS, edgecolor="none", zorder=5)
    ax.set_xlim(0, 0.078)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06])
    ax.set_xlabel("Reward degradation (\N{GREEK CAPITAL LETTER DELTA}R)", labelpad=8)
    ax.set_yticks(y, DROP_NAMES)
    ax.invert_yaxis()
    style_chart(ax, "x")
    for ypos, value in zip(y, DROP_VALUES):
        ax.text(value + 0.0022, ypos, f"\N{MINUS SIGN}{value:.4f}", va="center", ha="left", fontsize=8.45, color="#344054")
    label_panel(ax, "(b)", "Marginal contribution of each mechanism")


def make_figure() -> plt.Figure:
    configure()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), gridspec_kw={"width_ratios": [1.11, 1.00]})
    figure.subplots_adjust(left=0.080, right=0.975, top=0.920, bottom=0.170, wspace=0.31)
    draw_absolute(axes[0])
    draw_drop(axes[1])
    return figure


def main() -> None:
    figure = make_figure()
    figure.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    figure.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    figure.savefig(STEM.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.08, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    with Image.open(STEM.with_suffix(".tiff")) as image:
        image.convert("RGB").save(STEM.with_suffix(".tiff"), compression="tiff_lzw")
    for extension in [".pdf", ".svg", ".png", ".tiff"]:
        path = STEM.with_suffix(extension)
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

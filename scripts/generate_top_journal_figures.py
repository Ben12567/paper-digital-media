import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscript" / "figures"
ANALYSIS_DIR = ROOT / "results" / "local_qwen7b_10p" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BLUE = "#2F6FED"
DARK_BLUE = "#174EA6"
LIGHT_BLUE = "#EAF1FF"
GREEN = "#2E7D5B"
LIGHT_GREEN = "#EAF7F0"
ORANGE = "#D96C2C"
LIGHT_ORANGE = "#FFF2E8"
GRAY = "#5E6472"
LIGHT_GRAY = "#F4F6F8"
MID_GRAY = "#D6DAE0"
DARK = "#202124"
RED = "#C44536"


def load_rows(filename):
    with (ANALYSIS_DIR / filename).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, fc=LIGHT_GRAY, ec=MID_GRAY, color=DARK, lw=1.2):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=9.2,
        linespacing=1.15,
    )
    return box


def arrow(ax, start, end, color=GRAY, lw=1.25, mutation_scale=12):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=mutation_scale,
        ),
    )


def figure_1_pipeline():
    fig, ax = plt.subplots(figsize=(13.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.94,
        "EAI-CO: exploratory AI-based creative optimization pipeline",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        ha="left",
    )
    ax.text(
        0.02,
        0.885,
        "Structured generation is coupled with multi-objective evaluation and controlled refinement.",
        fontsize=9.7,
        color=GRAY,
        ha="left",
    )

    modules = [
        ("Campaign\nbrief", 0.035, LIGHT_GRAY, MID_GRAY),
        ("Structured\nencoding", 0.185, LIGHT_GRAY, MID_GRAY),
        ("Exploration\naxes", 0.335, LIGHT_BLUE, BLUE),
        ("Candidate\npool", 0.485, LIGHT_GRAY, MID_GRAY),
        ("Multi-objective\nevaluator", 0.635, LIGHT_ORANGE, ORANGE),
        ("Elite selection\n& refinement", 0.785, LIGHT_GREEN, GREEN),
    ]
    y, w, h = 0.47, 0.115, 0.19
    centers = []
    for label, x, fc, ec in modules:
        rounded_box(ax, (x, y), w, h, label, fc=fc, ec=ec, lw=1.4)
        centers.append((x + w / 2, y + h / 2))

    for i in range(len(centers) - 1):
        arrow(ax, (centers[i][0] + w / 2 - 0.01, centers[i][1]), (centers[i + 1][0] - w / 2 + 0.01, centers[i + 1][1]))

    # Reward component tags around evaluator.
    tags = [
        ("relevance", 0.575, 0.735),
        ("clarity", 0.665, 0.775),
        ("audience fit", 0.735, 0.715),
        ("engagement", 0.565, 0.355),
        ("safety", 0.670, 0.315),
        ("diversity", 0.745, 0.375),
    ]
    for text, tx, ty in tags:
        ax.text(
            tx,
            ty,
            text,
            ha="center",
            va="center",
            fontsize=8.2,
            color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.23,rounding_size=0.12", fc="white", ec="#F0B27A", lw=0.8),
        )

    # Final creative card.
    card_x, card_y, card_w, card_h = 0.92, 0.36, 0.07, 0.36
    rounded_box(ax, (card_x, card_y), card_w, card_h, "", fc="white", ec=BLUE, lw=1.5)
    ax.text(card_x + card_w / 2, card_y + card_h - 0.045, "Final\ncreative", ha="center", va="top", fontsize=9.2, fontweight="bold", color=DARK_BLUE)
    for i, label in enumerate(["Headline", "Body", "CTA", "Visual prompt"]):
        ax.plot([card_x + 0.012, card_x + card_w - 0.012], [card_y + 0.235 - i * 0.055] * 2, color=MID_GRAY, lw=1.2)
        ax.text(card_x + 0.014, card_y + 0.25 - i * 0.055, label, ha="left", va="bottom", fontsize=6.9, color=GRAY)
    arrow(ax, (0.785 + w, y + h / 2), (card_x, card_y + card_h / 2), color=BLUE, lw=1.5)

    # Search budget note.
    ax.text(
        0.335,
        0.20,
        "Exploration variables: emotional appeal, benefit framing, copy length, CTA type, color direction, layout intent, and audience pain point.",
        fontsize=8.8,
        color=GRAY,
        ha="left",
    )
    ax.text(
        0.635,
        0.14,
        "Reward = weighted relevance, clarity, aesthetics, audience fit, engagement, diversity, and safety - factuality penalty",
        fontsize=8.8,
        color=GRAY,
        ha="left",
    )
    save_figure(fig, "fig1_eai_co_pipeline")


def figure_2_cycle():
    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Exploration-to-exploitation optimization loop", fontsize=14, fontweight="bold", color=DARK)
    ax.text(0.03, 0.885, "Each round expands the candidate set, scores alternatives, and refines the next prompt population.", fontsize=9.7, color=GRAY)

    nodes = [
        ("Generate\ncandidates", 0.16, 0.68),
        ("Score\ncandidates", 0.42, 0.68),
        ("Select\nelite set", 0.68, 0.68),
        ("Mutate axes\n& prompts", 0.68, 0.36),
        ("Regenerate\ncandidates", 0.42, 0.36),
        ("Final\nselection", 0.16, 0.36),
    ]
    box_w, box_h = 0.15, 0.13
    for label, cx, cy in nodes:
        fc = LIGHT_BLUE if "Generate" in label or "Regenerate" in label else LIGHT_GRAY
        ec = BLUE if "Generate" in label or "Regenerate" in label else MID_GRAY
        if "Final" in label:
            fc, ec = LIGHT_GREEN, GREEN
        rounded_box(ax, (cx - box_w / 2, cy - box_h / 2), box_w, box_h, label, fc=fc, ec=ec)

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    centers = [(x, y) for _, x, y in nodes]
    for a, b in pairs:
        arrow(ax, centers[a], centers[b], color=BLUE if a in (0, 3) else GRAY)
    # feedback arrow from regeneration to scoring
    verts = [(0.42, 0.44), (0.42, 0.55), (0.42, 0.60)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    ax.add_patch(PathPatch(path, lw=1.25, edgecolor=BLUE, facecolor="none"))
    arrow(ax, (0.42, 0.59), (0.42, 0.62), color=BLUE)

    # Example mini table.
    table_x, table_y = 0.03, 0.07
    ax.text(table_x, table_y + 0.18, "Illustrative candidate evolution", fontsize=10.2, fontweight="bold", color=DARK)
    rows = [
        ("R1", "emotional appeal", "warm family-oriented copy"),
        ("R1", "price appeal", "discount-oriented copy"),
        ("R2", "refined family appeal", "clearer, safer, more audience-specific copy"),
    ]
    col_x = [table_x, table_x + 0.09, table_x + 0.26]
    ax.text(col_x[0], table_y + 0.12, "Round", fontsize=8.5, fontweight="bold", color=GRAY)
    ax.text(col_x[1], table_y + 0.12, "Strategy", fontsize=8.5, fontweight="bold", color=GRAY)
    ax.text(col_x[2], table_y + 0.12, "Candidate tendency", fontsize=8.5, fontweight="bold", color=GRAY)
    for i, row in enumerate(rows):
        yy = table_y + 0.08 - i * 0.04
        ax.text(col_x[0], yy, row[0], fontsize=8.3, color=DARK)
        ax.text(col_x[1], yy, row[1], fontsize=8.3, color=DARK)
        ax.text(col_x[2], yy, row[2], fontsize=8.3, color=DARK)
    ax.plot([table_x, table_x + 0.73], [table_y + 0.105, table_y + 0.105], color=MID_GRAY, lw=0.9)
    save_figure(fig, "fig2_exploration_loop")


def figure_3_reward_bar():
    methods = [
        ("B0_Template", "Template"),
        ("B2_OpenSource_Only", "Open-source"),
        ("B1_SingleShot_API", "Single-pass"),
        ("B3_PromptEngineered_AI", "Prompt-Eng."),
        ("Ours_EAI_CO", "EAI-CO"),
    ]
    reward_by_method = {row["method"]: float(row["mean_reward"]) for row in load_rows("primary_method_summary.csv")}
    labels = [label for _, label in methods]
    values = [reward_by_method[method] for method, _ in methods]
    colors = [MID_GRAY, "#AEB7C2", "#95A3B5", "#6D8BC3", BLUE]

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.62)
    ax.set_xlim(0.64, 0.83)
    ax.set_xlabel("Composite reward")
    ax.set_title("Main benchmark reward by method", loc="left", fontweight="bold", pad=12)
    ax.grid(axis="x", color="#E5E8EC", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MID_GRAY)
    ax.tick_params(axis="y", length=0)
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", fontsize=9.5, color=DARK)
    ax.text(0.645, -0.75, "N = 40 product-audience tasks; Qwen2.5-7B-Instruct backbone.", fontsize=8.4, color=GRAY)
    save_figure(fig, "fig3_main_reward_bar")


def figure_5_latency_tradeoff():
    methods = [
        ("B0_Template", "Template", MID_GRAY),
        ("B2_OpenSource_Only", "Open-source", "#95A3B5"),
        ("B1_SingleShot_API", "Single-pass", "#6D8BC3"),
        ("B3_PromptEngineered_AI", "Prompt-Eng.", "#4E73B7"),
        ("Ours_EAI_CO", "EAI-CO", BLUE),
    ]
    reward_by_method = {row["method"]: float(row["mean_reward"]) for row in load_rows("primary_method_summary.csv")}
    latency_by_method = {row["method"]: float(row["mean_latency_ms"]) for row in load_rows("cost_summary.csv")}
    data = [
        (label, latency_by_method[method], reward_by_method[method], color)
        for method, label, color in methods
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, latency, reward, color in data:
        size = 145 if label == "EAI-CO" else 95
        ax.scatter(latency, reward, s=size, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        dx = 150 if label != "Template" else 80
        dy = 0.004 if label != "EAI-CO" else -0.009
        ax.text(latency + dx, reward + dy, label, fontsize=8.8, color=DARK)

    ax.set_xscale("symlog", linthresh=100, linscale=0.8)
    ax.set_xlim(0, 16000)
    ax.set_ylim(0.66, 0.825)
    ax.set_xlabel("Average latency per task (ms; symlog scale)")
    ax.set_ylabel("Composite reward")
    ax.set_title("Reward-latency trade-off", loc="left", fontweight="bold", pad=12)
    ax.grid(color="#E5E8EC", lw=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MID_GRAY)
    ax.spines["bottom"].set_color(MID_GRAY)
    ax.annotate(
        "higher quality,\nhigher compute",
        xy=(latency_by_method["Ours_EAI_CO"], reward_by_method["Ours_EAI_CO"]),
        xytext=(5500, 0.815),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1),
        color=DARK_BLUE,
        fontsize=8.7,
        ha="center",
    )
    ax.text(5, 0.663, "Latency includes model invocations needed to produce a complete creative package.", fontsize=8.4, color=GRAY)
    save_figure(fig, "fig5_reward_latency_tradeoff")


def main():
    figure_1_pipeline()
    figure_2_cycle()
    figure_3_reward_bar()
    figure_5_latency_tradeoff()
    print(f"Generated figures in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

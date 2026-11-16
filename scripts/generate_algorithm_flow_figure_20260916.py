from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch


REPO = Path(__file__).resolve().parents[1]
TARGET_IMGS = REPO / "output" / "manuscript" / "imgs"
REVIEW_OUT = REPO / "output" / "figures" / "algorithm_flow_20260916"

try:
    fm.fontManager.addfont(r"C:\Windows\Fonts\simsun.ttc")
    FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc").get_name()
except Exception:
    FONT = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 8.2,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

COLORS = {
    "train_fill": "#EAF1F7",
    "train_edge": "#6C8BA8",
    "infer_fill": "#EEF5F1",
    "infer_edge": "#5B8E7D",
    "freeze_fill": "#F6F0E6",
    "freeze_edge": "#BA8B47",
    "output_fill": "#F1ECF7",
    "output_edge": "#8B77AD",
    "arrow": "#6A7480",
    "text": "#26323D",
    "muted": "#68737E",
}


def box(ax, x: float, y: float, width: float, height: float, text: str, *, fill: str, edge: str, fontsize: float = 8.1) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill,
        edgecolor=edge,
        linewidth=0.85,
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=COLORS["text"],
        fontsize=fontsize,
        linespacing=1.25,
        transform=ax.transAxes,
        zorder=4,
    )


def arrow(ax, x1: float, y1: float, x2: float, y2: float, *, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            linestyle="--" if dashed else "-",
            color=COLORS["arrow"],
            connectionstyle="arc3,rad=0",
            transform=ax.transAxes,
            zorder=2,
        )
    )


def make_figure() -> tuple[plt.Figure, dict]:
    fig, ax = plt.subplots(figsize=(146 / 25.4, 3.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.015, 0.94, "训练阶段", ha="left", va="center", fontsize=9.0, color=COLORS["train_edge"], weight="bold", transform=ax.transAxes)
    ax.text(0.015, 0.47, "样本外推断与评价阶段", ha="left", va="center", fontsize=9.0, color=COLORS["infer_edge"], weight="bold", transform=ax.transAxes)

    top_y, bottom_y = 0.70, 0.24
    width, height = 0.19, 0.17
    xs = [0.055, 0.285, 0.515, 0.745]

    top_texts = [
        "训练样本\n历史窗口\n未来 OHLC",
        "候选生成\n16条Kronos路径",
        "双粒度评分\n路径级+区间级",
        "联合训练\n验证集选参",
    ]
    top_styles = [
        (COLORS["train_fill"], COLORS["train_edge"]),
        (COLORS["train_fill"], COLORS["train_edge"]),
        (COLORS["train_fill"], COLORS["train_edge"]),
        (COLORS["train_fill"], COLORS["train_edge"]),
        (COLORS["freeze_fill"], COLORS["freeze_edge"]),
    ]
    for x, text, (fill, edge) in zip(xs, top_texts, top_styles):
        box(ax, x, top_y, width, height, text, fill=fill, edge=edge)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1 + width, top_y + height / 2, x2, top_y + height / 2)

    bottom_texts = [
        "样本外输入\n90日历史窗口",
        "冻结候选\n16条路径",
        "Momus评分聚合\n加权/Top-1",
        "评价输出\nMSE/IC/Rank IC\nTop-50组合",
    ]
    bottom_styles = [
        (COLORS["infer_fill"], COLORS["infer_edge"]),
        (COLORS["freeze_fill"], COLORS["freeze_edge"]),
        (COLORS["infer_fill"], COLORS["infer_edge"]),
        (COLORS["output_fill"], COLORS["output_edge"]),
    ]
    for x, text, (fill, edge) in zip(xs, bottom_texts, bottom_styles):
        box(ax, x, bottom_y, width, height, text, fill=fill, edge=edge, fontsize=7.7)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1 + width, bottom_y + height / 2, x2, bottom_y + height / 2)

    # The selected model parameters are passed from training to inference.
    arrow(ax, xs[-1] + width / 2, top_y, xs[1] + width / 2, bottom_y + height, dashed=True)
    ax.text(0.80, 0.57, "冻结参数", ha="center", va="center", fontsize=7.4, color=COLORS["freeze_edge"], transform=ax.transAxes)

    legend_items = [
        Patch(facecolor=COLORS["train_fill"], edgecolor=COLORS["train_edge"], label="训练/可学习模块"),
        Patch(facecolor=COLORS["freeze_fill"], edgecolor=COLORS["freeze_edge"], label="冻结或传递状态"),
        Patch(facecolor=COLORS["output_fill"], edgecolor=COLORS["output_edge"], label="预测与实证输出"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=7.4, handlelength=1.25, columnspacing=1.0)

    manifest = {
        "font": FONT,
        "claim": "整体流程区分训练阶段与样本外推断阶段，并显式标识冻结候选生成、Momus评分聚合和投资评价输出。",
        "stages": ["训练样本", "候选生成", "双粒度评分", "联合训练", "样本外输入", "冻结候选", "评分聚合", "评价输出"],
        "target_manuscript_modified": False,
        "target_body_referenced": False,
        "outputs": {
            "review": str(REVIEW_OUT),
            "manuscript_imgs": str(TARGET_IMGS / "algorithm_flow.svg"),
        },
    }
    return fig, manifest


def main() -> None:
    REVIEW_OUT.mkdir(parents=True, exist_ok=True)
    TARGET_IMGS.mkdir(parents=True, exist_ok=True)
    fig, manifest = make_figure()
    for folder in (REVIEW_OUT, TARGET_IMGS):
        for ext, kwargs in (("svg", {"dpi": 600}), ("png", {"dpi": 600}), ("pdf", {"dpi": 600})):
            fig.savefig(folder / f"algorithm_flow.{ext}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)
    (REVIEW_OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

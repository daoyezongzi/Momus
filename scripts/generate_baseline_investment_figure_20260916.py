from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "output" / "hs300_lowfreq_baseline_extension_20260916"
EXT_NO_TRAIN = REPO / "output" / "hs300_lowfreq_baseline_extension_no_training_20260916"
BASE = REPO / "output" / "hs300_lowfreq_investment_20260908" / "results"
TARGET_IMGS = REPO / "output" / "manuscript" / "imgs"
REVIEW_OUT = REPO / "output" / "figures" / "baseline_investment_20260916"

try:
    fm.fontManager.addfont(r"C:\Windows\Fonts\simsun.ttc")
    FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc").get_name()
except Exception:
    FONT = "DejaVu Sans"

COLORS = {
    "kronos": "#737A85",
    "median": "#8EA4BF",
    "weighted": "#3D827C",
    "weighted_light": "#B7D8D2",
    "top1": "#8B77AD",
    "top1_light": "#CFC5E2",
    "random": "#8E6C8A",
    "bullish": "#C17C5E",
    "lowvol": "#5B8E7D",
    "benchmark": "#BA8B47",
    "grid": "#E2E6EC",
    "zero": "#8A949E",
}

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_curve(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    data = pd.read_csv(path, parse_dates=["datetime"])
    dates = pd.DatetimeIndex(data["datetime"])
    return dates, data["net_nav"].to_numpy(dtype=float) - 1.0, np.cumprod(1.0 + data["bench"].to_numpy(dtype=float)) - 1.0


def load_period(period: str) -> dict[str, object]:
    dates, kronos, benchmark = read_curve(BASE / period / "simple_mean_daily.csv")
    _, median, _ = read_curve(EXT / period / "median_daily.csv")
    random_curves = np.vstack(
        [read_curve(EXT_NO_TRAIN / period / f"random_candidate_seed_{seed}_daily.csv")[1] for seed in range(5)]
    )
    _, most_bullish, _ = read_curve(EXT_NO_TRAIN / period / "most_bullish_daily.csv")
    _, lowest_volatility, _ = read_curve(EXT_NO_TRAIN / period / "lowest_volatility_daily.csv")
    weighted = np.vstack([read_curve(BASE / period / f"momus_seed_{seed}_daily.csv")[1] for seed in range(5)])
    top1 = np.vstack([read_curve(EXT / period / f"momus_top1_seed_{seed}_daily.csv")[1] for seed in range(5)])
    return {
        "dates": dates,
        "kronos": kronos,
        "median": median,
        "random": random_curves,
        "most_bullish": most_bullish,
        "lowest_volatility": lowest_volatility,
        "weighted": weighted,
        "top1": top1,
        "benchmark": benchmark,
    }


def style_axis(ax, period: str, *, show_xlabel: bool = True) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.axhline(0, color=COLORS["zero"], lw=0.75, ls=(0, (4, 2)), zorder=0)
    ax.spines["bottom"].set_color("#7B8490")
    ax.spines["left"].set_color("#7B8490")
    ax.tick_params(length=3, width=0.6, pad=3)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 6, 12] if period == "2025" else [1, 3, 6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    if show_xlabel:
        ax.set_xlabel("2025年" if period == "2025" else "2026年上半年", labelpad=4)


def make_figure() -> tuple[plt.Figure, dict]:
    data = {period: load_period(period) for period in ("2025", "2026H1")}
    fig, axes = plt.subplots(2, 1, figsize=(146 / 25.4, 6.15), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.16, hspace=0.27)
    manifest = {
        "font": FONT,
        "periods": {},
        "claim": "冻结低频候选缓存下七种候选聚合或规则方法的均值累计净收益率路径；Momus阴影保留五个检查点账户范围。",
        "rendering": "mean curves only; Momus seed ranges retained as translucent bands",
    }

    for ax, period in zip(axes, ("2025", "2026H1")):
        row = data[period]
        dates = row["dates"]
        x = pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)]).append(dates)
        kronos = np.r_[0.0, row["kronos"]]
        median = np.r_[0.0, row["median"]]
        benchmark = np.r_[0.0, row["benchmark"]]
        random = np.vstack([np.zeros((1, row["random"].shape[0])), row["random"].T])
        random_mean = random.mean(axis=1)
        most_bullish = np.r_[0.0, row["most_bullish"]]
        lowest_volatility = np.r_[0.0, row["lowest_volatility"]]
        weighted = np.vstack([np.zeros((1, row["weighted"].shape[0])), row["weighted"].T])
        top1 = np.vstack([np.zeros((1, row["top1"].shape[0])), row["top1"].T])
        weighted_mean = weighted.mean(axis=1)
        top1_mean = top1.mean(axis=1)
        weighted_lo, weighted_hi = weighted.min(axis=1), weighted.max(axis=1)
        top1_lo, top1_hi = top1.min(axis=1), top1.max(axis=1)

        ax.fill_between(x, weighted_lo, weighted_hi, color=COLORS["weighted_light"], alpha=0.28, linewidth=0, zorder=1)
        ax.fill_between(x, top1_lo, top1_hi, color=COLORS["top1_light"], alpha=0.25, linewidth=0, zorder=1)
        ax.plot(x, random_mean, color=COLORS["random"], lw=1.55, ls=(0, (4, 1.5)), zorder=3)
        ax.plot(x, most_bullish, color=COLORS["bullish"], lw=1.55, ls=(0, (1.5, 1.5)), zorder=3)
        ax.plot(x, lowest_volatility, color=COLORS["lowvol"], lw=1.65, ls=(0, (6, 1.5)), zorder=3)
        ax.plot(x, benchmark, color=COLORS["benchmark"], lw=1.35, ls=(0, (4, 2)), zorder=3)
        ax.plot(x, kronos, color=COLORS["kronos"], lw=1.65, zorder=4)
        ax.plot(x, median, color=COLORS["median"], lw=1.65, ls=(0, (2, 1.5)), zorder=4)
        ax.plot(x, weighted_mean, color=COLORS["weighted"], lw=2.05, zorder=5)
        ax.plot(x, top1_mean, color=COLORS["top1"], lw=1.95, zorder=5)
        ax.set_xlim(x[0], x[-1])
        style_axis(ax, period, show_xlabel=(period == "2026H1"))
        ax.set_title("2025年" if period == "2025" else "2026年上半年", pad=7, fontsize=9.5, fontweight="normal")
        if period == "2025":
            ax.set_ylabel("累计净收益率", labelpad=5)
        else:
            ax.tick_params(axis="y", labelleft=True)
        ax.text(-0.15, 1.04, "a" if period == "2025" else "b", transform=ax.transAxes, ha="left", va="bottom")
        manifest["periods"][period] = {
            "days": int(len(dates)),
            "kronos_final": float(kronos[-1]),
            "median_final": float(median[-1]),
            "random_mean_final": float(random_mean[-1]),
            "most_bullish_final": float(most_bullish[-1]),
            "lowest_volatility_final": float(lowest_volatility[-1]),
            "weighted_mean_final": float(weighted_mean[-1]),
            "top1_mean_final": float(top1_mean[-1]),
            "benchmark_final": float(benchmark[-1]),
        }

    handles = [
        Line2D([0], [0], color=COLORS["benchmark"], lw=1.35, ls=(0, (4, 2)), label="沪深300"),
        Line2D([0], [0], color=COLORS["kronos"], lw=1.65, label="Kronos等权平均"),
        Line2D([0], [0], color=COLORS["median"], lw=1.65, ls=(0, (2, 1.5)), label="逐点中位数"),
        Line2D([0], [0], color=COLORS["random"], lw=1.55, ls=(0, (4, 1.5)), label="随机候选均值"),
        Line2D([0], [0], color=COLORS["bullish"], lw=1.55, ls=(0, (1.5, 1.5)), label="最高终点涨幅"),
        Line2D([0], [0], color=COLORS["lowvol"], lw=1.65, ls=(0, (6, 1.5)), label="最低预测波动"),
        Line2D([0], [0], color=COLORS["weighted"], lw=2.05, label="Momus加权聚合均值"),
        Line2D([0], [0], color=COLORS["top1"], lw=1.95, label="Momus首位候选均值"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=4, columnspacing=0.9, handletextpad=0.35)
    return fig, manifest


def main() -> None:
    REVIEW_OUT.mkdir(parents=True, exist_ok=True)
    TARGET_IMGS.mkdir(parents=True, exist_ok=True)
    source_paths = [
        BASE / period / "simple_mean_daily.csv"
        for period in ("2025", "2026H1")
    ]
    source_paths += [
        BASE / period / f"momus_seed_{seed}_daily.csv"
        for period in ("2025", "2026H1")
        for seed in range(5)
    ]
    source_paths += [
        EXT / period / "median_daily.csv"
        for period in ("2025", "2026H1")
    ]
    source_paths += [
        EXT / period / f"momus_top1_seed_{seed}_daily.csv"
        for period in ("2025", "2026H1")
        for seed in range(5)
    ]
    source_paths += [
        EXT_NO_TRAIN / period / f"random_candidate_seed_{seed}_daily.csv"
        for period in ("2025", "2026H1")
        for seed in range(5)
    ]
    source_paths += [
        EXT_NO_TRAIN / period / method
        for period in ("2025", "2026H1")
        for method in ("most_bullish_daily.csv", "lowest_volatility_daily.csv")
    ]
    fig, manifest = make_figure()
    for name, folder in (("investment_baselines", REVIEW_OUT), ("investment_baselines", TARGET_IMGS)):
        for ext, kwargs in (("svg", {"dpi": 600}), ("png", {"dpi": 600}), ("pdf", {"dpi": 600})):
            fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)
    manifest.update(
        {
            "source_hashes": {str(path): sha256(path) for path in source_paths},
            "target_manuscript_modified": False,
            "target_body_referenced": False,
            "outputs": {
                "review": str(REVIEW_OUT),
                "manuscript_imgs": str(TARGET_IMGS / "investment_baselines.svg"),
            },
        }
    )
    (REVIEW_OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

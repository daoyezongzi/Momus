from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm


REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "output" / "hs300_lowfreq_baseline_mse_20260916" / "mse_summary.csv"
TARGET_IMGS = REPO / "output" / "manuscript" / "imgs"
REVIEW_OUT = REPO / "output" / "figures" / "baseline_mse_lowfreq_20260916"

try:
    fm.fontManager.addfont(r"C:\Windows\Fonts\simsun.ttc")
    FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc").get_name()
except Exception:
    FONT = "DejaVu Sans"

COLORS = {
    "simple_mean": "#737A85",
    "median": "#8EA4BF",
    "random_candidate": "#8E6C8A",
    "most_bullish": "#C17C5E",
    "lowest_volatility": "#5B8E7D",
    "momus_weighted": "#3D827C",
    "momus_top1": "#8B77AD",
    "positive": "#2F716B",
    "negative": "#B46E5B",
    "grid": "#E2E6EC",
}

ORDER = [
    "simple_mean",
    "median",
    "random_candidate",
    "most_bullish",
    "lowest_volatility",
    "momus_weighted",
    "momus_top1",
]
LABELS = {
    "simple_mean": "Kronos等权平均",
    "median": "逐点中位数",
    "random_candidate": "随机候选均值",
    "most_bullish": "最高终点涨幅",
    "lowest_volatility": "最低预测波动",
    "momus_weighted": "Momus加权聚合",
    "momus_top1": "Momus首位候选",
}

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 8.4,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data() -> pd.DataFrame:
    data = pd.read_csv(SUMMARY)
    data = data[(data["future_information"] == False) & data["method"].isin(ORDER)].copy()  # noqa: E712
    data["method"] = pd.Categorical(data["method"], categories=ORDER, ordered=True)
    return data.sort_values(["period", "method"])


def make_figure(data: pd.DataFrame) -> tuple[plt.Figure, dict]:
    periods = ["2025", "2026H1"]
    fig, axes = plt.subplots(1, 2, figsize=(146 / 25.4, 3.34), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.19, wspace=0.16)
    xmax = float(data["ohlc_mse"].max()) * 1.19
    manifest = {
        "font": FONT,
        "metric": "per-stock history-normalized OHLC-MSE",
        "future_information_rows_excluded": True,
        "periods": {},
        "claim": "在冻结候选缓存的低频补充评价中，Momus首位候选和加权聚合的标准化OHLC-MSE低于固定规则基准；收益结论不由本图推断。",
    }

    for panel_index, (ax, period) in enumerate(zip(axes, periods)):
        row = data[data["period"] == period].set_index("method").reindex(ORDER).reset_index()
        y = np.arange(len(ORDER))
        values = row["ohlc_mse"].to_numpy(dtype=float)
        improvements = row["mse_improvement_vs_simple_pct"].to_numpy(dtype=float)
        bars = ax.barh(
            y,
            values,
            height=0.60,
            color=[COLORS[method] for method in ORDER],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.set_yticks(y, [LABELS[method] for method in ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.arange(0, 2.26, 0.5))
        ax.grid(axis="x", color=COLORS["grid"], lw=0.55, zorder=0)
        ax.tick_params(length=3, width=0.6, pad=3)
        ax.spines["bottom"].set_color("#7B8490")
        ax.spines["left"].set_color("#7B8490")
        ax.set_title("2025年" if period == "2025" else "2026年上半年", pad=7, fontsize=9.5, fontweight="normal")
        if panel_index == 0:
            ax.set_xlabel("标准化 OHLC-MSE（越低越好）", labelpad=6)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.text(-0.28, 1.04, "a" if period == "2025" else "b", transform=ax.transAxes, ha="left", va="bottom")

        for bar, value, improvement in zip(bars, values, improvements):
            sign = "+" if improvement >= 0 else "-"
            text = f"{value:.3f}  ({sign}{abs(improvement):.1f}%)"
            color = COLORS["positive"] if improvement > 0.05 else COLORS["negative"] if improvement < -0.05 else "#66717D"
            ax.text(value + xmax * 0.018, bar.get_y() + bar.get_height() / 2, text, va="center", ha="left", fontsize=7.1, color=color)

        kronos = row.iloc[0]
        manifest["periods"][period] = {
            "forecast_dates": int(kronos["windows"]),
            "valid_stock_horizons": int(kronos["valid_stock_horizons"]),
            "excluded_stock_horizons": int(kronos["excluded_stock_horizons"]),
            "values": {str(method): float(row.loc[row["method"] == method, "ohlc_mse"].iloc[0]) for method in ORDER},
            "improvement_vs_kronos_pct": {str(method): float(row.loc[row["method"] == method, "mse_improvement_vs_simple_pct"].iloc[0]) for method in ORDER},
        }

    return fig, manifest


def main() -> None:
    REVIEW_OUT.mkdir(parents=True, exist_ok=True)
    TARGET_IMGS.mkdir(parents=True, exist_ok=True)
    data = load_data()
    fig, manifest = make_figure(data)
    for folder in (REVIEW_OUT, TARGET_IMGS):
        for ext, kwargs in (("svg", {"dpi": 600}), ("png", {"dpi": 600}), ("pdf", {"dpi": 600})):
            fig.savefig(folder / f"baseline_mse_lowfreq.{ext}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)
    manifest.update(
        {
            "source_sha256": {str(SUMMARY): sha256(SUMMARY)},
            "target_manuscript_modified": False,
            "target_body_referenced": False,
            "outputs": {
                "review": str(REVIEW_OUT),
                "manuscript_imgs": str(TARGET_IMGS / "baseline_mse_lowfreq.svg"),
            },
        }
    )
    (REVIEW_OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

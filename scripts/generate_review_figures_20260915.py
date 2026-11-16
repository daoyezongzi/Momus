from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator, PercentFormatter


REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "output" / "manuscript"
OUT = MANUSCRIPT / "figures" / "review_20260915_alternatives"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_CSV = REPO / "output" / "results" / "hs300_main_20260907" / "main_table.csv"
INV_DIR = REPO / "output" / "hs300_lowfreq_investment_20260908" / "results"
RUN_CSV = INV_DIR / "investment_per_run.csv"
BODY_TYP = MANUSCRIPT / "body.typ"

try:
    fm.fontManager.addfont(r"C:\Windows\Fonts\simsun.ttc")
    FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc").get_name()
except Exception:
    FONT = "DejaVu Sans"

FS = 9.5
WIDTH_IN = 146 / 25.4
COLORS = {
    "gray": "#737A85",
    "slate": "#9AAAC1",
    "teal": "#3D827C",
    "teal_light": "#B7D8D2",
    "purple": "#8B77AD",
    "gold": "#BA8B47",
    "ink": "#46515E",
    "grid": "#E2E6EC",
    "zero": "#8A949E",
}

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.force_edgecolor": False,
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(ax, axis="y"):
    ax.grid(axis=axis, color=COLORS["grid"], lw=0.55, zorder=0)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#7B8490")
    ax.tick_params(length=3, width=0.6, pad=4)


def panel_letter(ax, letter):
    ax.text(
        -0.16,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="black",
        fontsize=FS,
    )


def save_figure(fig, name: str, claim: str, sources: list[str], manifest: dict):
    fig.canvas.draw()
    for ext, kwargs in (
        ("svg", {"dpi": 600}),
        ("pdf", {"dpi": 600}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            OUT / f"{name}.{ext}",
            bbox_inches="tight",
            pad_inches=0.04,
            **kwargs,
        )
    plt.close(fig)
    manifest["figures"][name] = {
        "claim": claim,
        "sources": sources,
        "outputs": [f"{name}.svg", f"{name}.pdf", f"{name}.png"],
    }


def coverage_timeline(manifest: dict):
    periods = [
        {
            "label": "2024年",
            "start": "2024-05-21",
            "end": "2024-12-30",
            "stocks": 351,
            "windows": 5251,
            "color": COLORS["teal"],
        },
        {
            "label": "2025年",
            "start": "2025-05-21",
            "end": "2025-12-31",
            "stocks": 352,
            "windows": 5241,
            "color": COLORS["purple"],
        },
        {
            "label": "2026年上半年",
            "start": "2026-05-22",
            "end": "2026-06-30",
            "stocks": 352,
            "windows": 701,
            "color": COLORS["gold"],
        },
    ]
    # A calendar axis plus a narrow, borderless information column keeps the
    # timeline uncluttered while retaining the window and stock counts.
    fig = plt.figure(figsize=(WIDTH_IN, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.15, 1.05], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1], sharey=ax)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.965, bottom=0.23)
    for i, row in enumerate(periods):
        y = 2 - i
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        ax.plot([start, end], [y, y], color=row["color"], lw=8.0, alpha=0.18, solid_capstyle="round", zorder=1)
        ax.plot([start, end], [y, y], color=row["color"], lw=2.6, solid_capstyle="round", zorder=3)
        ax.scatter([start, end], [y, y], s=27, color="white", edgecolor=row["color"], linewidth=1.35, zorder=4)
        if (end - start).days < 100:
            date_text = f"{start:%Y-%m-%d}至{end:%Y-%m-%d}"
            ax.text((start + (end - start) / 2), y + 0.22, date_text, ha="center", va="bottom", color="black")
        else:
            ax.text(start, y + 0.22, start.strftime("%Y-%m-%d"), ha="left", va="bottom", color="black")
            ax.annotate(end.strftime("%Y-%m-%d"), xy=(end, y), xytext=(0, -16), textcoords="offset points", ha="right", va="top", color="black")
    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-07-20"))
    ax.set_ylim(-0.48, 2.55)
    ax.set_yticks([2, 1, 0], [r["label"] for r in periods])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("交易日期", labelpad=6)
    ax.set_ylabel("评价区间", labelpad=5)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_letter(ax, "a")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(-0.48, 2.55)
    ax_info.axis("off")
    ax_info.axvline(0.0, color=COLORS["grid"], lw=0.65, zorder=0)
    for i, row in enumerate(periods):
        y = 2 - i
        txt = ax_info.text(
            0.11,
            y,
            f"{row['windows']:,}个窗口\n{row['stocks']}只股票",
            ha="left",
            va="center",
            color="black",
        )
        txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    save_figure(
        fig,
        "coverage_timeline",
        "实际评价区间、有效窗口数量与股票覆盖范围存在明显时期差异。",
        ["body.typ:表5.1"],
        manifest,
    )


def make_effect_table(main: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("pointwise median", "逐点中位数", COLORS["slate"], "s"),
        ("Momus weighted", "Momus加权聚合", COLORS["teal"], "D"),
        ("Momus top-1 candidate", "Momus首位候选", COLORS["purple"], "^"),
    ]
    rows = []
    for period in ["2024", "2025", "2026H1"]:
        base = main[(main.period == period) & (main.Method == "Kronos simple mean")].iloc[0]
        for source, label, color, marker in methods:
            r = main[(main.period == period) & (main.Method == source)].iloc[0]
            rows.append(
                {
                    "period": period,
                    "method": label,
                    "color": color,
                    "marker": marker,
                    "mse": (base["OHLC-MSE"] - r["OHLC-MSE"]) / base["OHLC-MSE"] * 100,
                    "mae": (base["OHLC-MAE"] - r["OHLC-MAE"]) / base["OHLC-MAE"] * 100,
                    "ic": r["Price IC"] - base["Price IC"],
                    "rankic": r["Price RankIC"] - base["Price RankIC"],
                }
            )
    return pd.DataFrame(rows)


def main_effect_delta(manifest: dict):
    main = pd.read_csv(MAIN_CSV)
    effects = make_effect_table(main)
    periods = ["2024", "2025", "2026H1"]
    period_labels = ["2024年", "2025年", "2026年\n上半年"]
    metrics = [
        ("mse", "均方误差改善率（%）", (-1.2, 14.5), "pct"),
        ("mae", "平均绝对误差改善率（%）", (-1.5, 8.8), "pct"),
        ("ic", "价格路径相关系数差值", (-0.032, 0.066), "ic"),
        ("rankic", "价格路径秩相关系数差值", (-0.032, 0.064), "ic"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.35))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.21, wspace=0.46, hspace=0.62)
    offsets = {"逐点中位数": -0.17, "Momus加权聚合": 0.0, "Momus首位候选": 0.17}
    for ax, (metric, ylabel, ylim, fmt), letter in zip(axs.flat, metrics, "abcd"):
        for i in range(3):
            if i % 2 == 0:
                ax.axvspan(i - 0.44, i + 0.44, color="#F6F8FA", zorder=0)
        ax.axhline(0, color=COLORS["zero"], lw=0.85, ls=(0, (4, 2)), zorder=1)
        for method in ["逐点中位数", "Momus加权聚合", "Momus首位候选"]:
            part = effects[effects.method == method]
            values = [part[part.period == p][metric].iloc[0] for p in periods]
            xx = np.arange(3) + offsets[method]
            ax.plot(
                xx,
                values,
                color=part.color.iloc[0],
                lw=1.0,
                alpha=0.52,
                zorder=2,
            )
            ax.scatter(
                xx,
                values,
                s=34 if method != "Momus加权聚合" else 40,
                marker=part.marker.iloc[0],
                facecolor=part.color.iloc[0],
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(3), period_labels)
        ax.set_ylabel(ylabel, labelpad=5)
        if fmt == "ic":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%+.02f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        clean(ax)
        panel_letter(ax, letter)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["slate"], markeredgecolor="white", markersize=6.5, label="逐点中位数"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["teal"], markeredgecolor="white", markersize=6.5, label="Momus加权聚合"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["purple"], markeredgecolor="white", markersize=6.5, label="Momus首位候选"),
        Line2D([0], [0], color=COLORS["zero"], lw=0.85, ls=(0, (4, 2)), label="Kronos等权平均"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, columnspacing=1.0, handletextpad=0.42)
    for t in leg.get_texts():
        t.set_color("black")
    save_figure(
        fig,
        "main_effect_delta",
        "相对Kronos等权平均，Momus降低了路径数值误差，但价格路径相关性改善随时期与聚合方式变化。",
        ["output/results/hs300_main_20260907/main_table.csv"],
        manifest,
    )


def daily_stats(period: str, method: str, seed: int | None = None):
    if method == "simple_mean":
        path = INV_DIR / period / "simple_mean_daily.csv"
    else:
        path = INV_DIR / period / f"momus_seed_{seed}_daily.csv"
    d = pd.read_csv(path)
    vol = d["net_return"].std(ddof=1) * np.sqrt(252)
    sharpe = d["net_return"].mean() / d["net_return"].std(ddof=1) * np.sqrt(252)
    return {
        "vol": vol,
        "sharpe": sharpe,
        "mdd": d["net_drawdown"].min(),
        "path": path,
    }


def investment_metric_table(run: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    rows = []
    seed_values: dict[str, list[float]] = {"cumulative": [], "vol": [], "sharpe": [], "mdd": []}
    for period in ["2025", "2026H1"]:
        simple_row = run[(run.period == period) & (run.method == "simple_mean")].iloc[0]
        simple_daily = daily_stats(period, "simple_mean")
        rows.append(
            {
                "period": period,
                "method": "Kronos等权平均",
                "cumulative": simple_row.cumulative_net_return,
                "vol": simple_daily["vol"],
                "sharpe": simple_daily["sharpe"],
                "mdd": simple_daily["mdd"],
            }
        )
        seeds = run[(run.period == period) & (run.method.str.startswith("momus_seed"))].copy()
        vals = {k: [] for k in seed_values}
        for _, r in seeds.iterrows():
            seed = int(str(r.method).split("_")[-1])
            s = daily_stats(period, "momus", seed)
            vals["cumulative"].append(float(r.cumulative_net_return))
            vals["vol"].append(float(s["vol"]))
            vals["sharpe"].append(float(s["sharpe"]))
            vals["mdd"].append(float(s["mdd"]))
        for k in seed_values:
            seed_values[k].extend(vals[k])
        rows.append(
            {
                "period": period,
                "method": "Momus五账户均值",
                **{k: float(np.mean(vals[k])) for k in vals},
            }
        )
    return pd.DataFrame(rows), seed_values


def investment_metrics_dumbbell(manifest: dict):
    run = pd.read_csv(RUN_CSV)
    metric_df, _ = investment_metric_table(run)
    periods = ["2025", "2026H1"]
    labels = ["2025年", "2026年\n上半年"]
    panels = [
        ("cumulative", "累计净收益率", (-0.22, 0.33), "pct"),
        ("vol", "年化波动率", (0.14, 0.20), "pct"),
        ("sharpe", "夏普比率", (-2.60, 1.80), "num"),
        ("mdd", "最大回撤", (-0.24, -0.10), "pct"),
    ]
    # A method-to-method slopegraph makes the direction and magnitude of the
    # financial change explicit, while the pale points retain all five Momus
    # accounts as observed runs.
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH_IN, 3.25), sharey=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.28, wspace=0.43)
    period_colors = {"2025": COLORS["teal"], "2026H1": COLORS["purple"]}
    period_light = {"2025": COLORS["teal_light"], "2026H1": "#CFC5E2"}
    for ax, (metric, xlabel, ylim, fmt), letter in zip(axs, panels, "abcd"):
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color=COLORS["zero"], lw=0.8, ls=(0, (4, 2)), zorder=0)
        for i, period in enumerate(periods):
            kronos = float(metric_df[(metric_df.period == period) & (metric_df.method == "Kronos等权平均")][metric].iloc[0])
            momus = float(metric_df[(metric_df.period == period) & (metric_df.method == "Momus五账户均值")][metric].iloc[0])
            col = period_colors[period]
            # Seed dots sit around the Momus endpoint, exposing account
            # dispersion without turning the summary into a box plot.
            if metric == "cumulative":
                seed_vals = run[(run.period == period) & (run.method.str.startswith("momus_seed"))].cumulative_net_return.to_numpy()
            elif metric == "vol":
                seed_vals = np.array([daily_stats(period, "momus", s)["vol"] for s in range(5)])
            elif metric == "sharpe":
                seed_vals = np.array([daily_stats(period, "momus", s)["sharpe"] for s in range(5)])
            else:
                seed_vals = np.array([daily_stats(period, "momus", s)["mdd"] for s in range(5)])
            for j, v in enumerate(seed_vals):
                ax.scatter(1 + (j - 2) * 0.024, v, s=18, color=period_light[period], alpha=0.9, edgecolor="none", zorder=2)
            ax.plot([0, 1], [kronos, momus], color=col, lw=1.75, alpha=0.78, zorder=3)
            ax.scatter(0, kronos, s=39, marker="o", color=col, edgecolor="white", linewidth=0.7, zorder=5)
            ax.scatter(1, momus, s=46, marker="D", color=col, edgecolor="white", linewidth=0.7, zorder=5)
            # Raw values are printed at each endpoint; a small white halo
            # preserves legibility on top of the connecting slope.
            def fmt_value(v):
                if fmt == "pct":
                    return f"{v:.1%}"
                return f"{v:.2f}"
            for x0, y0, dy in [(0, kronos, 10 if i == 0 else -15), (1, momus, 10 if i == 0 else -15)]:
                t = ax.annotate(fmt_value(y0), xy=(x0, y0), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom" if dy > 0 else "top", color="black")
                t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        ax.set_xlim(-0.30, 1.30)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1], ["Kronos\n等权", "Momus\n均值"])
        ax.set_xlabel(xlabel, labelpad=8)
        if fmt == "pct":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        clean(ax, "y")
        panel_letter(ax, letter)
    handles = [
        Line2D([0], [0], color=COLORS["teal"], lw=1.75, marker="D", markerfacecolor=COLORS["teal"], markeredgecolor="white", markersize=5.8, label="2025年"),
        Line2D([0], [0], color=COLORS["purple"], lw=1.75, marker="D", markerfacecolor=COLORS["purple"], markeredgecolor="white", markersize=5.8, label="2026年上半年"),
        Line2D([0], [0], color=COLORS["teal_light"], lw=0, marker="o", markerfacecolor=COLORS["teal_light"], markeredgecolor="none", markersize=5.2, label="Momus独立账户"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, columnspacing=1.0, handletextpad=0.42)
    for t in leg.get_texts():
        t.set_color("black")
    save_figure(
        fig,
        "investment_metrics_dumbbell",
        "五个独立Momus账户的均值相对Kronos等权基线在累计收益、风险和风险调整收益上的变化随时期而变。",
        ["output/hs300_lowfreq_investment_20260908/results/investment_per_run.csv", "output/hs300_lowfreq_investment_20260908/results/{2025,2026H1}/*_daily.csv"],
        manifest,
    )


def investment_risk_return(manifest: dict):
    """Risk-return plane for the financial interpretation of Table 5.3."""
    run = pd.read_csv(RUN_CSV)
    metric_df, _ = investment_metric_table(run)
    periods = ["2025", "2026H1"]
    panel_labels = ["2025年", "2026年上半年"]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.28, wspace=0.20)

    for ax, period, period_label, letter in zip(axs, periods, panel_labels, "ab"):
        simple_row = run[(run.period == period) & (run.method == "simple_mean")].iloc[0]
        kronos = metric_df[(metric_df.period == period) & (metric_df.method == "Kronos等权平均")].iloc[0]
        momus = metric_df[(metric_df.period == period) & (metric_df.method == "Momus五账户均值")].iloc[0]
        simple_daily = pd.read_csv(INV_DIR / period / "simple_mean_daily.csv")
        benchmark_x = simple_daily["bench"].std(ddof=1) * np.sqrt(252)
        benchmark_y = float(simple_row.benchmark_cumulative_return)

        # The pale points expose account dispersion; the line between the two
        # larger markers is the observed shift from the baseline to Momus.
        seeds = run[(run.period == period) & (run.method.str.startswith("momus_seed"))]
        seed_x = np.array([daily_stats(period, "momus", int(str(m).split("_")[-1]))["vol"] for m in seeds.method])
        seed_y = seeds.cumulative_net_return.to_numpy()
        ax.scatter(seed_x, seed_y, s=30, color=COLORS["teal_light"], alpha=0.85, edgecolor="white", linewidth=0.45, zorder=2)
        ax.plot([kronos.vol, momus.vol], [kronos.cumulative, momus.cumulative], color="#AEB9C1", lw=1.45, zorder=1)
        ax.scatter(kronos.vol, kronos.cumulative, s=47, marker="o", color=COLORS["gray"], edgecolor="white", linewidth=0.75, zorder=4)
        ax.scatter(momus.vol, momus.cumulative, s=54, marker="D", color=COLORS["teal"], edgecolor="white", linewidth=0.75, zorder=5)
        ax.scatter(benchmark_x, benchmark_y, s=49, marker="^", color=COLORS["gold"], edgecolor="white", linewidth=0.75, zorder=5)

        def callout(text, xy, xytext, ha="left", va="bottom"):
            t = ax.annotate(text, xy=xy, xytext=xytext, textcoords="offset points", ha=ha, va=va, color="black")
            t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

        # Direct labels keep the meaning of each point visible without a
        # detached legend for the three main financial objects.
        if period == "2025":
            callout(f"Kronos {kronos.cumulative:.1%}", (kronos.vol, kronos.cumulative), (7, 10))
            callout(f"Momus均值 {momus.cumulative:.1%}", (momus.vol, momus.cumulative), (8, -14), va="top")
            callout(f"沪深300 {benchmark_y:.1%}", (benchmark_x, benchmark_y), (-7, 9), ha="right")
        else:
            # Keep the two negative-return labels inside the panel so that the
            # lower-period annotation does not collide with the x-axis.
            callout(f"Kronos {kronos.cumulative:.1%}", (kronos.vol, kronos.cumulative), (-8, 13), ha="right")
            callout(f"Momus均值 {momus.cumulative:.1%}", (momus.vol, momus.cumulative), (8, 11), va="bottom")
            callout(f"沪深300 {benchmark_y:.1%}", (benchmark_x, benchmark_y), (-7, 9), ha="right")

        ax.set_xlim(0.14, 0.205)
        ax.set_ylim(-0.24, 0.335)
        ax.set_xlabel(f"{period_label}\n年化波动率", labelpad=8)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(0, color=COLORS["zero"], lw=0.8, ls=(0, (4, 2)), zorder=0)
        clean(ax)
        panel_letter(ax, letter)

    axs[0].set_ylabel("累计净收益率", labelpad=5)
    axs[1].tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gray"], markeredgecolor="white", markersize=6.2, label="Kronos等权平均"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["teal"], markeredgecolor="white", markersize=6.2, label="Momus五账户均值"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal_light"], markeredgecolor="white", markersize=5.4, label="Momus独立账户"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["gold"], markeredgecolor="white", markersize=6.2, label="沪深300"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, columnspacing=0.9, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color("black")
    save_figure(
        fig,
        "investment_risk_return",
        "风险收益平面直接呈现两期的累计净收益率、年化波动率与基准位置，并保留Momus账户离散性。",
        ["output/hs300_lowfreq_investment_20260908/results/investment_per_run.csv", "output/hs300_lowfreq_investment_20260908/results/{2025,2026H1}/*_daily.csv"],
        manifest,
    )


def investment_equity_accounts(manifest: dict):
    periods = ["2025", "2026H1"]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.25), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.25, wspace=0.23)
    all_curves = []
    for ax, period, letter in zip(axs, periods, "ab"):
        simple = pd.read_csv(INV_DIR / period / "simple_mean_daily.csv")
        seeds = [pd.read_csv(INV_DIR / period / f"momus_seed_{i}_daily.csv") for i in range(5)]
        dates = pd.to_datetime(simple["datetime"])
        x = pd.DatetimeIndex([dates.iloc[0] - pd.offsets.BDay(1)] + list(dates))
        seed_curves = np.vstack([np.r_[0.0, d.net_nav.to_numpy() - 1.0] for d in seeds])
        mean_curve = seed_curves.mean(axis=0)
        lower, upper = seed_curves.min(axis=0), seed_curves.max(axis=0)
        kronos_curve = np.r_[0.0, simple.net_nav.to_numpy() - 1.0]
        benchmark_curve = np.r_[0.0, np.cumprod(1 + simple.bench.to_numpy()) - 1.0]
        all_curves.extend([lower, upper, kronos_curve, benchmark_curve])
        ax.fill_between(x, lower, upper, color=COLORS["teal_light"], alpha=0.38, linewidth=0, zorder=1)
        for curve in seed_curves:
            ax.plot(x, curve, color=COLORS["teal"], alpha=0.28, lw=0.75, zorder=2)
        ax.plot(x, mean_curve, color=COLORS["teal"], lw=1.65, zorder=4)
        ax.plot(x, kronos_curve, color=COLORS["gray"], lw=1.20, zorder=4)
        ax.plot(x, benchmark_curve, color=COLORS["gold"], lw=1.20, ls=(0, (4, 2)), zorder=3)
        ax.axhline(0, color=COLORS["zero"], lw=0.8, zorder=0)
        if period == "2025":
            ticks = pd.to_datetime(["2025-01-01", "2025-06-01", "2025-12-01"])
            ticklabels = ["1月", "6月", "12月"]
            xlabel = "2025年"
        else:
            ticks = pd.to_datetime(["2026-01-01", "2026-03-01", "2026-06-01"])
            ticklabels = ["1月", "3月", "6月"]
            xlabel = "2026年上半年"
        ax.set_xticks(ticks, ticklabels)
        ax.set_xlabel(xlabel, labelpad=7)
        clean(ax, "y")
        panel_letter(ax, letter)
    lo = min(float(np.min(c)) for c in all_curves)
    hi = max(float(np.max(c)) for c in all_curves)
    axs[0].set_ylim(min(-0.25, lo - 0.02), max(0.32, hi + 0.02))
    axs[0].set_ylabel("累计净收益率", labelpad=5)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axs[1].tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([0], [0], color=COLORS["gold"], lw=1.20, ls=(0, (4, 2)), label="沪深300"),
        Line2D([0], [0], color=COLORS["gray"], lw=1.20, label="Kronos等权平均"),
        Line2D([0], [0], color=COLORS["teal"], lw=1.65, label="Momus五账户均值"),
        Line2D([0], [0], color=COLORS["teal"], alpha=0.28, lw=4.0, label="Momus账户范围"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, columnspacing=0.9, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color("black")
    save_figure(
        fig,
        "investment_equity_accounts",
        "分时期初始化的净值路径显示Momus五个独立账户的离散性，以及相对Kronos和沪深300的时期依赖性。",
        ["output/hs300_lowfreq_investment_20260908/results/{2025,2026H1}/*_daily.csv"],
        manifest,
    )


def main():
    run = pd.read_csv(RUN_CSV)
    source_paths = {
        "body.typ": BODY_TYP,
        "main_table.csv": MAIN_CSV,
        "investment_per_run.csv": RUN_CSV,
    }
    for period in ["2025", "2026H1"]:
        for name in ["simple_mean_daily.csv"] + [f"momus_seed_{i}_daily.csv" for i in range(5)]:
            source_paths[f"{period}/{name}"] = INV_DIR / period / name
    manifest = {
        "created": "2026-09-15",
        "backend": "Python matplotlib",
        "font": FONT,
        "font_size_pt": FS,
        "width_mm": 146,
        "text_color": "black",
        "primary_format": "SVG with editable text",
        "scope": "independent review candidates; existing module figures and body.typ are unchanged",
        "source_hashes": {k: sha256(v) for k, v in source_paths.items()},
        "figures": {},
    }
    coverage_timeline(manifest)
    main_effect_delta(manifest)
    investment_metrics_dumbbell(manifest)
    investment_risk_return(manifest)
    investment_equity_accounts(manifest)
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "figures": list(manifest["figures"]), "font": FONT}, ensure_ascii=False))


if __name__ == "__main__":
    main()

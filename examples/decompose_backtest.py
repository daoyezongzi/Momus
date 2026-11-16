"""各策略"交易胜率 + 单窗均收益"分解表。

从连续回测输出的 equity_*.csv 汇总每笔窗口级交易，回答"策略赚钱/亏钱出在
哪个环节"：下注频率(交易窗口数) / 方向准度(胜率) / 综合盈利(单窗均收益)。

口径语义（与 critic/backtest.py:trade_pnl 一致）：
- long_only  口径：仅 position > 0 视为真交易（做空信号被 clip 成空仓）
- long_short 口径：position != 0 均为真交易（含做空）
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

STRATEGIES = ["simple_mean", "critic_weighted", "critic_top1"]
REGIMES = ["gross_long_short", "gross_long_only", "net_long_short", "net_long_only"]
REGIME_LABELS = {
    "gross_long_short": "毛·多空对称",
    "gross_long_only": "毛·只做多",
    "net_long_short": "净·多空对称",
    "net_long_only": "净·只做多",
}


def parse_filename(basename):
    """解析 equity_{code}_{regime}_{strategy}.csv → (code, regime, strategy)。

    regime 和 strategy 都含下划线，用已知集合从尾部反向匹配最稳。
    """
    stem = basename[len("equity_"):-len(".csv")]
    for strat in STRATEGIES:
        if stem.endswith("_" + strat):
            rest = stem[: -(len(strat) + 1)]
            for regime in REGIMES:
                if rest.endswith("_" + regime):
                    code = rest[: -(len(regime) + 1)]
                    return code, regime, strat
    return None, None, None


def load_trades(path, regime):
    """读取一个 equity csv，返回该股票在该口径下每个真实交易窗口的收益率列表。"""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    eq = [float(r["equity"]) for r in rows]
    pos = [int(float(r["position"])) for r in rows]
    long_only = regime.endswith("long_only")
    rets = []
    for i in range(1, len(rows)):
        p = pos[i]
        is_trade = (p > 0) if long_only else (p != 0)
        if is_trade:
            rets.append(eq[i] / eq[i - 1] - 1.0)
    return rets


def build_decomposition(result_dir):
    """扫描目录，按 (regime, strategy) 聚合所有股票的窗口级交易。"""
    agg = defaultdict(list)  # (regime, strategy) -> [ret, ...]
    files = glob.glob(os.path.join(result_dir, "equity_*.csv"))
    for fp in files:
        code, regime, strat = parse_filename(os.path.basename(fp))
        if code is None:
            continue
        agg[(regime, strat)].extend(load_trades(fp, regime))
    return agg


def format_table(agg, n_stocks):
    lines = [
        "━" * 78,
        "各策略「交易胜率 + 单窗均收益」分解表",
        "━" * 78,
        f"股票数: {n_stocks} | 持有期: 10 交易日/窗口 | 口径语义见脚本注释",
        "━" * 78,
        "",
    ]
    for regime in REGIMES:
        lines.append(f"### {REGIME_LABELS[regime]}")
        lines.append("")
        lines.append(f"{'策略':<16} | {'交易窗口':>8} | {'胜率':>7} | {'单窗均收益':>10} | {'单窗中位数':>10}")
        lines.append("-" * 66)
        for strat in STRATEGIES:
            rets = agg.get((regime, strat), [])
            n = len(rets)
            if n == 0:
                lines.append(f"{strat:<16} | {'0':>8} | {'-':>7} | {'-':>10} | {'-':>10}")
                continue
            wins = sum(1 for r in rets if r > 0)
            wr = wins / n * 100
            avg = sum(rets) / n * 100
            srt = sorted(rets)
            med = srt[n // 2] * 100 if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2 * 100
            lines.append(
                f"{strat:<16} | {n:>8} | {wr:>6.1f}% | {avg:>+9.3f}% | {med:>+9.3f}%"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="策略交易胜率/单窗均收益分解表")
    parser.add_argument("--result-dir", default="output/results/continuous_full/")
    parser.add_argument("--out", default=None, help="输出文件（默认打印到 stdout）")
    args = parser.parse_args()

    agg = build_decomposition(args.result_dir)
    codes = set()
    for fp in glob.glob(os.path.join(args.result_dir, "equity_*.csv")):
        code, _, _ = parse_filename(os.path.basename(fp))
        if code:
            codes.add(code)

    table = format_table(agg, len(codes))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"已保存: {args.out}")
    print(table)


if __name__ == "__main__":
    main()

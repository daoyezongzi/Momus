"""连续资金曲线回测可视化。

生成图表：
1. 净值曲线图（三策略对比）
2. 回撤曲线图
3. 多股票对比热力图（可选）
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'simple_mean': '#95a5a6',      # 灰色
    'critic_weighted': '#3498db',  # 蓝色
    'critic_top1': '#e74c3c',      # 红色
}

STRATEGY_LABELS = {
    'simple_mean': 'Simple Mean (baseline)',
    'critic_weighted': 'Critic Weighted',
    'critic_top1': 'Critic Top-1',
}


def plot_equity_curve(
    equity_data: Dict[str, pd.DataFrame],
    code: str,
    regime_label: str,
    initial_capital: float,
    output_file: Path,
):
    """绘制净值曲线图。

    Args:
        equity_data: dict[strategy_name, DataFrame(['date', 'equity', 'position'])]
        code: 股票代码
        regime_label: 口径标签
        initial_capital: 初始资金
        output_file: 输出文件路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # 子图1: 净值曲线
    for strategy_name in ['simple_mean', 'critic_weighted', 'critic_top1']:
        if strategy_name not in equity_data:
            continue

        df = equity_data[strategy_name]
        df['date'] = pd.to_datetime(df['date'])

        ax1.plot(
            df['date'],
            df['equity'],
            label=STRATEGY_LABELS[strategy_name],
            color=COLORS[strategy_name],
            linewidth=2,
            alpha=0.9,
        )

    # 初始资金基准线
    ax1.axhline(
        y=initial_capital,
        color='black',
        linestyle='--',
        linewidth=1,
        alpha=0.5,
        label=f'初始资金 ({initial_capital:,.0f})'
    )

    ax1.set_ylabel('净值 (元)', fontsize=12)
    ax1.set_title(f'{code} — {regime_label} 净值曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 子图2: 回撤曲线
    for strategy_name in ['simple_mean', 'critic_weighted', 'critic_top1']:
        if strategy_name not in equity_data:
            continue

        df = equity_data[strategy_name]
        equity = df['equity'].values

        # 计算回撤
        cumulative_max = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative_max) / cumulative_max * 100  # 百分比

        ax2.plot(
            df['date'],
            drawdown,
            label=STRATEGY_LABELS[strategy_name],
            color=COLORS[strategy_name],
            linewidth=2,
            alpha=0.9,
        )

        # 填充回撤区域
        ax2.fill_between(
            df['date'],
            drawdown,
            0,
            color=COLORS[strategy_name],
            alpha=0.2,
        )

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_title('回撤曲线', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  图表已保存: {output_file}")


def plot_multi_stock_heatmap(
    summary_data: pd.DataFrame,
    metric: str,
    output_file: Path,
):
    """绘制多股票对比热力图。

    Args:
        summary_data: DataFrame with columns ['code', 'strategy', metric]
        metric: 要可视化的指标名称
        output_file: 输出文件路径
    """
    # 透视表
    pivot = summary_data.pivot(index='code', columns='strategy', values=metric)

    # 重排列顺序
    pivot = pivot[['simple_mean', 'critic_weighted', 'critic_top1']]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in pivot.columns])
    ax.set_yticklabels(pivot.index)

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格添加数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = f'{value:.2%}' if 'return' in metric or 'mdd' in metric else f'{value:.2f}'
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    ax.set_title(f'多股票对比 — {metric}', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  热力图已保存: {output_file}")


def main(args):
    """主函数。"""
    equity_dir = Path(args.equity_dir)

    if not equity_dir.exists():
        print(f"错误: 目录不存在 {equity_dir}")
        return

    # 读取所有净值曲线文件
    equity_files = list(equity_dir.glob('equity_*.csv'))

    if not equity_files:
        print(f"错误: 未找到净值曲线文件 (equity_*.csv) 在 {equity_dir}")
        return

    # 按股票和口径分组
    data_by_code_regime = {}

    for csv_file in equity_files:
        # 文件名格式: equity_{code}_{regime}_{strategy}.csv
        parts = csv_file.stem.split('_')
        if len(parts) < 4:
            continue

        code = parts[1]
        regime = parts[2]
        strategy = '_'.join(parts[3:])

        key = (code, regime)
        if key not in data_by_code_regime:
            data_by_code_regime[key] = {}

        df = pd.read_csv(csv_file)
        data_by_code_regime[key][strategy] = df

    # 生成每个股票×口径的图表
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    regime_labels = {
        "gross_long_short": "毛·多空对称",
        "gross_long_only": "毛·只做多",
        "net_long_short": "净·多空对称",
        "net_long_only": "净·只做多",
    }

    for (code, regime), equity_data in data_by_code_regime.items():
        print(f"\n生成图表: {code} — {regime}")

        output_file = figures_dir / f"equity_{code}_{regime}.png"

        plot_equity_curve(
            equity_data,
            code,
            regime_labels.get(regime, regime),
            args.initial_capital,
            output_file,
        )

    print(f"\n所有图表已生成，保存在: {figures_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="连续资金曲线回测可视化")
    parser.add_argument(
        "--equity-dir",
        default="output/results/continuous/",
        help="净值曲线 CSV 文件目录"
    )
    parser.add_argument(
        "--figures-dir",
        default="output/figures/continuous/",
        help="图表输出目录"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="初始资金"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

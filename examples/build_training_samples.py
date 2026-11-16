"""构造 Critic 训练样本

从历史K线数据构造训练样本，包括：
- 主标的K线窗口
- 关联标的K线窗口
- 未来收益率标签
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import importlib.util

def load_config(config_path):
    """动态加载配置文件"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.CORRELATED_UNIVERSE

def load_kline(data_dir, code):
    """加载K线数据"""
    csv_path = Path(data_dir) / f"{code}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    return df

def calc_future_return(df, idx, target_days):
    """计算未来收益率"""
    if idx + target_days >= len(df):
        return None

    current_close = df.loc[idx, 'close']
    future_close = df.loc[idx + target_days, 'close']
    return (future_close - current_close) / current_close

def extract_window(df, idx, window):
    """提取K线窗口特征"""
    if idx < window:
        return None

    window_df = df.iloc[idx - window + 1:idx + 1]

    # 提取 OHLCV
    features = {
        'open': window_df['open'].values,
        'high': window_df['high'].values,
        'low': window_df['low'].values,
        'close': window_df['close'].values,
        'volume': window_df['volume'].values,
    }

    return features

def build_samples(config, data_dir, window, target_days):
    """构造训练样本"""
    samples = []

    for main_code, related_codes in config.items():
        print(f"处理主标的: {main_code}")

        # 加载主标的数据
        main_df = load_kline(data_dir, main_code)
        if main_df is None:
            print(f"  WARN: 主标的数据不存在")
            continue

        # 加载关联标的数据
        related_dfs = {}
        for code in related_codes:
            df = load_kline(data_dir, code)
            if df is not None:
                related_dfs[code] = df

        if len(related_dfs) == 0:
            print(f"  WARN: 无可用关联标的")
            continue

        # 遍历主标的每个交易日
        valid_count = 0
        for idx in range(window, len(main_df) - target_days):
            # 提取主标的窗口
            main_window = extract_window(main_df, idx, window)
            if main_window is None:
                continue

            # 计算未来收益率
            future_return = calc_future_return(main_df, idx, target_days)
            if future_return is None:
                continue

            # 提取关联标的窗口（对齐日期）
            main_date = main_df.loc[idx, 'timestamps']
            related_windows = {}

            for code, df in related_dfs.items():
                # 找到最接近的日期
                date_diff = (df['timestamps'] - main_date).abs()
                nearest_idx = date_diff.idxmin()

                # 检查日期差异不超过3天
                if date_diff[nearest_idx].days > 3:
                    continue

                window_data = extract_window(df, nearest_idx, window)
                if window_data is not None:
                    related_windows[code] = window_data

            # 至少需要2个关联标的
            if len(related_windows) < 2:
                continue

            # 构造样本
            sample = {
                'main_code': main_code,
                'main_window': main_window,
                'related_windows': related_windows,
                'future_return': future_return,
                'date': main_date,
            }

            samples.append(sample)
            valid_count += 1

        print(f"  生成 {valid_count} 个样本")

    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data-dir', type=str, required=True, help='K线数据目录')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--window', type=int, default=20, help='K线窗口长度')
    parser.add_argument('--target-days', type=int, default=5, help='目标预测天数')
    args = parser.parse_args()

    print("=" * 60)
    print("构造 Critic 训练样本")
    print("=" * 60)

    # 加载配置
    print(f"\n加载配置: {args.config}")
    config = load_config(args.config)
    print(f"  产业链数量: {len(config)}")

    # 构造样本
    print(f"\n开始构造样本...")
    print(f"  K线窗口: {args.window}")
    print(f"  预测天数: {args.target_days}")

    samples = build_samples(config, args.data_dir, args.window, args.target_days)

    print(f"\n样本构造完成:")
    print(f"  总样本数: {len(samples)}")

    # 保存样本
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"  已保存到: {output_path}")

    # 统计信息
    print(f"\n样本统计:")
    print(f"  平均关联标的数: {np.mean([len(s['related_windows']) for s in samples]):.1f}")
    print(f"  收益率范围: [{np.min([s['future_return'] for s in samples]):.4f}, {np.max([s['future_return'] for s in samples]):.4f}]")

if __name__ == '__main__':
    main()

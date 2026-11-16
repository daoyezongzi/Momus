"""生成候选路径缓存（Kronos inference）供 Critic 评估使用。

用法：
    # 生成 new_pool_86 在 2021 年的候选缓存
    python examples/generate_candidate_cache.py \\
        --data-dir data/market_2021_hs300 \\
        --codes-file data/references/new_pool_86.txt \\
        --output output/candidates/new_pool_2021_N16.pkl \\
        --sample-count 16 \\
        --lookback 90 \\
        --pred-len 10 \\
        --device cuda

    # 批量生成（2021/2025/2026H1）
    for period in 2021 2025 2026H1; do
        python examples/generate_candidate_cache.py \\
            --data-dir data/market_${period}_hs300 \\
            --codes-file data/references/new_pool_86.txt \\
            --output output/candidates/new_pool_${period}_N16.pkl \\
            --sample-count 16 --device cuda
    done

输出格式：
    {
        'windows': List[Tuple[code, x_df, x_ts, y_ts, y_true]],
        'candidates': List[np.ndarray],  # shape (N, pred_len, 6)
        'metadata': {
            'sample_count': int,
            'lookback': int,
            'pred_len': int,
            'data_dir': str,
            'codes_file': str,
            'total_windows': int,
        }
    }
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append(".")


def _load_dotenv(path=".env"):
    """加载 .env 文件"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def load_kline(data_dir, code):
    """从 CSV 加载 K 线数据"""
    csv_path = Path(data_dir) / f"{code}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if 'timestamps' not in df.columns:
        return None

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    required_cols = ["open", "high", "low", "close", "volume", "amount"]
    if not all(c in df.columns for c in required_cols):
        return None

    return df


def make_walk_forward_windows(df, lookback, pred_len, max_windows=None):
    """Walk-forward 切窗：从 lookback 开始，每次前进 pred_len 天"""
    windows = []
    n_rows = len(df)
    start_idx = lookback

    while start_idx + pred_len <= n_rows:
        x_df = df.iloc[start_idx - lookback:start_idx][
            ["open", "high", "low", "close", "volume", "amount"]].copy()
        x_ts = df.iloc[start_idx - lookback:start_idx]["timestamps"]
        y_ts = df.iloc[start_idx:start_idx + pred_len]["timestamps"]
        y_true = df.iloc[start_idx:start_idx + pred_len][
            ["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)

        windows.append({
            "x_df": x_df,
            "x_ts": pd.Series(x_ts.values),
            "y_ts": pd.Series(y_ts.values),
            "y_true": y_true,
        })

        start_idx += pred_len

        if max_windows and len(windows) >= max_windows:
            break

    return windows


def generate_candidates_for_window(predictor, x_df, x_ts, y_ts, pred_len, sample_count):
    """为单个窗口生成候选路径（不使用 critic/reranker）"""
    paths, _, _, _ = predictor.generate_paths(
        x_df, x_ts, y_ts, pred_len,
        sample_count=sample_count,
        verbose=False,
        target_code=None,
    )
    return paths


def main():
    parser = argparse.ArgumentParser(description="生成候选路径缓存")

    # 数据源
    parser.add_argument("--data-dir", type=str, required=True,
                       help="K线数据目录（如 data/market_2021_hs300）")
    parser.add_argument("--codes-file", type=str, required=True,
                       help="股票代码列表文件（如 data/references/new_pool_86.txt）")

    # 输出
    parser.add_argument("--output", type=str, required=True,
                       help="输出缓存文件路径（.pkl）")

    # Kronos 参数
    parser.add_argument("--sample-count", type=int, default=16,
                       help="每个窗口生成的候选数量")
    parser.add_argument("--lookback", type=int, default=90,
                       help="历史窗口长度（天）")
    parser.add_argument("--pred-len", type=int, default=10,
                       help="预测长度（天）")
    parser.add_argument("--max-windows", type=int, default=None,
                       help="每只股票最多处理的窗口数（None=全部）")
    parser.add_argument("--device", type=str, default="cpu",
                       help="设备（cpu/cuda）")

    args = parser.parse_args()

    # 加载 Kronos（不需要 critic）
    from model import Kronos, KronosTokenizer
    from critic import CriticKronosPredictor

    print("加载 Kronos 模型...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = model.to(args.device)

    # 创建 predictor（critic=None, reranker=None）
    predictor = CriticKronosPredictor(
        model=model,
        tokenizer=tokenizer,
        critic=None,
        reranker=None,
        device=args.device,
    )

    # 加载股票列表
    print(f"加载股票列表: {args.codes_file}")
    with open(args.codes_file) as f:
        codes = [line.strip() for line in f if line.strip()]
    print(f"  共 {len(codes)} 只股票")

    # 生成缓存
    all_windows = []
    all_candidates = []

    for ci, code in enumerate(codes):
        print(f"[{ci + 1}/{len(codes)}] 处理 {code}...")

        df = load_kline(args.data_dir, code)
        if df is None:
            print(f"  跳过（无数据）")
            continue

        windows = make_walk_forward_windows(
            df, args.lookback, args.pred_len, args.max_windows)

        if len(windows) == 0:
            print(f"  跳过（数据不足）")
            continue

        print(f"  生成 {len(windows)} 个窗口的候选...")

        for wi, w in enumerate(windows):
            candidates = generate_candidates_for_window(
                predictor, w["x_df"], w["x_ts"], w["y_ts"],
                args.pred_len, args.sample_count)

            all_windows.append((code, w["x_df"], w["x_ts"], w["y_ts"], w["y_true"]))
            all_candidates.append(candidates)

            if (wi + 1) % 10 == 0:
                print(f"    进度: {wi + 1}/{len(windows)}")

    # 保存缓存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        "windows": all_windows,
        "candidates": all_candidates,
        "metadata": {
            "sample_count": args.sample_count,
            "lookback": args.lookback,
            "pred_len": args.pred_len,
            "data_dir": args.data_dir,
            "codes_file": args.codes_file,
            "total_windows": len(all_windows),
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\n缓存已保存: {output_path}")
    print(f"  总窗口数: {len(all_windows)}")
    print(f"  候选形状: ({args.sample_count}, {args.pred_len}, 6)")


if __name__ == "__main__":
    main()

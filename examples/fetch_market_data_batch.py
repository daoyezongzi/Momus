"""批量抓取全市场股票数据（基于 Tushare）

目标：
1. 获取全市场股票列表（A股主板、创业板、科创板等）
2. 按市值、流动性等筛选标的池
3. 批量抓取历史K线数据
4. 保存为统一格式供训练使用

用法：
  # 抓取沪深300成分股（大盘股，流动性好）
  python examples/fetch_market_data_batch.py --index hs300 --start 20220101 --end 20231231

  # 抓取中证500成分股
  python examples/fetch_market_data_batch.py --index zz500 --start 20220101 --end 20231231

  # 自定义市值筛选（市值 > 100亿，成交额 > 1亿）
  python examples/fetch_market_data_batch.py --min-market-cap 10000000000 --min-volume 100000000

  # 组合：沪深300 + 产业链扩展
  python examples/fetch_market_data_batch.py --index hs300 --include-related --start 20220101 --end 20231231
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append("../")
sys.path.append(".")

from critic.data_provider import make_tushare_fetch


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


def get_index_constituents(pro, index_code):
    """获取指数成分股列表

    常用指数代码：
    - 000300.SH: 沪深300
    - 000905.SH: 中证500
    - 000852.SH: 中证1000
    - 399006.SZ: 创业板指
    - 000688.SH: 科创50
    """
    try:
        df = pro.index_weight(index_code=index_code)
        if df is None or len(df) == 0:
            return []
        # 返回成分股代码列表（去重、去除空值）
        codes = df['con_code'].dropna().unique().tolist()
        return codes
    except Exception as e:
        print(f"获取指数成分股失败: {e}")
        return []


def get_stock_basic(pro, list_status='L', exchange=None):
    """获取股票基础信息

    Args:
        list_status: L=上市, D=退市, P=暂停上市
        exchange: SSE=上交所, SZSE=深交所
    """
    try:
        df = pro.stock_basic(exchange=exchange, list_status=list_status,
                             fields='ts_code,symbol,name,area,industry,market,list_date')
        return df
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()


def filter_by_market_metrics(pro, codes, min_market_cap=None, min_volume=None, trade_date=None):
    """按市值和成交额筛选股票

    Args:
        min_market_cap: 最小市值（元）
        min_volume: 最小日均成交额（元，最近20日）
        trade_date: 查询日期（格式：20231231），默认为最新交易日
    """
    if not min_market_cap and not min_volume:
        return codes

    filtered = []
    for code in codes:
        try:
            # 获取最新市值
            if min_market_cap:
                df = pro.daily_basic(ts_code=code, trade_date=trade_date,
                                     fields='ts_code,trade_date,total_mv')
                if df is None or len(df) == 0:
                    continue
                market_cap = df.iloc[0]['total_mv'] * 10000  # 万元转元
                if market_cap < min_market_cap:
                    continue

            # 获取最近20日日均成交额
            if min_volume:
                df = pro.daily(ts_code=code, start_date=None, end_date=trade_date)
                if df is None or len(df) < 20:
                    continue
                avg_amount = df.head(20)['amount'].mean() * 1000  # 千元转元
                if avg_amount < min_volume:
                    continue

            filtered.append(code)
            time.sleep(0.1)  # 限流控制

        except Exception as e:
            print(f"  筛选 {code} 时出错: {e}")
            continue

    return filtered


def batch_fetch_and_save(pro, codes, start_date, end_date, output_dir, delay=0.2):
    """批量抓取K线数据并保存

    Args:
        codes: 股票代码列表
        start_date: 开始日期（格式：20220101）
        end_date: 结束日期（格式：20231231）
        output_dir: 输出目录
        delay: 请求间隔（秒）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetch_fn = make_tushare_fetch(os.environ.get("TUSHARE_TOKEN"))

    success_count = 0
    fail_count = 0

    for i, ts_code in enumerate(codes):
        print(f"[{i+1}/{len(codes)}] 抓取 {ts_code}...", end=" ")

        # 提取纯代码（去除 .SH/.SZ 后缀）
        code = ts_code.split('.')[0]

        try:
            df = fetch_fn(code, start_date, end_date)

            if df is None or len(df) == 0:
                print("FAIL 无数据")
                fail_count += 1
                continue

            # 保存为 CSV
            output_file = output_path / f"{code}.csv"
            df_with_timestamps = df.reset_index()
            df_with_timestamps.rename(columns={'date': 'timestamps'}, inplace=True)
            df_with_timestamps.to_csv(output_file, index=False)

            print(f"OK {len(df)} 条记录 -> {output_file.name}")
            success_count += 1

            time.sleep(delay)

        except Exception as e:
            print(f"FAIL 错误: {e}")
            fail_count += 1

    print(f"\n抓取完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def extend_with_related_stocks(pro, base_codes):
    """基于产业链配置扩展相关标的

    从 critic/config.py 的 CORRELATED_UNIVERSE 中提取所有关联标的
    """
    from critic.config import CORRELATED_UNIVERSE

    extended = set(base_codes)

    for target, related in CORRELATED_UNIVERSE.items():
        # 统一格式：去除 .SH/.SZ 后缀
        target_clean = target.split('.')[0] if '.' in target else target

        if any(target_clean in code for code in base_codes):
            # 添加关联标的
            for rel in related:
                rel_clean = rel.split('.')[0] if '.' in rel else rel
                extended.add(rel_clean)

    # 转换回 tushare 格式（添加后缀）
    result = []
    for code in extended:
        if code.startswith(('0', '2', '3')):
            result.append(f"{code}.SZ")
        else:
            result.append(f"{code}.SH")

    return result


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="批量抓取全市场股票数据")

    # 数据源
    parser.add_argument("--token", type=str, default=os.environ.get("TUSHARE_TOKEN"),
                       help="Tushare API token（默认从环境变量读取）")

    # 股票池选择（互斥）
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=str, choices=['hs300', 'zz500', 'zz1000', 'cyb', 'kc50'],
                       help="指数成分股: hs300=沪深300, zz500=中证500, zz1000=中证1000, cyb=创业板指, kc50=科创50")
    group.add_argument("--all-stocks", action="store_true",
                       help="全市场A股（需后续按市值/流动性筛选，否则数量过大）")
    group.add_argument("--codes-file", type=str,
                       help="从文件读取股票代码列表（每行一个代码）")

    # 筛选条件
    parser.add_argument("--min-market-cap", type=float, default=None,
                       help="最小市值（亿元）")
    parser.add_argument("--min-volume", type=float, default=None,
                       help="最小日均成交额（亿元，最近20日）")
    parser.add_argument("--include-related", action="store_true",
                       help="包含产业链关联标的（从 config.py 扩展）")

    # 时间范围
    parser.add_argument("--start", type=str, default="20220101",
                       help="开始日期（格式：20220101）")
    parser.add_argument("--end", type=str, default="20231231",
                       help="结束日期（格式：20231231）")

    # 输出
    parser.add_argument("--output-dir", type=str, default="./market_data",
                       help="输出目录")
    parser.add_argument("--codes-output", type=str, default="./market_codes.txt",
                       help="输出股票代码列表文件")
    parser.add_argument("--delay", type=float, default=0.2,
                       help="请求间隔（秒）")

    args = parser.parse_args()

    if not args.token:
        print("错误: 未提供 Tushare token。请设置环境变量 TUSHARE_TOKEN 或使用 --token 参数")
        return

    # 初始化 Tushare
    import tushare as ts
    pro = ts.pro_api(args.token)

    print("=" * 60)
    print("批量抓取市场数据")
    print("=" * 60)

    # 获取股票列表
    codes = []

    if args.index:
        index_map = {
            'hs300': '000300.SH',
            'zz500': '000905.SH',
            'zz1000': '000852.SH',
            'cyb': '399006.SZ',
            'kc50': '000688.SH'
        }
        index_code = index_map[args.index]
        print(f"\n获取 {args.index.upper()} 成分股...")
        codes = get_index_constituents(pro, index_code)
        print(f"  获得 {len(codes)} 只股票")

    elif args.all_stocks:
        print("\n获取全市场A股列表...")
        df = get_stock_basic(pro, list_status='L')
        codes = df['ts_code'].tolist()
        print(f"  获得 {len(codes)} 只股票")

        if not args.min_market_cap and not args.min_volume:
            print("\n⚠️  警告: 全市场股票数量过大（>5000只），建议使用 --min-market-cap 或 --min-volume 筛选")
            response = input("是否继续？(y/N): ")
            if response.lower() != 'y':
                return

    elif args.codes_file:
        print(f"\n从 {args.codes_file} 读取股票列表...")
        if os.path.exists(args.codes_file):
            with open(args.codes_file) as f:
                codes = [line.strip() for line in f if line.strip()]
            # 转换为 tushare 格式
            ts_codes = []
            for code in codes:
                if '.' not in code:
                    if code.startswith(('0', '2', '3')):
                        ts_codes.append(f"{code}.SZ")
                    else:
                        ts_codes.append(f"{code}.SH")
                else:
                    ts_codes.append(code)
            codes = ts_codes
            print(f"  读取 {len(codes)} 只股票")
        else:
            print(f"错误: 文件不存在 {args.codes_file}")
            return
    else:
        print("\n使用 data/references/codes.txt 中的股票列表...")
        if os.path.exists("data/references/codes.txt"):
            with open("data/references/codes.txt") as f:
                codes = [line.strip() for line in f if line.strip()]
            # 转换为 tushare 格式
            ts_codes = []
            for code in codes:
                if '.' not in code:
                    if code.startswith(('0', '2', '3')):
                        ts_codes.append(f"{code}.SZ")
                    else:
                        ts_codes.append(f"{code}.SH")
                else:
                    ts_codes.append(code)
            codes = ts_codes
            print(f"  读取 {len(codes)} 只股票")
        else:
            print("错误: 未指定股票池且 data/references/codes.txt 不存在")
            return

    # 按市值/流动性筛选
    if args.min_market_cap or args.min_volume:
        print("\n按市值/流动性筛选...")
        min_cap = args.min_market_cap * 100000000 if args.min_market_cap else None
        min_vol = args.min_volume * 100000000 if args.min_volume else None
        codes = filter_by_market_metrics(pro, codes, min_cap, min_vol)
        print(f"  筛选后剩余 {len(codes)} 只股票")

    # 扩展产业链关联标的
    if args.include_related:
        print("\n扩展产业链关联标的...")
        original_count = len(codes)
        codes = extend_with_related_stocks(pro, codes)
        print(f"  扩展后共 {len(codes)} 只股票（新增 {len(codes) - original_count} 只）")

    if not codes:
        print("错误: 未获取到任何股票代码")
        return

    # 保存股票代码列表
    with open(args.codes_output, 'w') as f:
        for code in codes:
            # 保存为纯代码（去除后缀）
            clean_code = code.split('.')[0]
            f.write(f"{clean_code}\n")
    print(f"\n股票代码列表已保存至: {args.codes_output}")

    # 批量抓取数据
    print(f"\n开始批量抓取数据（{args.start} ~ {args.end}）...")
    success, fail = batch_fetch_and_save(pro, codes, args.start, args.end,
                                          args.output_dir, args.delay)

    print("\n" + "=" * 60)
    print(f"抓取完成:")
    print(f"  成功: {success} 只")
    print(f"  失败: {fail} 只")
    print(f"  输出目录: {args.output_dir}")
    print(f"  代码列表: {args.codes_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

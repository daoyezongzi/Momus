"""按产业链逻辑扩展训练集

目标：
1. 保持产业链上下游结构
2. 扩展现有6条产业链的标的数量
3. 新增更多产业链（消费电子、房地产、钢铁有色等）
4. 利用 Tushare 的行业分类和概念板块自动发现关联

策略：
- 从现有产业链核心标的出发
- 利用 Tushare 的 concept、industry 字段找同产业链标的
- 手工验证上下游关系（避免噪声）

用法：
  python examples/expand_industry_chains.py --base-config critic/config.py --output expanded_chains.py
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.append("../")
sys.path.append(".")

# 产业链扩展模板（核心逻辑 + 可扩展标的池）
INDUSTRY_CHAIN_EXPANSION = {
    # ===== 现有6条产业链扩展 =====
    "新能源车/锂电池": {
        "core": ["300750", "002594"],  # 宁德时代、比亚迪
        "concepts": ["新能源汽车", "锂电池", "动力电池"],
        "extend_pool": [
            # 上游：锂矿/正负极材料
            "002466", "002460", "603799", "002074", "300618",  # 天齐、赣锋、华友、国轩、寒锐
            # 中游：电池
            "300014", "002812", "300438", "688005",  # 亿纬、恩捷、鹏辉、容百
            # 下游：整车/零部件
            "601127", "601633", "600104", "002920",  # 赛力斯、长城、上汽、德赛
        ],
    },
    "白酒": {
        "core": ["600519", "000858"],  # 茅台、五粮液
        "concepts": ["白酒"],
        "extend_pool": [
            "000568", "600809", "002304", "000799", "600702", "603369",
            # 泸州、汾酒、洋河、酒鬼、舍得、今世缘
        ],
    },
    "半导体": {
        "core": ["688981", "002371"],  # 中芯国际、北方华创
        "concepts": ["半导体", "芯片", "集成电路"],
        "extend_pool": [
            "688012", "603986", "002049", "688126", "688396", "688008",
            # 中微、兆易、紫光、沪硅、华润微、澜起
            "688223", "688778", "300782",  # 晶科、中芯集成、卓胜微
        ],
    },
    "光伏": {
        "core": ["601012", "002129"],  # 隆基、TCL中环
        "concepts": ["光伏", "太阳能"],
        "extend_pool": [
            "300274", "688599", "600438", "002459", "601865", "688032",
            # 阳光、天合、通威、晶澳、福莱特、禾迈
        ],
    },
    "医药/CXO": {
        "core": ["600276", "603259"],  # 恒瑞、药明康德
        "concepts": ["医药", "CRO", "医疗器械"],
        "extend_pool": [
            "300760", "300347", "002821", "300595", "603127", "688185",
            # 迈瑞、泰格、凯莱英、欧普康视、昭衍、康希诺
        ],
    },
    "银行/金融": {
        "core": ["600036", "601318"],  # 招商银行、中国平安
        "concepts": ["银行", "保险", "券商"],
        "extend_pool": [
            "601166", "600000", "600016", "601288", "601601", "601688",
            # 兴业、浦发、民生、农行、太保、华泰
        ],
    },

    # ===== 新增产业链 =====
    "消费电子": {
        "core": ["002415", "300433"],  # 海康威视、蓝思科技
        "concepts": ["消费电子", "智能手机", "电子元器件"],
        "extend_pool": [
            "002475", "002241", "300136", "002008", "300719",
            # 立讯、歌尔、信维、大族、安集
        ],
    },
    "5G通信": {
        "core": ["600050", "000063"],  # 中国联通、中兴通讯
        "concepts": ["5G", "通信设备", "光通信"],
        "extend_pool": [
            "600522", "002583", "300308", "002396", "600487",
            # 中天、海能达、中际旭创、星网锐捷、亨通光电
        ],
    },
    "房地产/建材": {
        "core": ["000002", "600048"],  # 万科、保利地产
        "concepts": ["房地产", "建材", "水泥"],
        "extend_pool": [
            "600340", "600585", "603816", "002271", "603806",
            # 华夏幸福、海螺、顾家家居、东方雨虹、福斯特
        ],
    },
    "钢铁/有色": {
        "core": ["600019", "601600"],  # 宝钢、中国铝业
        "concepts": ["钢铁", "有色金属", "铜"],
        "extend_pool": [
            "000629", "600111", "002110", "601899", "000878",
            # 攀钢、北方稀土、三钢、紫金矿业、云铜
        ],
    },
}


def load_dotenv(path=".env"):
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


def get_stock_industry_info(pro, ts_code):
    """获取股票的行业和概念信息"""
    try:
        # 基础信息
        basic = pro.stock_basic(ts_code=ts_code, fields='ts_code,name,industry')
        if basic is None or len(basic) == 0:
            return None

        # 概念板块
        concepts = pro.concept_detail(ts_code=ts_code, fields='concept_name')
        concept_list = concepts['concept_name'].tolist() if concepts is not None and len(concepts) > 0 else []

        return {
            'ts_code': ts_code,
            'name': basic.iloc[0]['name'],
            'industry': basic.iloc[0]['industry'],
            'concepts': concept_list
        }
    except Exception as e:
        print(f"  获取 {ts_code} 信息失败: {e}")
        return None


def validate_chain_relationship(pro, target, related):
    """验证两只股票的产业链关系（基于概念/行业重叠度）"""
    import time

    target_info = get_stock_industry_info(pro, target)
    time.sleep(0.1)
    related_info = get_stock_industry_info(pro, related)
    time.sleep(0.1)

    if not target_info or not related_info:
        return False

    # 同行业
    if target_info['industry'] == related_info['industry']:
        return True

    # 概念重叠
    target_concepts = set(target_info['concepts'])
    related_concepts = set(related_info['concepts'])
    overlap = target_concepts & related_concepts

    return len(overlap) >= 2  # 至少2个共同概念


def expand_industry_chains(pro, validate=True):
    """扩展产业链配置"""
    expanded_universe = {}

    for chain_name, chain_config in INDUSTRY_CHAIN_EXPANSION.items():
        print(f"\n处理产业链: {chain_name}")
        core_stocks = chain_config["core"]
        extend_pool = chain_config["extend_pool"]

        # 为每个核心标的配置关联标的池
        for target in core_stocks + extend_pool:
            if target in expanded_universe:
                continue  # 已处理

            # 从扩展池中选择关联标的
            related = []
            for candidate in extend_pool:
                if candidate == target:
                    continue

                # 可选：验证关联关系（慢，但质量高）
                if validate:
                    target_ts = f"{target}.{'SZ' if target.startswith(('0','2','3')) else 'SH'}"
                    candidate_ts = f"{candidate}.{'SZ' if candidate.startswith(('0','2','3')) else 'SH'}"

                    if validate_chain_relationship(pro, target_ts, candidate_ts):
                        related.append(candidate)
                else:
                    related.append(candidate)

                # 限制关联标的数量（避免噪声）
                if len(related) >= 5:
                    break

            if related:
                expanded_universe[target] = related
                print(f"  {target}: {len(related)} 个关联标的")

    return expanded_universe


def generate_config_file(universe, output_path):
    """生成新的 config.py 文件"""
    lines = [
        '"""扩展后的产业链关联配置',
        '',
        '自动生成，基于 expand_industry_chains.py',
        '"""',
        '',
        'CORRELATED_UNIVERSE = {',
    ]

    # 按产业链分组输出
    for chain_name, chain_config in INDUSTRY_CHAIN_EXPANSION.items():
        lines.append(f'    # ===== {chain_name} =====')
        all_stocks = chain_config["core"] + chain_config["extend_pool"]

        for stock in all_stocks:
            if stock in universe:
                related = universe[stock]
                related_str = ', '.join([f'"{r}"' for r in related])
                lines.append(f'    "{stock}": [{related_str}],')

        lines.append('')

    lines.append('}')
    lines.append('')
    lines.append('MAX_CORRELATED = 5')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n已生成配置文件: {output_path}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="扩展产业链配置")
    parser.add_argument("--token", type=str, default=os.environ.get("TUSHARE_TOKEN"))
    parser.add_argument("--output", type=str, default="critic/config_expanded.py",
                       help="输出配置文件路径")
    parser.add_argument("--codes-output", type=str, default="codes_expanded.txt",
                       help="输出股票代码列表")
    parser.add_argument("--validate", action="store_true",
                       help="验证产业链关系（慢但质量高）")
    parser.add_argument("--skip-validation", dest="validate", action="store_false",
                       help="跳过验证，直接使用扩展池（快）")
    parser.set_defaults(validate=False)

    args = parser.parse_args()

    if not args.token:
        print("错误: 未提供 Tushare token")
        return

    import tushare as ts
    pro = ts.pro_api(args.token)

    print("=" * 60)
    print("产业链扩展工具")
    print("=" * 60)

    # 扩展产业链
    expanded_universe = expand_industry_chains(pro, validate=args.validate)

    # 统计
    total_stocks = len(expanded_universe)
    total_relationships = sum(len(v) for v in expanded_universe.values())

    print(f"\n扩展完成:")
    print(f"  标的数量: {total_stocks} 只（原31只）")
    print(f"  关联关系: {total_relationships} 对")
    print(f"  平均每只关联: {total_relationships / total_stocks:.1f} 只")

    # 生成配置文件
    generate_config_file(expanded_universe, args.output)

    # 生成代码列表
    with open(args.codes_output, 'w') as f:
        for code in sorted(expanded_universe.keys()):
            f.write(f"{code}\n")
    print(f"已生成代码列表: {args.codes_output}")

    # 按产业链统计
    print("\n各产业链标的数量:")
    for chain_name, chain_config in INDUSTRY_CHAIN_EXPANSION.items():
        all_stocks = chain_config["core"] + chain_config["extend_pool"]
        count = sum(1 for s in all_stocks if s in expanded_universe)
        print(f"  {chain_name}: {count} 只")


if __name__ == "__main__":
    main()

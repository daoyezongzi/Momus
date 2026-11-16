"""检查样本缓存文件的结构"""
import pickle
import sys

def check_sample_structure(pkl_file):
    try:
        with open(pkl_file, 'rb') as f:
            samples = pickle.load(f)

        print(f"文件: {pkl_file}")
        print(f"样本总数: {len(samples)}")

        if len(samples) > 0:
            sample = samples[0]
            print(f"\n第一个样本的类型: {type(sample)}")

            if hasattr(sample, '__dict__'):
                print(f"样本属性: {list(sample.__dict__.keys())}")
                for key, val in sample.__dict__.items():
                    if hasattr(val, 'shape'):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  {key}: {type(val)}")
            elif isinstance(sample, dict):
                print(f"样本键: {list(sample.keys())}")
                for key, val in sample.items():
                    if hasattr(val, 'shape'):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  {key}: {type(val)}")

            # 尝试查找股票代码
            print("\n尝试查找股票代码字段...")
            if hasattr(sample, '__dict__'):
                for key in ['code', 'stock_code', 'symbol', 'ts_code', 'meta', 'metadata']:
                    if key in sample.__dict__:
                        print(f"  找到字段 '{key}': {sample.__dict__[key]}")
            elif isinstance(sample, dict):
                for key in ['code', 'stock_code', 'symbol', 'ts_code', 'meta', 'metadata']:
                    if key in sample:
                        print(f"  找到字段 '{key}': {sample[key]}")

    except FileNotFoundError:
        print(f"文件不存在: {pkl_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    files = ["critic_samples_full.pkl", "critic_samples_2024.pkl"]
    for f in files:
        print("=" * 60)
        check_sample_structure(f)
        print()

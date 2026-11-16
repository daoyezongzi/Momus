# Momus

Momus 基于 Kronos 生成金融时间序列的候选路径，使用 Critic 结合目标标的与关联标的的历史信息进行评分和重排序。代码包含训练、预测、回测、统计分析与 Web 界面。

## 安装

建议使用 Python 3.11。

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

Linux 或 macOS 激活环境：

```bash
source .venv/bin/activate
```

安装核心依赖与测试工具：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-research.txt
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `model/` | Kronos、KronosTokenizer 和 KronosPredictor |
| `critic/` | Critic 模型、候选路径评分、数据检查与回测 |
| `examples/` | 数据准备、训练、预测和实验命令 |
| `finetune/` | Qlib 数据处理与模型训练 |
| `finetune_csv/` | CSV 数据训练与配置模板 |
| `scripts/` | 研究图表生成代码 |
| `tests/` | 模型与研究流程测试 |
| `webui/` | Flask 预测界面 |

## 数据与模型

运行训练和回测需要自行准备行情数据及相应模型权重。不同命令的输入路径和参数可通过 `--help` 查看：

```bash
python examples/train_critic_pipeline.py --help
python examples/generate_candidate_cache.py --help
python examples/run_e2_backtest.py --help
```

模型代码使用 `NeoQuasar/Kronos-Tokenizer-base` 和 `NeoQuasar/Kronos-small` 等 Hugging Face 模型标识符。首次加载需要下载对应权重，具体配置以所运行命令为准。

需要 Tushare 时，将 `.env.example` 复制为 `.env` 并填写个人 `TUSHARE_TOKEN`。仅自动读取 `.env` 的命令会加载该文件，其余命令需要设置环境变量或传入参数。

`scripts/` 中的研究图表命令需要对应实验结果。图表输出使用项目内的 `output/` 目录；涉及稿件的命令从 `output/manuscript/` 读取输入文件。

## 验证

以下测试检查数据时间限制、样本划分、评价指标和统计函数：

```bash
python -m pytest tests/test_schemas.py tests/test_splits.py tests/test_leakage.py tests/test_statistics.py tests/test_evaluation.py -q
```

模型集成测试需要下载模型权重，部分研究流程还需要额外的数据与依赖。Qlib 相关命令需要安装 `pyqlib`；Comet 记录功能需要安装 `comet_ml` 并配置个人账户。

## Web 界面

```bash
python -m pip install -r webui/requirements.txt
python webui/run.py
```

## 许可证

采用 MIT License。Kronos 原始版权声明保留于 `LICENSE`。

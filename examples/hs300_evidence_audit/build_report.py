"""Build the requested audit deliverable from checked artifacts; never edit paper."""
import json
from pathlib import Path

import pandas as pd

from audit import ROOT,OLD,OUT,save,dump,verify_protection


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])


def main():
    protect=verify_protection()
    full=json.loads((OUT/'full_replay_status.json').read_text())
    ci=pd.read_csv(OUT/'paired_intervals.csv')
    primary=ci[(ci.method=='momus_weighted')&(ci.baseline=='simple_mean')&(ci.metric=='mse_improvement_pct')&(ci.resampling=='stock')]
    coverage=pd.read_csv(OUT/'coverage.csv')
    groups=pd.read_csv(OUT/'forecast_subgroups.csv')
    inv=pd.read_csv(OUT/'investment_correction_summary.csv')
    inv=inv[inv.stage=='benchmark_and_initial_nav_fix']
    ranks=pd.read_csv(OUT/'candidate_ranking_per_seed.csv').groupby('period').mean(numeric_only=True)
    ledger=pd.read_csv(OUT/'investment_ledger_checks.csv')
    gaps=pd.read_csv(OUT/'large_gap_sensitivity.csv')
    claims=[
      ('C01','三个已评价时期的加权OHLC误差下降','支持（限当前固定输入和股票池）','全量PyTorch复算；配对区间；main_table.csv','三期均观察到MSE/MAE下降；披露固定候选与回溯样本边界'),
      ('C02','新样本完全未参与训练','不支持','每期65只代码与3380窗口训练缓存重叠','注明65只训练已见与286/287只未见；未见分组仍有误差改善'),
      ('C03','严格逐时点沪深300成分、无生存偏差','未确认','六份年份名单哈希相同；index_weight未指定日期后去重','沪深300固定股票池；成员来源日期与PIT资格未确认'),
      ('C04','三期均严格晚于Kronos预训练截止日','不支持','2024有1052窗口entry早于2024-07-01','2024作回溯扩展评价；另报7月后4199窗口结果'),
      ('C05','误差改善伴随一致的Price IC/RankIC改善','不支持','2025相关性下降；2026H1增量区间跨零','区分数值误差与路径相关性，不把路径IC称为股票截面IC'),
      ('C06','候选评分含质量排序信息','支持（描述性）','HS300全窗口候选评分与负真实MSE的Spearman约0.44至0.46','评分与候选误差有正向排序关联；非语义结构消融证明'),
      ('C07','双粒度结构在HS300上显著优于同容量单头','未确认','此次无HS300同容量/语义消融；旧样本不能替代','旧机制证据注明样本与设置，不外推结构优势'),
      ('C08','top-1优于weighted证明权重校准是瓶颈','证据有限','2024及2026H1 top-1更好，2025接近；无校准干预','作为待检验解释，不写成已定位的因果机制'),
      ('C09','候选数量实验证明计算效率提升','不支持','无运行耗时/计算量配对测量','只陈述固定候选前缀下的精度与预算关系'),
      ('C10','投资表现具有稳定优势','当前不可判断真实可交易优势','两项核算错误；公司行为未计；异步局部截面','现有简化账本未提供稳定增益证据；不据此判断真实盈利或不可盈利'),
      ('C11','Kronos原论文完整复现','不支持','统计轴部分一致；数据、采样、调仓、费用及年化因子不同','对齐任务与若干指标；逐项披露实现差异'),
      ('C12','2026H1支持跨市场状态稳定性','证据有限','352股票但仅2个主要origin批次；10/20日期块数2/1','股票聚类区间仅条件于已观察日期；时间区间不报告'),
    ]
    save('claim_evidence.csv',[dict(id=i,claim=c,status=s,evidence=e,allowed_statement=a) for i,c,s,e,a in claims])
    ct=table(['时期','有效股票','窗口','训练重叠/未见','origin日期','≥300只信号日期','实际预测覆盖'],[
      [r.period,r.stocks,r.windows,f'{r.training_overlap_stocks}/{r.unseen_stocks}',r.origin_dates,r.origins_with_at_least_300_signals,f'{r.first_entry} 至 {r.last_horizon}'] for r in coverage.itertuples()])
    pt=table(['时期','加权MSE改善','股票聚类95%区间','三期主比较校正区间'],[
      [r.period,f'{r.effect:.2f}%',f'[{r.ci95_low:.2f}%, {r.ci95_high:.2f}%]',f'[{r.three_period_bonferroni_low:.2f}%, {r.three_period_bonferroni_high:.2f}%]'] for r in primary.itertuples()])
    unseen=groups[(groups.method=='momus_weighted')&(groups.baseline=='simple_mean')&(groups.slice=='training_unseen')]
    ut=table(['时期','训练未见股票','窗口','MSE改善'],[[r.period,r.stocks,r.windows,f'{r.mse_improvement_pct:.2f}%'] for r in unseen.itertuples()])
    it=table(['时期','方法','实际累计收益','同期指数收益','AER','IR','收益观察日'],[
      [r.period,r.method,f'{r.total_return_net_mean*100:.2f}%',f'{r.benchmark_total_mean*100:.2f}%',f'{r.aer_mean*100:.2f}%',f'{r.ir_mean:.3f}',int(r.valid_days_mean)] for r in inv.itertuples()])
    rt=table(['时期','候选排序Spearman','Top-1命中事后最优候选'],[[p,f'{r.candidate_rank_spearman:.4f}',f'{r.top1_oracle_hit*100:.2f}%'] for p,r in ranks.iterrows()])
    gt=table(['时期','诊断性保留窗口','排除窗口','MSE改善'],[[r.period,r.windows,r.excluded,f'{r.mse_improvement_pct:.2f}%'] for r in gaps[(gaps.method=='momus_weighted')&(gaps.slice=='exclude_history_or_future_gap_gt_20pct')].itertuples()])
    claimtable=table(['编号','主张','审计状态','可使用表述'],[[i,c,s,a] for i,c,s,e,a in claims])
    report=f'''# 沪深300主线证据审计与补充统计

审计日期：2026-09-08。审计对象：现有HS300固定候选实验。阶段：证据审计完成，论文未修改；预测主表通过计算核验，投资表仍有进入正式论文前的必要修复项。

## 1. 可以保留的主线与当前结论

继续以沪深300实验为论文主线。新增样本上的OHLC误差改善通过全量独立推理与统计核查，三期均为正。应限定为当前固定股票池、未复权价格目标、固定候选和既有checkpoint下的回溯结果，不能把它写成完整历史成分实验或实时预测验证。

论文、原预测与投资结果、市场CSV、冻结协议、模型源码、模型权重和候选缓存均未覆盖。保护清单中{protect['checked_files']}个文件的前后SHA256一致。原工作区已有修改保留。

投资旧表不能原样进入论文。已输出两项明确核算修正；修正后仍保留未计公司行为和异步局部信号的假设，因此它是审计对照，不是最终有效的投资回测。不能把简化账本的负结果升级成“模型没有投资价值”的结论。

## 2. 样本与时间证据

{ct}

- 六份`hs300_2021`至`hs300_2026H1`名单逐字节相同，共354个唯一代码，SHA256均为`7bbaf931...47abe14d91edc`。抓取函数调用`index_weight(index_code=...)`后对`con_code`去重，未限定trade_date/start_date/end_date。原始成员返回记录没有随名单保存，无法从去重代码反推出当时成分或成员生效日期。文件名中的年份不能当成历史成员资格证据。
- [Tushare成分接口文档](https://tushare.pro/document/2?doc_id=96)明确返回带交易日期的月度成分/权重记录。正文可称“沪深300固定股票池”，并解释构建方式和未确认的PIT条件；不能写“已消除生存偏差”。
- 3380条训练缓存包含86只代码；所有训练历史长度为60，1240个训练窗口具有关联输入。缓存全部缺少样本日期meta，训练年份来自既有实验记录，不能将该记录冒充本次独立重建的逐样本日期证据。
- 新评价历史长度为90，预测长度10，步长10；全部通过`corr_missing`推理。完整模型结构未变，但不能照搬旧正文“主评价31只股票使用关联输入”的描述。
- 同股相邻未来窗口不重叠，历史窗口仍高度重叠；不同股票共享市场日期。351/352只股票不等于351/352次独立市场实验。
- Kronos截止日与任务轴见第7节。2024有1052个预测窗口的起始日期早于2024-07-01；保留三期原主表，另报2024-07-01及以后origin的4199个窗口，其加权MSE改善为6.32%。该切片也是事后补充，不能重新命名为未查看的确认集。

## 3. 预测计算核验及统计

所有原始行情行和候选真值逐元素匹配；三期有效行情均无重复日期、非有限值、非正成交量或OHLC顺序错误。这些是可计算性检查，不代表成员资格、复权和可交易性已通过。

全量PyTorch推理独立于原NumPy推理引擎，覆盖11,193个窗口、五个种子、四种主方法和N=1/4/8/16。重复历史编码仅为执行优化，并与原模型直接forward交叉检查。895,440个逐窗口指标比较的最大绝对误差为{full['max_metric_abs_error']:.3g}；全部投资信号亦复算一致。N=1两种聚合完全相同；无重新训练、候选采样或种子选择。

{pt}

统计量为“先计算同窗口方法差，再对五种子平均”；主效应保持原来的窗口等权口径。以股票为簇进行5000次配对重采样，固定RNG=20260908，每次保留股票全部窗口。MSE改善率在每次重采样中用误差总和之比计算。三期主比较另报告98.333% Bonferroni百分位区间；不是对所有探索比较的整体显著性保证。五次重复等权基线不计为五次独立生成。

股票区间条件于已经观察到的时间路径，不能控制共同市场冲击。因此同时报告按10/20个交易日分块以及股票×时间块的交叉重采样敏感性；2024/2025的10日块各15个、20日块各8个，改善方向仍为正。8个块的推断仅作探索。2026H1只有2/1个时间块，名义区间已主动留空，避免产生退化而过窄的“置信区间”。块按origin归属，仍无法保证跨块的历史/未来完全独立。

所有区间是查看既有结果后的补充分析、条件于固定候选和已选checkpoint；不能追认成原始事前注册统计检验。未把不同年份拼成一个选择性总体，也未按结果调整block长度。

训练未见股票分组仍保留改善，提供有限的股票迁移证据：

{ut}

股票等权差值、逐种子结果、全部比较及IC/RankIC区间见CSV。2025年加权相对等权的IC与RankIC增量分别为-0.014864和-0.015196，股票聚类区间均为负；2026H1对应增量的区间均跨零。因此支持误差下降，不支持跨时期相关性一致改善。

## 4. HS300候选排序及机制边界

{rt}

这里的Spearman在每个窗口的16个候选之间计算“预测评分”与“负真实候选MSE”的相关性，再汇总窗口和种子。它与沿10日预测路径计算的Price RankIC是两个统计轴，必须使用不同名称。真值只在打分完成后用于评价，事后最优候选不是可部署基线。

这些结果支持候选评分存在质量排序信息。Top-1在2024及2026H1的MSE低于weighted，但不能仅凭该比较确定权重校准是原因。新样本上未做同容量单头或语义目标消融；原86只股票的此类证据只能在注明原设置后作为补充，不能说已在HS300重复验证结构优势。

## 5. 投资核算修复、影响与未通过项

### 5.1 已证明并修正的两项核算问题

**基准open/close写反。** 原脚本把腾讯行数据解析为`date,close,open,high,low,volume`。实际字段顺序为`date,open,close,high,low,...`，由[AKShare官方解析源码](https://github.com/akfamily/akshare/blob/main/akshare/index/index_stock_zh.py#L343)及601行原始API响应核对确认。例如2024-01-02的正确收盘为3386.35，原文件将3426.27开盘值命名为close。只在独立目录输出修正CSV；没有重新采样信号或替换原基准文件。

**初始NAV口径漏计首日。** 旧表从首次入场日收盘NAV开始计算收益，首日交易收益和初始买入费用被排除在总收益、年化收益与AER/IR之外。保留原协议输出作为对照，新修正版在首笔交易前的前一交易日收盘设初始现金1亿元，并以该日真实指数收盘为共同基准；第一天策略在开盘入场，指数按同日close-to-close计收益。由此新观察日数为151/153/26，而非旧150/152/25。

`investment_correction_per_seed.csv`分别保留原表复现、只修正基准、基准与首日同时修正三个阶段。所有交易、价格、持仓、信号、费用保持不变。以下是两项修复后的五种子均值，**仅为仍有缺陷的固定交易账本对照，不可作为最终投资结果**：

{it}

AER为252乘日主动收益均值，IR为sqrt(252)乘主动收益均值除以样本标准差；AER不是实际累计超额收益。尤其26个观察日的-111%左右AER是线性年化数值，不是同期损失超过本金。原“gross_equity=equity+累计费用”只表示同一固定交易路径的费用加回，不是真正重新模拟的无成本策略，本报告不把它称作毛收益反事实。

已独立重放69,307条成交记录和30条策略序列，核对开盘成交价、成本、现金、持仓与逐日估值；相对原始舍入CSV的最大NAV差为{ledger.max_nav_error_currency.max():.4f}元（初始1亿元）。最小持有期违规计数为0。舍入后的微小负余额/余股单独记录，不能据此认定原全精度账本透支。

### 5.2 必须解决后才可入稿的实质问题

**公司行为遗漏已确认。** 股票行情调用Tushare `daily`，无复权因子；现有账户没有分红现金或转增股份记录。[Tushare日线说明](https://tushare.pro/document/2?doc_id=27)注明未复权。作为确切例子，英维克002837于2024-05-27每10股转增3股并派2元现金，见[公司权益分派实施公告](https://epaper.cs.com.cn/zgzqb/html/2024-05/18/nw.D110000zgzqb_20240518_2-B093.htm)。回放显示全部10条2024策略序列在该日前持有约62437.56股，而旧账本既未加入转增股份也未入账红利；当日原始开盘相对前收下降约23.70%。这证明问题实际涉及持仓，不只是潜在风险。不能只补这一案例就声称全部公司行为已修复；完整修复需要所有实际持仓对应事件、支付/上市日期与明确税费约定。

**异步窗口形成大量局部截面。** 2024有61个信号日期，其中46日不足50个信号；2025为86/71；2026H1为9/7。三期每天信号数的中位数均为1。旧代码只对当日新信号排名，将不在当日局部top集合的现有持仓纳入移出范围，并可能只补进1只股票。缺少新预测不能解释为预测排名下降。实际持仓数因此经常低于50，目标“Top-50/drop-5”不能等同于完整同期截面的每日Top-50。

缺失信号采用沿用至过期、仅同步批次调仓、或重新生成每日预测，是不同策略规则，不能默默择优。本次只定位并量化问题，没有把策略重设计混入核算纠错。原始全表留存；最终投资重算还需明确一条与固定缓存兼容的信号有效期/调仓规则。

## 6. 未复权跳变对预测结果的敏感性

原始价格目标包含送转等非市场价格连续变化。例如[中际旭创2024年实施公告](https://static.cninfo.com.cn/finalpage/2024-05-30/1220213405.PDF)确认2024-06-06转增；[新易盛2026半年报](https://static.cninfo.com.cn/finalpage/2026-08-25/1225499406.PDF)确认2026-06-11转增。不能将这些机械价格变化简单当作普通市场涨跌。

为检查主结果是否由大幅跳变窗口驱动，事后标记`abs(open_t/close_{{t-1}}-1)>20%`，分别排除未来含标记、历史或未来含标记的窗口复算。完整原主表不变。后一种更严格诊断结果为：

{gt}

改善方向保留；但该阈值也会排除真实市场波动、漏掉小额除息，因此只能排查极端跳变依赖，不能替代复权修复或形成新的确认样本。未改候选、历史、真值或股票池。

## 7. Kronos指标与实验协议来源边界

下表依据原论文、固定commit的官方代码及Qlib当前源码分别核对。源快照与SHA256在`sources/manifest.json`。Qlib当前源码是接口行为证据，不代表已恢复原论文依赖版本。

| 项目 | 已确认的官方来源内容 | 当前HS300实际实现与边界 |
|---|---|---|
| Price IC/RankIC统计轴 | [论文D.3](https://arxiv.org/html/2508.02739v1#A4.SS3)按样本、OHLC通道的预测与真实序列计算，再平均通道 | 10步horizon内Pearson/average-tie Spearman，再平均四通道和窗口；轴相符。精确estimator、无效通道处理及全任务加权规则不能冒称官方函数复现 |
| Return IC/RankIC | 同一论文在资产类别/频率内对终点收益向量计算 | 当前Price指标不属于此轴，更不等同于逐日期股票截面收益RankIC |
| 投资signal | 论文公式13使用预测收盘均值相对当前收盘的收益率 | 当前signal与论文公式相符；[官方demo](https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune/qlib_test.py#L274)的mean是绝对价差，与论文公式和本项目存在差异 |
| 窗口与交易 | 论文投资90/10、日频决策，CSI300 top50/drop5、至少持有5日 | 本项目90/10、每股stride10，异步origin局部截面；非日频预测复现 |
| 费用 | 论文写每笔0.15%；官方demo开0.1%、平0.15%、最低5 | 本项目采用demo费用而非论文统一费率；分数股且无完整涨跌停规则 |
| AER/IR | 论文命名；官方demo调用Qlib risk_analysis；[Qlib源码](https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py#L23)使用均值、样本标准差及年化因子 | 本项目明确固定252。当前Qlib freq=day默认238；原论文版本未确认。不能称252为已证实的官方默认 |
| MDD | 当前Qlib sum模式为累计日收益之和的回撤 | 本项目为复利NAV峰谷回撤；定义合理但口径不同 |
| 候选数量与随机性 | 论文增加采样路径并平均；五次随机运行 | 本项目固定一份N16候选的前缀，五个Momus训练种子；不衡量独立候选生成的波动 |

Kronos论文明确预训练数据截至2024年6月；本文2024期含截止前窗口。训练时期、模型选择、候选生成日期和样本来源日期应分别披露，不能统一称为“严格未来验证”。

## 8. 主张与证据对照

{claimtable}

## 9. 后续论文重组的准入状态

1. **可用**：HS300固定股票池上的预测主表、候选数量实验、候选排序描述、训练已见/未见分组、有限条件下的配对区间。正文必须注明固定成员、90日输入、无关联输入、未复权目标及时间覆盖。
2. **需收紧措辞**：跨时期稳定性、结构机制、计算效率、严格未来验证、原Kronos完整复现。尤其2026H1时间推断不可由股票数代替。
3. **暂不可作为正式投资证据**：原投资表及只修正两项核算后的表。完整公司行为处理及异步信号规则解决前，仅作为审计附件，不能用其正负结果作强投资论证。
4. **论文仍未改动**：本阶段到证据报告为止。若下一阶段只重组预测主线，可将未通过的投资部分明确列为待补实证；若要求完整投资表，应先完成上述修复，保持旧缓存和原结果可追溯。

## 10. 复现与附件

使用现有`C:/Users/Soyo/anaconda3/envs/torch5060/python.exe`，无需安装新依赖。运行顺序：

```powershell
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/audit.py freeze
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/audit.py forecast
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/full_forecast.py
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/audit.py statistics
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/audit.py investment
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/diagnostics.py
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/test_audit.py
& 'C:/Users/Soyo/anaconda3/envs/torch5060/python.exe' examples/hs300_evidence_audit/build_report.py
```

原模型/候选/数据和此前未提交的评价脚本需存在；其身份由保护清单记录。本轮不将此前未提交文件擅自纳入提交。首次联网来源快照已保存在`sources/`，以后复算从现有快照读取，避免随远端变化。

主要附件：[逐项主张](claim_evidence.csv)、[配对区间](paired_intervals.csv)、[预测全量复算](full_forecast_replay_checks.csv)、[候选数量复算](full_scaling_replay.csv)、[三阶段投资纠错](investment_correction_summary.csv)、[独立账本核查](investment_ledger_checks.csv)、[保护输入核查](protected_inputs_check.json)。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    dump('audit_status.json',{'audit_completed':True,'manuscript_modified':False,'models_or_candidates_modified':False,'forecast_arithmetic_pass':full['passed'],'investment_accounting_fixes_completed':['benchmark field order','initial NAV return baseline'],'investment_paper_ready':False,'unresolved':['corporate action accounting','asynchronous sparse signal portfolio policy','PIT membership evidence','few temporal blocks in 2026H1'],'protected_files':protect['checked_files']})
    print(OUT/'REPORT.md')


if __name__=='__main__':main()

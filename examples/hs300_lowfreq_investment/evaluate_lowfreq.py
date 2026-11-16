"""Twelve paired low-frequency portfolios, strict coverage and ledger audits."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from common_lowfreq import ROOT, BASE, OUT, DATA, digest, identity, write_json, freeze, protect, calendar
from validation import validate_signal_schedule, audit_backtest, audit_independent_ledger
from qlib_runner import run_backtest
from prepare import bridge_mergers
from evaluate import LABELS, measures, block_intervals, cross_section_metrics


def verify_provider():
    ref = json.loads((BASE / "provider_identity.json").read_text(encoding="utf-8"))
    provider = BASE / ref["path"]
    for name, key in [("manifest.json", "manifest_sha256"), ("calendars/day.txt", "calendar_sha256"),
                      ("instruments/all.txt", "instruments_sha256")]:
        if digest(provider / name) != ref[key]:
            raise ValueError(f"Changed provider metadata: {name}")
    manifest = json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["feature_sha256"].items():
        if digest(provider / name) != sha:
            raise ValueError(f"Changed provider data: {name}")
    return provider


def load_shards(rows, protocol, cal, input_markets, candidate_root=None, generation_path=None):
    folder = candidate_root or OUT / "candidates"
    generation = json.loads((generation_path or OUT / "generation_manifest.json").read_text(encoding="utf-8"))
    if generation_path is None:
        if generation["protocol"] != identity(protocol):
            raise ValueError("Candidate generation is not bound to this frozen protocol")
        for path, sha in generation["source_sha256"].items():
            if digest(path) != sha:
                raise ValueError(f"Generation source changed: {path}")
    signals, expected, shards, checks = [[] for _ in LABELS], {}, [], []
    history_lags = []
    for row in rows:
        day, origin = pd.Timestamp(row["execution"]), pd.Timestamp(row["origin"])
        path = folder / f"{day:%Y%m%d}.npz"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if digest(path) != meta["sha256"] or meta["manifest"] != identity(generation):
            raise ValueError(f"Shard identity mismatch: {path}")
        with np.load(path, allow_pickle=False) as s:
            j = cal.get_loc(day)
            if str(s["origin"]) != row["origin"] or cal[j-1] != origin or str(s["execution"]) != row["execution"]:
                raise ValueError("Shard time alignment mismatch")
            if not pd.DatetimeIndex(s["future"]).equals(cal[j:j+10]):
                raise ValueError("Unexpected forecast label calendar")
            codes = s["codes"].tolist()
            if codes != protocol["universes"][row["execution"]] or len(set(codes)) != 300:
                raise ValueError("Shard fails exact contemporaneous 300-member coverage")
            if (pd.DatetimeIndex(s["history_end"]) > origin).any():
                raise ValueError("Forecast input uses future observations")
            if s["paths"].shape != (300,16,10,6) or s["scores"].shape != (5,300,16) or s["signals"].shape != (6,300):
                raise ValueError("Unexpected candidate/score/signal shape")
            if not all(np.isfinite(s[k]).all() for k in ("paths", "scores", "signals", "hist", "mean", "scale", "last_close")):
                raise ValueError("Nonfinite shard values")
            # Independent scalar-window aggregation, not a copy of the generator's einsum.
            rebuilt = np.empty((6,300))
            history_error = 0.
            for stock in range(300):
                history = input_markets[codes[stock]].loc[:origin]
                history = history.loc[history.volume.gt(0)].iloc[-90:]
                if len(history) != 90 or str(history.index[-1].date()) != s["history_end"][stock]:
                    raise ValueError("Cached history end/count differs from frozen observable input")
                raw = history[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=np.float32)
                mean, scale = raw.mean(axis=0), raw.std(axis=0)+1e-5
                normalized = np.clip((raw-mean)/scale, -5., 5.)
                for rebuilt_input, cached in [(mean,s["mean"][stock]), (scale,s["scale"][stock]),
                                               (normalized,s["hist"][stock])]:
                    np.testing.assert_allclose(rebuilt_input, cached, rtol=1e-6, atol=1e-6)
                np.testing.assert_equal(float(history.close.iloc[-1]), s["last_close"][stock])
                history_error = max(history_error, float(np.max(np.abs(normalized-s["hist"][stock]))))
                if history.index[-1] < origin:
                    history_lags.append(dict(execution=row["execution"], origin=row["origin"],
                        instrument=codes[stock], history_end=str(history.index[-1].date()),
                        stale_market_days=int(cal.get_loc(origin)-cal.get_loc(history.index[-1])),
                        denominator="last positive-volume observed adjusted close at/before origin"))
                closes = s["paths"][stock,:,:,3].astype(float)
                avg = closes.mean()
                rebuilt[0,stock] = (avg*s["scale"][stock,3]+s["mean"][stock,3])/s["last_close"][stock]-1
                for seed in range(5):
                    score = s["scores"][seed,stock].astype(float)
                    weights = np.exp(score-score.max())
                    avg = np.average(closes.mean(axis=1), weights=weights)
                    rebuilt[seed+1,stock] = (avg*s["scale"][stock,3]+s["mean"][stock,3])/s["last_close"][stock]-1
            # Baseline generator averages float32 paths before denormalizing.
            np.testing.assert_allclose(rebuilt, s["signals"], rtol=2e-5, atol=2e-6)
            expected[origin] = codes
            index = pd.MultiIndex.from_product([[origin], codes], names=["datetime", "instrument"])
            for target, values in zip(signals, s["signals"]):
                target.append(pd.Series(values, index=index))
            shards.append(dict(path=path, origin=origin, day=day, codes=codes))
            checks.append(dict(execution=row["execution"], stocks=300, candidates_per_stock=16,
                max_history_replay_error=history_error,
                max_signal_replay_error=float(np.max(np.abs(rebuilt-s["signals"]))), sha256=meta["sha256"]))
    signals = [pd.concat(parts) for parts in signals]
    for signal in signals:
        validate_signal_schedule(signal, expected)
    return signals, expected, shards, checks, history_lags


def write_report(root, summaries, protocol):
    table = pd.DataFrame(summaries)
    rows = []
    for period in table.period.unique():
        part = table[table.period == period]
        for label, selected in [("Kronos普通平均", part[part.method == "simple_mean"]),
                                ("Momus五种子指标均值", part[part.method != "simple_mean"])]:
            m = selected.select_dtypes(include="number").mean()
            rows.append(f"| {period} | {label} | {m.cumulative_net_return:.2%} | {m.benchmark_cumulative_return:.2%} | {m.max_drawdown:.2%} | {m.average_daily_turnover:.2%} | {m.mean_return_ic:.4f} | {m.mean_return_rank_ic:.4f} |")
    lines = ["# 沪深300低频投资补充验证", "",
        "本实验覆盖2025年和2026年上半年，分别从首个交易日起每10个交易日决策一次，共25＋12＝37个日期；每个日期300只当时生效成分股全部生成预测。两个时期分别以1亿元现金初始化，净值不可直接拼接成连续账户。", "",
        "Kronos基线对16条候选普通平均；Momus对同一批候选评分加权。保留五组既有Momus权重，分别回测；下表的Momus行是五次运行指标均值，不是五模型信号合成后的可交易组合，也不是五次独立候选生成。", "",
        "| 时期 | 方法 | 扣费累计收益 | 同期指数收益 | 最大回撤 | 日均换手 | 收益信号IC | 收益信号RankIC |",
        "|---|---|---:|---:|---:|---:|---:|---:|", *rows, "",
        "IC/RankIC逐预测日期在300只股票间计算。对应真值是未来10个交易日调整后收盘价的均值，相对截至预测原点最近一条正成交量观测的调整收盘价的变化率，不是第10日端点收益或实际开盘持有收益。停牌等情况下分母所对应日期可能早于origin；这也是冻结协议中origin close的实际实现含义，详见 history_lag_audit.csv。最后一批的完整10日标签可能延伸到期末以后；这些标签只用于评价，不参与信号和交易。", "",
        "交易保留Qlib 0.9.7的Top50/drop5/最少持有5交易日，目标95%风险仓位，只在指定日期使用前一交易日收盘信息、下一开盘执行。间隔日期不产生订单，继续估值。保留持仓不整体重配权重；卖出受阻时持仓可能超过50。退出指数的旧持仓在之后的调仓日按原策略逐步退出，并非立即清仓。期末按收盘估值，不强制平仓。", "",
        "双边费率各0.15%、最低5元，按100股取整；使用统一开盘±9.5%研究限制，无滑点或冲击成本，未覆盖所有板块/ST法定交易规则。复权合成份额与合并股权估值为研究近似，不是精确分红税费/支付日/零碎股登记账户。历史成员变更档案未获独立供应商全量确认。", "",
        "这是补充性的低频调仓实验，不是Kronos原日频投资复现。日期与协议在本批结果生成前冻结，但模型已经选定，也已有前期探索，不能追认为全研究事前注册。37个决策截面提供的市场时间证据有限；日收益区块区间不能把每日估值变成独立预测样本。", "",
        "逐种子指标见 investment_per_run.csv；配对日收益差的10/20交易日循环区块区间见 paired_date_intervals.json（5000次，RNG=20260908）。区间条件于固定模型和一份候选缓存，只作补充分析。每日净值、成交、持仓及协议均保留在各时期目录。", "",
        "核验记录：signal_replay_checks.csv、ledger_checks.json、evaluation_manifest.json。原论文、原模型与缓存未覆盖。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="reuse two paused 2024 shards ten market days apart")
    args = parser.parse_args()
    protocol = freeze()
    protect("before_evaluation")
    root = OUT / ("smoke_results" if args.smoke else "results")
    root.mkdir(exist_ok=True)
    write_json(root / "evaluation_status.json", dict(status="RUNNING", smoke=args.smoke))
    sources = list((ROOT / "examples/hs300_lowfreq_investment").glob("*.py"))
    sources += [ROOT / "examples/hs300_daily_investment" / name for name in ("qlib_runner.py", "evaluate.py", "prepare.py", "data.py")]
    write_json(root / "evaluation_manifest.json", dict(protocol=identity(protocol),
        source_sha256={str(p):digest(p) for p in sources}, provider_identity=json.loads((BASE / "provider_identity.json").read_text(encoding="utf-8"))))
    provider = verify_provider()
    cal = calendar()
    members = pd.read_csv(BASE / "membership.csv", parse_dates=["start_date", "end_date"])
    markets = {c: pd.read_csv(DATA / "market" / f"{c}.csv", parse_dates=["date"]).set_index("date")
               for c in members.instrument.unique()}
    input_markets = markets
    markets, _ = bridge_mergers(markets, cal, json.loads((BASE / "merger_events.json").read_text(encoding="utf-8"))["events"])
    periods = protocol["periods"]
    rows = protocol["schedule"]
    if args.smoke:
        periods = {"smoke": ["2024-01-02", "2024-01-29"]}
        days = cal[(cal >= "2024-01-02") & (cal <= "2024-01-29")][::10]
        rows = [dict(period="smoke", execution=str(d.date()), origin=str(cal[cal.get_loc(d)-1].date())) for d in days]
        protocol = dict(protocol)
        protocol["universes"] = {str(d.date()):sorted(members.loc[(members.start_date <= d) & (members.end_date >= d), "instrument"]) for d in days}
    summaries, inference, ledger, replay, history_lags = [], [], [], [], []
    started = time.time()
    for period, (start, end) in periods.items():
        days = cal[(cal >= start) & (cal <= end)]
        selected = [r for r in rows if r["period"] == period]
        signals, expected, shards, checks, lag_rows = load_shards(selected, protocol, cal, input_markets,
            BASE / "candidates" if args.smoke else None, BASE / "generation_manifest.json" if args.smoke else None)
        replay.extend(checks)
        history_lags.extend(lag_rows)
        reports = []
        directory = root / period
        directory.mkdir(exist_ok=True)
        execution_dates = [r["execution"] for r in selected]
        for label, signal in zip(LABELS, signals):
            print(f"backtest {period} {label}", flush=True)
            result = run_backtest(provider, signal, days[0], days[-1], expected_signals=expected,
                                  allow_missing_signal_dates=True)
            audit = audit_backtest(result, execution_dates, days)
            audit["independent_ledger"] = audit_independent_ledger(result, markets, cal)
            # Every actually purchased name must belong to the dated forecast universe.
            for order in result.trades.itertuples():
                day = str(pd.Timestamp(order.datetime).date())
                if order.direction == "buy" and order.instrument not in protocol["universes"][day]:
                    raise ValueError("Purchased a name outside the scheduled constituent universe")
            ledger.append(dict(period=period, method=label, **audit))
            result.report.to_csv(directory / f"{label}_daily.csv", float_format="%.12g")
            result.trades.to_csv(directory / f"{label}_orders.csv", index=False, float_format="%.12g")
            holdings = []
            for day, position in result.positions.items():
                for code in position.get_stock_list():
                    holdings.append(dict(date=str(day.date()), instrument=code, synthetic_amount=position.get_stock_amount(code),
                        adjusted_price=position.get_stock_price(code), weight=position.get_stock_weight(code), cash=position.get_cash()))
            pd.DataFrame(holdings).to_csv(directory / f"{label}_positions.csv", index=False, float_format="%.12g")
            result.metadata.update(scheduled_execution_dates=execution_dates, no_orders_between_decisions=True)
            write_json(directory / f"{label}_protocol.json", result.metadata)
            summaries.append(dict(period=period, method=label, decision_dates=len(selected), **measures(result.report, result.trades)))
            reports.append(result.report)
        ic = cross_section_metrics(shards, markets, cal)
        ic["label"] = "mean next 10 market-day adjusted closing marks / latest positive-volume observed adjusted close at or before origin - 1"
        if len(ic) != len(selected)*6 or not (ic.stocks == 300).all():
            raise ValueError("Incomplete cross-sectional IC coverage")
        ic.to_csv(directory / "decision_return_ic.csv", index=False, float_format="%.12g")
        for row in summaries:
            if row["period"] == period:
                part = ic[ic.method == row["method"]]
                row.update(mean_return_ic=float(part.return_ic.mean()), mean_return_rank_ic=float(part.return_rank_ic.mean()))
        delta = np.mean([r.net_return.to_numpy() for r in reports[1:]], axis=0)-reports[0].net_return.to_numpy()
        inference.append(dict(period=period, estimand="mean daily net-return difference: mean of five Momus accounts minus shared-candidate baseline",
            daily_mean_difference=float(delta.mean()), decision_dates=len(selected), candidate_repeats=1,
            blocks=[block_intervals(delta,b,np.random.default_rng(20260908)) for b in (10,20)]))
        pd.DataFrame(summaries).to_csv(root / "investment_per_run.csv", index=False, float_format="%.12g")
        pd.DataFrame(replay).to_csv(root / "signal_replay_checks.csv", index=False, float_format="%.12g")
        pd.DataFrame(history_lags, columns=["execution", "origin", "instrument", "history_end", "stale_market_days", "denominator"]).to_csv(root / "history_lag_audit.csv", index=False)
        write_json(root / "ledger_checks.json", ledger)
        write_json(root / "paired_date_intervals.json", inference)
    protect("after_evaluation")
    if not args.smoke:
        write_report(root, summaries, protocol)
    write_json(root / "evaluation_status.json", dict(status="SMOKE_PASSED" if args.smoke else "COMPLETE",
        portfolios=len(summaries), forecast_dates=len(replay), elapsed_seconds=time.time()-started,
        ledger_audits_passed=len(ledger), manuscript_modified=False))


if __name__ == "__main__":
    main()

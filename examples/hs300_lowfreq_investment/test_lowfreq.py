"""Deterministic ledgers for sparse decisions with daily Qlib accounting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from examples.hs300_daily_investment.qlib_runner import run_backtest, write_provider
from examples.hs300_daily_investment.test_qlib_runner import frame, scores
from examples.hs300_lowfreq_investment.validation import (
    audit_backtest, audit_independent_ledger, validate_signal_schedule,
)


A, B, C = "SH600000", "SZ000001", "SH600001"


class LowFrequencyLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.provider = Path(self.temp.name) / "provider"
        self.dates = pd.bdate_range("2025-01-02", periods=23)
        self.frames = {code: frame(self.dates) for code in ("SH000300", A, B, C)}

    def run_case(self, rows, end=15, account=100_000):
        origins = self.dates[[i * 10 for i in range(len(rows))]]
        sig = scores(origins, rows)
        expected = dict(zip(origins, [set(row) for row in rows]))
        validate_signal_schedule(sig, expected)
        write_provider(self.frames, self.provider)
        result = run_backtest(
            self.provider, sig, self.dates[1], self.dates[end],
            topk=1, n_drop=1, hold_thresh=5, account=account,
            expected_signals=expected, allow_missing_signal_dates=True,
        )
        executions = self.dates[[i * 10 + 1 for i in range(len(rows))]]
        check = audit_backtest(result, executions, self.dates[1:end + 1], account=account)
        self.assertEqual(check["status"], "PASSED")
        replay = audit_independent_ledger(result, self.frames, self.dates, account=account)
        self.assertEqual(replay["status"], "PASSED")
        return result

    def test_independent_replay_checks_minimum_fee(self):
        result = self.run_case([{A: 2., B: 1.}, {A: 1., B: 2.}], account=2_000)
        self.assertGreaterEqual(len(result.trades), 3)
        self.assertTrue((result.trades.deal_amount > 0).all())
        np.testing.assert_array_equal(result.trades.cost, np.full(len(result.trades), 5.))

    def test_independent_replay_detects_source_price_and_fee_errors(self):
        result = self.run_case([{A: 2., B: 1.}, {A: 1., B: 2.}])
        broken = {code: data.copy() for code, data in self.frames.items()}
        broken[A].loc[self.dates[4], "close"] = 11.
        with self.assertRaisesRegex(AssertionError, "Independent ledger close_price"):
            audit_independent_ledger(result, broken, self.dates, account=100_000)
        broken[A] = self.frames[A].copy()
        broken[A].loc[self.dates[1], "open"] = 11.
        with self.assertRaisesRegex(AssertionError, "Independent ledger execution_price"):
            audit_independent_ledger(result, broken, self.dates, account=100_000)
        result.trades.loc[result.trades.index[0], "cost"] += 1.
        with self.assertRaisesRegex(AssertionError, "Independent ledger fee"):
            audit_independent_ledger(result, self.frames, self.dates, account=100_000)

    def test_independent_replay_uses_provider_precision_and_carries_only_past_marks(self):
        for code in (A, B):
            self.frames[code][["open", "high", "low", "close"]] = 10.123456789
        self.frames[A].loc[self.dates[5], ["open", "high", "low", "close"]] = np.nan
        self.frames[A].loc[self.dates[5], ["limit_buy", "limit_sell"]] = 1.
        self.frames[A].loc[self.dates[6]:, ["open", "high", "low", "close"]] = 11.987654321
        result = self.run_case([{A: 2., B: 1.}, {A: 1., B: 2.}])
        self.assertEqual(result.report.loc[self.dates[5], "net_nav"],
                         result.report.loc[self.dates[4], "net_nav"])
        self.assertGreater(result.report.loc[self.dates[6], "net_nav"],
                           result.report.loc[self.dates[5], "net_nav"])

    def test_ten_day_gap_marks_daily_without_orders_and_counts_holding_days(self):
        self.frames[A].loc[self.dates[4]:, ["open", "high", "low", "close"]] = 12.
        result = self.run_case([{A: 2., B: 1.}, {A: 1., B: 2.}])
        actual_dates = set(pd.DatetimeIndex(result.trades.datetime).normalize())
        self.assertEqual(actual_dates, {self.dates[1], self.dates[11]})
        idle = result.report.loc[~result.report.index.isin(actual_dates)]
        np.testing.assert_array_equal(idle.cost, np.zeros(len(idle)))
        np.testing.assert_array_equal(idle.turnover, np.zeros(len(idle)))
        self.assertGreater(result.report.loc[self.dates[4], "net_nav"],
                           result.report.loc[self.dates[3], "net_nav"])
        original_amount = result.positions[self.dates[1]].get_stock_amount(A)
        for date in self.dates[2:11]:
            self.assertEqual(result.positions[date].get_stock_amount(A), original_amount)
        sells = result.trades.query("direction == 'sell' and instrument == @A")
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells.iloc[0].datetime.normalize(), self.dates[11])
        # Only two decision dates elapsed: selling proves hold5 counts daily bars.
        self.assertNotIn(A, result.positions[self.dates[11]].get_stock_list())
        self.assertIn(B, result.positions[self.dates[11]].get_stock_list())

    def test_removed_member_waits_for_next_scheduled_decision(self):
        result = self.run_case([{A: 2., B: 1.}, {B: 2., C: 1.}])
        for date in self.dates[1:11]:
            self.assertIn(A, result.positions[date].get_stock_list())
        sells = result.trades.query("direction == 'sell' and instrument == @A")
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells.iloc[0].datetime.normalize(), self.dates[11])
        self.assertNotIn(A, result.positions[self.dates[11]].get_stock_list())

    def test_removed_member_blocked_at_decision_waits_until_later_schedule(self):
        self.frames[A].loc[self.dates[11], "limit_sell"] = 1.
        result = self.run_case(
            [{A: 2., B: 1.}, {B: 2., C: 1.}, {B: 2., C: 1.}], end=22,
        )
        for date in self.dates[11:21]:
            self.assertIn(A, result.positions[date].get_stock_list())
        sells = result.trades.query("direction == 'sell' and instrument == @A")
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells.iloc[0].datetime.normalize(), self.dates[21])
        self.assertNotIn(A, result.positions[self.dates[21]].get_stock_list())


class FrozenScheduleTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=12)
        self.expected = {self.dates[0]: {A, B}, self.dates[10]: {A, B}}
        self.signal = scores(self.dates[[0, 10]], [{A: 2., B: 1.}, {A: 1., B: 2.}])

    def test_complete_schedule_accepted(self):
        validate_signal_schedule(self.signal, self.expected)

    def test_missing_scheduled_date_rejected(self):
        broken = self.signal.drop(self.dates[10], level="datetime")
        with self.assertRaisesRegex(ValueError, "Signal dates differ"):
            validate_signal_schedule(broken, self.expected)

    def test_extra_signal_date_rejected(self):
        extra = scores(self.dates[[5]], [{A: 2., B: 1.}])
        with self.assertRaisesRegex(ValueError, "Signal dates differ"):
            validate_signal_schedule(pd.concat([self.signal, extra]), self.expected)

    def test_missing_stock_rejected(self):
        broken = self.signal.drop((self.dates[10], B))
        with self.assertRaisesRegex(ValueError, "Incomplete scheduled universe"):
            validate_signal_schedule(broken, self.expected)


if __name__ == "__main__":
    unittest.main()

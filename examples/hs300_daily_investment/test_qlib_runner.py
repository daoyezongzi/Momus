"""Small deterministic financial ledgers executed by the pinned Qlib engine."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from examples.hs300_daily_investment.qlib_runner import validate_frames, write_provider, run_backtest


def frame(dates, price=10.):
    x = pd.DataFrame(index=dates)
    for col in ("open", "high", "low", "close"):
        x[col] = float(price)
    x["volume"], x["factor"] = 1_000_000., 1.
    x["limit_buy"], x["limit_sell"] = 0., 0.
    return x


def scores(dates, daily):
    rows = [(date, code, value) for date, values in zip(dates, daily) for code, value in values.items()]
    return pd.DataFrame(rows, columns=["datetime", "instrument", "score"]).set_index(
        ["datetime", "instrument"])["score"]


class QlibRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = Path(self.temp.name) / "provider"
        self.dates = pd.bdate_range("2024-01-02", periods=9)
        self.frames = {code: frame(self.dates) for code in ("SH000300", "SH600000", "SZ000001")}

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, sig, start=1, end=1, **kwargs):
        write_provider(self.frames, self.provider)
        return run_backtest(self.provider, sig, self.dates[start], self.dates[end],
                            topk=1, n_drop=1, account=kwargs.pop("account", 10_000), **kwargs)

    def test_first_day_fee_and_open_to_close(self):
        self.frames["SH600000"].loc[self.dates[1], ["close", "high"]] = 11.
        result = self.run_case(scores(self.dates, [{"SH600000": 2, "SZ000001": 1}]))
        row = result.report.iloc[0]
        self.assertAlmostEqual(row["account"], 10886.5)
        self.assertAlmostEqual(row["return"], .09)
        self.assertAlmostEqual(row["cost"], .00135)
        self.assertEqual(result.trades.iloc[0].raw_shares, 900)

    def test_signal_lag_and_minimum_holding_bars(self):
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}] +
                     [{"SH600000": 1, "SZ000001": 2}] * 7)
        result = self.run_case(sig, end=7, account=100_000)
        first = result.trades.iloc[0]
        self.assertEqual(first.instrument, "SH600000")
        sells = result.trades.query("direction == 'sell' and instrument == 'SH600000'")
        self.assertEqual(sells.iloc[0].datetime.normalize(), self.dates[6])
        # The official strategy may spend leftover cash on B before A can sell.
        self.assertTrue(result.metadata["dates_above_topk"])

    def test_retained_holdings_are_not_rebalanced(self):
        self.frames["SH600000"].loc[self.dates[2]:, ["open", "high", "low", "close"]] = 12.
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}] * 7)
        result = self.run_case(sig, end=6)
        self.assertEqual(len(result.trades), 1)

    def test_split_preserves_value_and_maps_physical_shares(self):
        # Raw price halves 10 -> 5; adjusted OHLC remains 10, factor doubles.
        self.frames["SH600000"].loc[self.dates[2]:, "factor"] = 2.
        self.frames["SH600000"].loc[self.dates[2]:, "volume"] = 500_000.
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}] * 4)
        result = self.run_case(sig, end=3)
        np.testing.assert_allclose(result.report["account"], [9986.5] * 3)
        synthetic = result.positions[self.dates[3]].get_stock_amount("SH600000")
        self.assertEqual(synthetic * 2, 1800)

    def test_factor_controls_lot_rounding_and_initial_drawdown(self):
        self.frames["SH600000"]["factor"] = 2.
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}])
        result = self.run_case(sig)
        # 950 adjusted units are 1900 physical shares at raw price 5.
        self.assertEqual(result.trades.iloc[0].raw_shares, 1900)
        self.assertEqual(result.trades.iloc[0].deal_amount, 950)
        self.assertAlmostEqual(result.report.iloc[0].net_drawdown, -14.25 / 10000)

    def test_open_limit_blocks_trade_even_if_close_unlocked(self):
        x = self.frames["SH600000"]
        x.loc[self.dates[1], ["open", "high"]] = 11.
        x.loc[self.dates[1], "limit_buy"] = 1.
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}])
        result = self.run_case(sig)
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.report.iloc[0]["account"], 10000)

    def test_invalid_price_factor_or_limit_flags_rejected(self):
        for field in ("open", "factor", "limit_buy"):
            with self.subTest(field=field):
                broken = {k: v.copy() for k, v in self.frames.items()}
                broken["SH600000"].loc[self.dates[1], field] = np.nan
                with self.assertRaises(ValueError):
                    validate_frames(broken)

    def test_missing_signals_and_provider_overwrite_rejected(self):
        write_provider(self.frames, self.provider)
        with self.assertRaises(FileExistsError):
            write_provider(self.frames, self.provider)
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}])
        with self.assertRaisesRegex(ValueError, "Missing signal date"):
            run_backtest(self.provider, sig, self.dates[1], self.dates[2], topk=1)
        with self.assertRaisesRegex(ValueError, "universe mismatch"):
            run_backtest(self.provider, sig, self.dates[1], self.dates[1], topk=1,
                         expected_signals={self.dates[0]: {"SH600000"}})

    def test_explicit_whole_day_signal_gap_keeps_holdings(self):
        sig = scores(self.dates, [{"SH600000": 2, "SZ000001": 1}])
        result = self.run_case(sig, end=3, allow_missing_signal_dates=True)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.report.holding_count.tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()

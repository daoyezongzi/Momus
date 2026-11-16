import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import adjusted_frame
from prepare import bridge_mergers


def raw_frame(dates, prices):
    frame = pd.DataFrame({name: prices for name in ("open", "high", "low", "close")}, index=dates)
    frame["volume"] = 1000.
    frame["amount"] = frame.close * frame.volume
    frame["previous_observed_close"] = frame.close.shift(1).fillna(frame.close.iloc[0])
    frame["reference_close"] = frame.previous_observed_close
    frame["explicit_reference"] = 0
    return frame


class PipelineTests(unittest.TestCase):
    def test_future_prices_do_not_change_forecast_inputs(self):
        from generate import inputs
        dates = pd.bdate_range("2024-01-01", periods=110)
        frame = raw_frame(dates, np.arange(110) + 10.)
        origin, future = dates[99], dates[100:110]
        first = inputs({"SH600000": frame}, ["SH600000"], origin, future)
        changed = frame.copy(); changed.loc[future, ["open", "high", "low", "close", "volume", "amount"]] *= 1000
        second = inputs({"SH600000": changed}, ["SH600000"], origin, future)
        for key in ("hist", "mean", "scale", "last_close", "x_stamp", "y_stamp"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_split_adjustment_preserves_value_and_turnover(self):
        dates = pd.bdate_range("2025-01-01", periods=2)
        raw = raw_frame(dates, [10., 5.]); raw.loc[dates[1], "volume"] = 2000.; raw["amount"] = 10000.
        factors = pd.DataFrame({"factor": [1., 2.]}, index=dates)
        frame = adjusted_frame(raw, factors)
        np.testing.assert_array_equal(frame.close, [10., 10.])
        np.testing.assert_array_equal(frame.volume, [1000., 1000.])
        np.testing.assert_array_equal(frame.amount, [10000., 10000.])
        self.assertEqual(frame.reference_close.iloc[1], 5.)
        self.assertEqual(frame.limit_sell.iloc[1], 0.)

    def test_no_future_factor_backfill(self):
        dates = pd.bdate_range("2025-01-01", periods=2)
        factors = pd.DataFrame({"factor": [2.]}, index=[dates[1]])
        with self.assertRaisesRegex(ValueError, "do not backfill"):
            adjusted_frame(raw_frame(dates, [10., 5.]), factors)

    def test_merger_unit_identity_and_unavailable_shares(self):
        dates = pd.bdate_range("2025-01-01", periods=5)
        old = adjusted_frame(raw_frame(dates[:1], [12.]), pd.DataFrame({"factor": [1.]}, index=dates[:1]))
        new = adjusted_frame(raw_frame(dates, [20., 20., 20., 10., 10.]), pd.DataFrame({"factor": [1., 1., 1., 2., 2.]}, index=dates))
        event = {"absorbed_instrument": "SH600837", "successor_instrument": "SH601211", "exchange_ratio_new_raw_shares_per_old_raw_share": .62, "last_trading_date": str(dates[0].date()), "delisting_date": str(dates[1].date()), "first_unrestricted_public_holder_sell_date": str(dates[3].date())}
        frames, _ = bridge_mergers({"SH600837": old, "SH601211": new}, dates, [event])
        proxy = frames["SH600837"].loc[dates[1]:]
        # 100 original synthetic units = 100 original raw shares => 62 new
        # shares, subsequently 124 after a 2-for-1 split; wealth stays 1240.
        np.testing.assert_allclose(proxy.close * 100, 1240.)
        self.assertEqual(proxy.limit_sell.iloc[0], 1.)
        self.assertEqual(proxy.limit_sell.loc[dates[3]], 0.)
        self.assertAlmostEqual(proxy.factor.iloc[-1] * 100, 124.)


if __name__ == "__main__":
    unittest.main()

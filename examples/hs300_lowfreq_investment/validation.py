"""Strict low-frequency contracts, independent of the candidate generator."""
import numpy as np
import pandas as pd


def validate_signal_schedule(signal, expected):
    if not isinstance(signal.index, pd.MultiIndex) or signal.index.has_duplicates:
        raise ValueError("Signals require unique origin/instrument pairs")
    actual_dates = set(pd.DatetimeIndex(signal.index.get_level_values("datetime")))
    expected_dates = {pd.Timestamp(d) for d in expected}
    if actual_dates != expected_dates:
        raise ValueError("Signal dates differ from frozen scheduled origins")
    if not np.isfinite(signal.to_numpy()).all():
        raise ValueError("Nonfinite signal")
    for day, codes in expected.items():
        actual = set(signal.xs(pd.Timestamp(day), level="datetime").index)
        if actual != set(codes):
            raise ValueError(f"Incomplete scheduled universe: {day}")


def audit_backtest(result, execution_dates, expected_daily_dates, account=100_000_000):
    report, trades = result.report, result.trades
    allowed = set(pd.DatetimeIndex(execution_dates))
    if not report.index.equals(pd.DatetimeIndex(expected_daily_dates)):
        raise ValueError("Daily NAV coverage differs from the full evaluation calendar")
    if not trades.empty and not set(pd.DatetimeIndex(trades.datetime).normalize()).issubset(allowed):
        raise ValueError("Order submitted outside a scheduled execution date")
    idle = report.loc[~report.index.isin(allowed)]
    np.testing.assert_allclose(idle.cost, 0., atol=1e-12)
    np.testing.assert_allclose(idle.turnover, 0., atol=1e-12)
    np.testing.assert_allclose((1+report.net_return).cumprod(), report.net_nav, rtol=1e-10)
    reconstructed, max_error = [], 0.
    for day, position in result.positions.items():
        value = position.get_cash() + sum(position.get_stock_amount(c)*position.get_stock_price(c)
                                          for c in position.get_stock_list())
        error = abs(value-float(report.loc[day, "account"]))
        max_error = max(max_error, error)
        np.testing.assert_allclose(value, report.loc[day, "account"], rtol=1e-10, atol=1e-5)
        reconstructed.append(value)
    previous = report["account"].shift(1).fillna(account)
    daily_cost = (trades.groupby(pd.DatetimeIndex(trades.datetime).normalize()).cost.sum()
                  if not trades.empty else pd.Series(dtype=float))
    np.testing.assert_allclose(report.cost*previous, daily_cost.reindex(report.index, fill_value=0),
                               rtol=1e-8, atol=1e-5)
    return dict(status="PASSED", daily_dates=len(report), scheduled_dates=len(allowed),
                orders=len(trades), idle_days=len(idle), max_cash_positions_nav_error=max_error,
                orders_only_on_schedule=True, initial_cash_included=True)


def audit_independent_ledger(result, market, calendar, account=100_000_000):
    """Replay fills against source OHLC, independent of Qlib's position ledger.

    ``market`` must contain the same already-adjusted, merger-bridged synthetic
    instruments supplied to the provider. Shares here are synthetic units;
    adjustment factors therefore do not trigger another share/cash mutation.
    Source quotes are rounded to float32 exactly as the provider writer does.
    Missing closes carry only past known marks; execution opens never fill.
    This verifies accounting and the fixed 0.15%/minimum-5 fee convention, not
    order optimality, historical vendor truth, statutory rules, or dividends.
    """
    report, trades = result.report, result.trades
    cal = pd.DatetimeIndex(calendar)
    dates = pd.DatetimeIndex(report.index)
    if (dates.empty or not dates.isin(cal).all() or dates.has_duplicates or
            not dates.equals(cal[(cal >= dates[0]) & (cal <= dates[-1])])):
        raise ValueError("Independent ledger requires the complete daily report calendar")
    if set(result.positions) != set(dates):
        raise ValueError("Independent ledger requires every daily position snapshot")
    order_dates = pd.DatetimeIndex(trades.datetime).normalize()
    if not order_dates.isin(dates).all() or not order_dates.is_monotonic_increasing:
        raise ValueError("Fill ledger is outside the report or not chronological")
    needed = set(trades.instrument)
    needed.update(code for position in result.positions.values() for code in position.get_stock_list())
    quotes = {}
    for code in needed:
        if code not in market:
            raise ValueError(f"Missing source market for ledger: {code}")
        frame = market[code]
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate source market dates: {code}")
        quotes[code] = frame.reindex(cal)[["open", "close"]].astype(np.float32).astype(float)
        quotes[code]["close"] = quotes[code].close.ffill()
    cash, amounts = float(account), {}
    maxima = dict(cash=0., quantity=0., close_price=0., execution_price=0.,
                  fill_value=0., fee=0., account=0.)
    fills, zero_fills, holding_marks = 0, 0, 0

    def check(key, actual, expected, *, rtol=1e-10, atol=1e-5):
        if not np.isfinite(actual) or not np.isfinite(expected):
            raise ValueError(f"Nonfinite independent ledger {key}")
        maxima[key] = max(maxima[key], abs(float(actual)-float(expected)))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   err_msg=f"Independent ledger {key}")

    for day in dates:
        # DataFrame selection preserves the recorded serial execution order.
        for order in trades.loc[order_dates == day].itertuples():
            quantity = float(order.deal_amount)
            if not np.isfinite(quantity) or quantity < 0:
                raise ValueError("Invalid executed synthetic quantity")
            if quantity == 0:
                check("fill_value", float(order.value), 0.)
                check("fee", float(order.cost), 0.)
                zero_fills += 1
                continue
            if order.direction not in ("buy", "sell"):
                raise ValueError("Unknown execution direction")
            price = float(quotes[order.instrument].loc[day, "open"])
            if not np.isfinite(price) or price <= 0:
                raise ValueError("Executed fill lacks a positive contemporaneous source open")
            check("execution_price", float(order.price), price, rtol=1e-12, atol=1e-9)
            value = quantity*price
            fee = max(value*.0015, 5.)
            check("fill_value", float(order.value), value)
            check("fee", float(order.cost), fee)
            old = amounts.get(order.instrument, 0.)
            if order.direction == "buy":
                cash -= value+fee
                amounts[order.instrument] = old+quantity
            else:
                if quantity > old and not np.isclose(quantity, old, rtol=1e-10, atol=1e-8):
                    raise ValueError("Executed sale exceeds independently reconstructed holdings")
                cash += value-fee
                remaining = old-quantity
                if abs(remaining) <= 1e-8:
                    amounts.pop(order.instrument, None)
                else:
                    amounts[order.instrument] = remaining
            fills += 1
        position = result.positions[day]
        if set(position.get_stock_list()) != set(amounts):
            raise ValueError(f"Independent held-instrument set differs on {day}")
        check("cash", float(position.get_cash()), cash)
        value = cash
        for code, quantity in amounts.items():
            check("quantity", float(position.get_stock_amount(code)), quantity, atol=1e-8)
            close = float(quotes[code].loc[day, "close"])
            if not np.isfinite(close) or close <= 0:
                raise ValueError("Held instrument lacks a positive past-known closing mark")
            check("close_price", float(position.get_stock_price(code)), close, rtol=1e-12, atol=1e-9)
            value += quantity*close
            holding_marks += 1
        check("account", float(report.loc[day, "account"]), value)
        np.testing.assert_allclose(report.loc[day, "net_nav"], value/account,
                                   rtol=1e-10, atol=1e-12)
    return dict(status="PASSED", daily_dates=len(dates), executed_fills=fills,
                zero_fills=zero_fills, source_holding_marks=holding_marks,
                max_absolute_errors=maxima, source_quote_precision="provider float32",
                value_rtol=1e-10, value_atol=1e-5, quantity_atol=1e-8,
                price_rtol=1e-12, price_atol=1e-9,
                policy="serial fill replay; source open execution and past-close marks; synthetic units; 0.15% fee each side with minimum 5; no additional corporate-action cash ledger")

import numpy as np

def roi_reward(price_now, price_next, position):
    roi = (price_next - price_now) / price_now
    return position * roi  # nếu position=0 (Hold) → roi=0

def sharpe_reward(returns, eps=1e-8):
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns) + eps
    return mean_r / std_r

def max_drawdown(equity_curve):
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    return np.max(drawdowns)

def transaction_cost_penalty(prev_pos, new_pos, balance, fee_rate=0.001):
    if prev_pos == new_pos:
        return 0.0
    return -abs(new_pos - prev_pos) * fee_rate * balance

def apply_slippage(price, position, slippage_rate=0.0005):
    """
    position: +1=Buy, -1=Sell, 0=Hold → no slippage
    """
    if position > 0:      # Buy
        return price * (1 + slippage_rate)
    elif position < 0:    # Sell
        return price * (1 - slippage_rate)
    else:                 # Hold
        return price

def composite_reward(
    price_now,
    price_next,
    position,
    prev_position,
    balance_history,
    roi_history,
    weights=(1.0, 0.5, 0.8, 0.3),
    fee_rate=0.001,
    slippage_rate=0.0005,
):
    # --- Adjust price for slippage (respect Hold) ---
    exec_price_now = apply_slippage(price_now, position, slippage_rate)
    exec_price_next = apply_slippage(price_next, position, slippage_rate)

    # --- ROI term ---
    roi = roi_reward(exec_price_now, exec_price_next, position)

    # --- Sharpe term ---
    sharpe = sharpe_reward(np.array(roi_history + [roi]))

    # --- MDD term ---
    mdd = max_drawdown(np.array(balance_history + [balance_history[-1]]))

    # --- Transaction cost (only when switching pos) ---
    cost = transaction_cost_penalty(prev_position, position, balance_history[-1], fee_rate)

    w1, w2, w3, w4 = weights
    total_reward = w1 * roi + w2 * sharpe - w3 * mdd - w4 * abs(cost)

    return float(total_reward), {
        "roi": roi,
        "sharpe": sharpe,
        "mdd": mdd,
        "cost": cost
    }

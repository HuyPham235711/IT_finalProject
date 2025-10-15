def calc_sma(values, period):
    sma = []
    for i in range(len(values)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(values[i-period+1:i+1]) / period)
    return sma

def calc_rsi(closes, period=14):
    rsi = []
    gains, losses = [], []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))

    for i in range(len(closes)):
        if i < period:
            rsi.append(None)
        else:
            avg_gain = sum(gains[i-period:i]) / period
            avg_loss = sum(losses[i-period:i]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi.append(100 - (100 / (1 + rs)) if avg_loss != 0 else 100)
    return rsi

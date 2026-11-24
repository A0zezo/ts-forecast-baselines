"""Baseline forecasters on a synthetic seasonal series, compared by MAPE and MAE."""
import numpy as np


def make_series(n=156, period=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 0.5 * t + 15 * np.sin(2 * np.pi * t / period) + rng.normal(0, 2, n)


def naive(train, h, period=12):
    return np.repeat(train[-1], h)


def seasonal_naive(train, h, period=12):
    return np.array([train[-period + (i % period)] for i in range(h)])


def moving_average(train, h, period=12):
    return np.repeat(train[-period:].mean(), h)


def holt_winters_additive(train, h, period=12, alpha=0.3, beta=0.05, gamma=0.2):
    level = train[:period].mean()
    trend = (train[period:2 * period].mean() - train[:period].mean()) / period
    season = list(train[:period] - level)
    for i, y in enumerate(train):
        s = season[i % period]
        new_level = alpha * (y - s) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season[i % period] = gamma * (y - new_level) + (1 - gamma) * s
        level = new_level
    n = len(train)
    return np.array([level + (i + 1) * trend + season[(n + i) % period] for i in range(h)])


def mae(a, f):
    return float(np.mean(np.abs(a - f)))


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


MODELS = {"naive": naive, "moving_average": moving_average,
          "seasonal_naive": seasonal_naive, "holt_winters": holt_winters_additive}


def evaluate(series, h=24):
    train, test = series[:-h], series[-h:]
    return {name: (mae(test, fn(train, h)), mape(test, fn(train, h))) for name, fn in MODELS.items()}


if __name__ == "__main__":
    res = evaluate(make_series())
    print(f"{'model':<16}{'MAE':>8}{'MAPE %':>9}")
    for name, (a, b) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<16}{a:>8.2f}{b:>9.2f}")

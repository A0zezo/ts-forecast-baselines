import numpy as np
from forecast import *


def test_shapes():
    s = make_series()
    for fn in MODELS.values():
        assert fn(s[:-24], 24).shape == (24,)


def test_seasonal_naive_exact_on_pure_season():
    s = np.tile(np.arange(12.0), 5)
    assert np.allclose(seasonal_naive(s[:-12], 12), s[-12:])


def test_seasonal_models_beat_naive():
    r = evaluate(make_series())
    assert r["seasonal_naive"][0] < r["naive"][0]
    assert r["holt_winters"][0] < r["naive"][0]


def test_metrics():
    assert mae(np.array([1., 2.]), np.array([2., 2.])) == 0.5
    assert abs(mape(np.array([100.]), np.array([90.])) - 10) < 1e-9

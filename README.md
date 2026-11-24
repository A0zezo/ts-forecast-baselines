# ts-forecast-baselines

Four forecasting baselines (naive, moving average, seasonal naive, additive Holt-Winters written in NumPy) compared on a synthetic monthly-style series (trend + yearly seasonality + noise, seed 0, 156 points). The last 24 points are held out.

## Run

```bash
pip install -r requirements.txt
python forecast.py
pytest
```

## Output (reproduced)

```
model                MAE   MAPE %
holt_winters        2.94     1.70
seasonal_naive      9.76     5.67
moving_average     11.07     6.20
naive              13.53     7.58

$ pytest -q
4 passed in 0.04s
```

## Limits

- The data is synthetic and the Holt-Winters parameters (alpha 0.3, beta 0.05, gamma 0.2) are fixed by hand, not tuned. The ranking reflects this one series only.

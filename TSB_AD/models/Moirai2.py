from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseDetector


@dataclass
class _Moirai2Config:
    model_id: str = "Salesforce/moirai-2.0-R-small"
    win_size: int = 128
    prediction_length: int = 1
    batch_size: int = 32
    quantile_low: float = 0.1
    quantile_high: float = 0.9
    device: Optional[str] = None
    freq: str = "h"


class Moirai2(BaseDetector):
    """
    Zero-shot anomaly detector built on top of Moirai 2.0 forecasting.

    Design:
    - for each timestamp t, use the previous `win_size` points as context;
    - forecast the next `prediction_length` values with Moirai 2.0;
    - convert forecast to anomaly score via plain squared forecast error;
    - average scores across overlapping predictions and dimensions.

    This matches the repo's unsupervised FM pattern:
        clf.fit(data)
        score = clf.decision_scores_
    """

    def __init__(
        self,
        win_size: int = 128,
        prediction_length: int = 1,
        batch_size: int = 32,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        quantile_low: float = 0.1,
        quantile_high: float = 0.9,
        device: Optional[str] = None,
        freq: str = "h",
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)

        if prediction_length < 1:
            raise ValueError("prediction_length must be >= 1")
        if win_size < 4:
            raise ValueError("win_size must be >= 4")
        if not (0.0 < quantile_low < 0.5):
            raise ValueError("quantile_low must be in (0, 0.5)")
        if not (0.5 < quantile_high < 1.0):
            raise ValueError("quantile_high must be in (0.5, 1.0)")
        if quantile_low >= quantile_high:
            raise ValueError("quantile_low must be < quantile_high")

        self.cfg = _Moirai2Config(
            model_id=model_id,
            win_size=win_size,
            prediction_length=prediction_length,
            batch_size=batch_size,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            device=device,
            freq=freq,
        )

        self._forecast_model = None
        self._predictor = None

    # ---------- public API ----------

    def fit(self, X, y=None):
        X = self._validate_input_array(X)
        self._lazy_init_model(target_dim=1)
        self.decision_scores_ = self._score_array(X)
        self._postprocess_after_fit()
        return self

    def decision_function(self, X):
        X = self._validate_input_array(X)
        if self._forecast_model is None or self._predictor is None:
            self._lazy_init_model(target_dim=1)
        return self._score_array(X)

    # ---------- internals ----------

    def _validate_input_array(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X[:, None]
        elif X.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape={X.shape}")

        if len(X) <= self.cfg.win_size:
            raise ValueError(
                f"Series too short: len(X)={len(X)} must be > win_size={self.cfg.win_size}"
            )

        # simple NaN handling; Moirai can work with missingness in principle,
        # but repo baselines usually expect a clean numeric array.
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        return X

    def _lazy_init_model(self, target_dim: int = 1):
        if self._forecast_model is not None and self._predictor is not None:
            return

        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except Exception as e:
            raise ImportError(
                "Moirai2 dependencies are missing. "
                "Install uni2ts and gluonts first, e.g.:\n"
                "pip install gluonts<=0.14.4\n"
                "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git"
            ) from e

        kwargs = {}
        if self.cfg.device is not None:
            kwargs["map_location"] = self.cfg.device

        module = Moirai2Module.from_pretrained(
            self.cfg.model_id,
            **kwargs,
        )

        # Official examples instantiate Moirai2Forecast with prediction_length,
        # context_length, target_dim and feature dims.
        self._forecast_model = Moirai2Forecast(
            module=module,
            prediction_length=self.cfg.prediction_length,
            context_length=self.cfg.win_size,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self._predictor = self._forecast_model.create_predictor(
            batch_size=self.cfg.batch_size
        )

    def _score_array(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        channel_scores = np.zeros((n, d), dtype=np.float32)

        for dim in range(d):
            channel_scores[:, dim] = self._score_univariate(X[:, dim])

        # Repo expects one anomaly score per timestamp.
        score = np.mean(channel_scores, axis=1)

        # Avoid all-zero prefix: pad with first valid score.
        first_valid = min(self.cfg.win_size, len(score) - 1)
        if first_valid > 0:
            score[:first_valid] = score[first_valid]

        return score.astype(np.float32)

    def _score_univariate(self, series: np.ndarray) -> np.ndarray:
        from gluonts.dataset.common import ListDataset

        n = len(series)
        H = self.cfg.prediction_length
        W = self.cfg.win_size
        agg = np.zeros(n, dtype=np.float32)
        cnt = np.zeros(n, dtype=np.float32)

        start = pd.Period("2000-01-01 00:00", freq=self.cfg.freq)

        # rolling zero-shot forecast
        for t in range(W, n - H + 1):
            context = series[t - W : t].astype(np.float32)
            future = series[t : t + H].astype(np.float32)

            ds = ListDataset(
                [{"start": start, "target": context}],
                freq=self.cfg.freq,
            )

            try:
                forecast = next(self._predictor.predict(ds))
            except Exception as e:
                raise RuntimeError(
                    f"Moirai2 predictor failed at step t={t}. "
                    f"Check uni2ts/gluonts compatibility and model download."
                ) from e

            median = self._extract_quantile_or_mean(forecast, 0.5, H)
            # Use a plain forecast-error score: anomaly score is the squared
            # deviation from the point forecast at each predicted timestamp.
            step_score = np.square(future - median)

            idx = np.arange(t, t + H)
            agg[idx] += step_score.astype(np.float32)
            cnt[idx] += 1.0

        valid = cnt > 0
        score = np.zeros(n, dtype=np.float32)
        score[valid] = agg[valid] / cnt[valid]

        # If for some reason tail positions are uncovered, extend last valid value.
        if np.any(valid):
            last_val = score[np.where(valid)[0][-1]]
            score[~valid] = last_val

        return score

    @staticmethod
    def _extract_quantile_or_mean(forecast, q: float, H: int, fallback=None):
        arr = None

        # GluonTS QuantileForecast usually supports .quantile(q)
        try:
            arr = forecast.quantile(q)
        except Exception:
            arr = None

        # some forecast objects expose .mean
        if arr is None and abs(q - 0.5) < 1e-9:
            try:
                arr = forecast.mean
            except Exception:
                arr = None

        if arr is None and fallback is not None:
            arr = fallback

        if arr is None:
            raise RuntimeError(
                "Could not extract forecast quantile/mean from Moirai2 output."
            )

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if len(arr) < H:
            # defensive padding
            pad_val = arr[-1] if len(arr) else 0.0
            arr = np.pad(arr, (0, H - len(arr)), constant_values=pad_val)

        return arr[:H]

    def _postprocess_after_fit(self):
        # BaseDetector in this repo looks pyod-like; if helper exists, use it.
        if hasattr(self, "_process_decision_scores"):
            try:
                self._process_decision_scores()
                return
            except Exception:
                pass

        # fallback
        scores = np.asarray(self.decision_scores_, dtype=np.float32)
        q = 100 * (1.0 - self.contamination)
        self.threshold_ = float(np.percentile(scores, q))
        self.labels_ = (scores > self.threshold_).astype(int)

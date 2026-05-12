import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from numpy.lib.stride_tricks import sliding_window_view


class Chronos2Fast:
    def __init__(
        self,
        context_len=256,
        pred_len=1,
        stride=1,
        batch_size=64,
        device="cuda",
        model_name="amazon/chronos-2",
    ):
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.batch_size = batch_size
        self.device = device

        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    def _build_df(self, windows, start_timestamp="2000-01-01"):
        n_windows, context_len = windows.shape
        if n_windows == 0:
            return pd.DataFrame({"id": [], "timestamp": [], "target": []})

        ts = pd.date_range(
            start=start_timestamp,
            periods=context_len,
            freq="D",
        ).values

        ids = np.repeat(np.arange(n_windows, dtype=np.int64), context_len)
        timestamps = np.tile(ts, n_windows)
        targets = windows.reshape(-1).astype(np.float32, copy=False)

        return pd.DataFrame(
            {
                "id": ids,
                "timestamp": timestamps,
                "target": targets,
            }
        )

    @torch.no_grad()
    def score(self, values):
        values = np.asarray(values, dtype=np.float32).ravel()

        total_len = self.context_len + self.pred_len
        if len(values) < total_len:
            return np.array([], dtype=np.float32)

        # Shape: (n_windows, context_len + pred_len)
        # Each row corresponds to values[s : s + total_len], where s goes from 0..len-total_len.
        windows_full = sliding_window_view(values, window_shape=total_len)
        windows_full = windows_full[:: self.stride]

        windows_all = windows_full[:, : self.context_len]
        gts_all = windows_full[:, self.context_len :]

        all_scores = []

        for start in range(0, len(windows_all), self.batch_size):
            batch_windows = windows_all[start : start + self.batch_size]
            batch_gts = gts_all[start : start + self.batch_size]

            df = self._build_df(batch_windows)

            # `predict_df` can be surprisingly expensive when input validation is enabled.
            # Newer chronos-forecasting versions expose `validate_inputs`; keep a backward-compatible fallback.
            try:
                pred_df = self.pipeline.predict_df(
                    df,
                    prediction_length=self.pred_len,
                    target="target",
                    timestamp_column="timestamp",
                    id_column="id",
                    quantile_levels=[0.5],
                    validate_inputs=False,
                )
            except TypeError:
                pred_df = self.pipeline.predict_df(
                    df,
                    prediction_length=self.pred_len,
                    target="target",
                    timestamp_column="timestamp",
                    id_column="id",
                    quantile_levels=[0.5],
                )

            preds = pred_df["0.5"].values.reshape(-1, self.pred_len)

            scores = ((preds - batch_gts) ** 2).mean(axis=1)
            all_scores.append(scores)

        return np.concatenate(all_scores).astype(np.float32)

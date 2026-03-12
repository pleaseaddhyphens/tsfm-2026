from chronos import Chronos2Pipeline
import numpy as np
import pandas as pd
import torch


class Chronos2AnomalyDetector:

    def __init__(
        self,
        context_len=256,
        pred_len=1,
        stride=1,
        device="cpu",
        batch_size=256,
        probabilistic=True,
    ):

        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.batch_size = batch_size
        self.probabilistic = probabilistic

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    def _windows(self, values):

        windows = []
        gts = []

        for t in range(self.context_len, len(values) - self.pred_len + 1, self.stride):

            windows.append(values[t-self.context_len:t])
            gts.append(values[t:t+self.pred_len])

        return np.array(windows), np.array(gts)

    def _build_df(self, windows):

        rows = []

        for i, w in enumerate(windows):

            ts = pd.date_range(
                start="2000-01-01",
                periods=len(w),
                freq="D"
            )

            for t, v in zip(ts, w):

                rows.append({
                    "id": f"series_{i}",
                    "timestamp": t,
                    "target": v
                })

        return pd.DataFrame(rows)

    def score(self, values):

        windows, gts = self._windows(values)

        preds = []
        q_low = []
        q_high = []

        for i in range(0, len(windows), self.batch_size):

            batch = windows[i:i+self.batch_size]

            df = self._build_df(batch)

            pred_df = self.pipeline.predict_df(
                df,
                prediction_length=self.pred_len,
                target="target",
                timestamp_column="timestamp",
                id_column="id",
                quantile_levels=[0.1, 0.5, 0.9]
            )

            preds.append(pred_df["0.5"].values)
            q_low.append(pred_df["0.1"].values)
            q_high.append(pred_df["0.9"].values)

        preds = np.concatenate(preds).reshape(-1, self.pred_len)
        q_low = np.concatenate(q_low).reshape(-1, self.pred_len)
        q_high = np.concatenate(q_high).reshape(-1, self.pred_len)

        y_true = gts

        if self.probabilistic:

            upper = np.maximum(0, y_true - q_high)
            lower = np.maximum(0, q_low - y_true)

            scores = (upper + lower) ** 2

        else:

            scores = (preds - y_true) ** 2

        return scores.mean(axis=1)
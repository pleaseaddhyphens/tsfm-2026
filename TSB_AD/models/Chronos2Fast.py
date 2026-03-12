import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline


class Chronos2Fast:

    def __init__(self, context_len=256, pred_len=1, stride=1, device="cpu"):

        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device
        )

    def _build_batch_df(self, windows):

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

        windows = []
        gts = []

        for t in range(self.context_len, len(values) - self.pred_len + 1, self.stride):

            windows.append(values[t-self.context_len:t])
            gts.append(values[t:t+self.pred_len])

        windows = np.array(windows)
        gts = np.array(gts)

        context_df = self._build_batch_df(windows)

        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=self.pred_len,
            target="target",
            timestamp_column="timestamp",
            id_column="id",
            quantile_levels=[0.5]
        )

        preds = pred_df["0.5"].values.reshape(-1, self.pred_len)

        scores = ((preds - gts) ** 2).mean(axis=1)

        return scores
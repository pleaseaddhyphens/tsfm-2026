from chronos import Chronos2Pipeline
import numpy as np
import pandas as pd

from .base import BaseDetector


class Chronos2Detector(BaseDetector):

    def __init__(
        self,
        win_size=256,
        prediction_length=1,
        input_c=1,
        stride=1,
        device="cpu",
        model_name="amazon/chronos-2",
    ):

        self.model_name = "Chronos2"
        self.win_size = win_size
        self.prediction_length = prediction_length
        self.input_c = input_c
        self.stride = stride
        self.device = device

        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device
        )

        self.score_list = []

    def fit(self, data):

        for channel in range(self.input_c):

            values = data[:, channel]

            scores = self._rolling_scores(values)

            self.score_list.append(scores)

        scores_merge = np.mean(np.array(self.score_list), axis=0)

        padded_scores = np.zeros(len(data))
        padded_scores[: self.win_size + self.prediction_length - 1] = scores_merge[0]
        padded_scores[self.win_size + self.prediction_length - 1:] = scores_merge

        self.decision_scores_ = padded_scores

    def decision_function(self, X):
        pass

    def _build_df(self, values, start_timestamp="2000-01-01"):

        return pd.DataFrame({
            "id": "series_0",
            "timestamp": pd.date_range(
                start=start_timestamp,
                periods=len(values),
                freq="D"
            ),
            "target": values
        })

    def _rolling_scores(self, values):

        preds = []
        gts = []

        for t in range(self.win_size, len(values) - self.prediction_length + 1, self.stride):

            context = values[t - self.win_size:t]

            context_df = self._build_df(context)

            pred_df = self.pipeline.predict_df(
                context_df,
                prediction_length=self.prediction_length,
                target="target",
                timestamp_column="timestamp",
                id_column="id",
                quantile_levels=[0.5]
            )

            y_pred = pred_df["0.5"].values[: self.prediction_length]
            y_true = values[t:t + self.prediction_length]

            mse = ((y_pred - y_true) ** 2).mean()

            preds.append(y_pred.mean())
            gts.append(y_true.mean())

        scores = (np.array(preds) - np.array(gts)) ** 2

        return scores
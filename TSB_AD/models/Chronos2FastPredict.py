import numpy as np
import torch
from chronos import Chronos2Pipeline


class Chronos2FastPredict:
    """
    Chronos2 anomaly scorer using `Chronos2Pipeline.predict()` (no pandas).

    Computes an anomaly score per time step as MSE between the forecast median and
    the ground truth over rolling windows.

    Notes for chronos==2.2.2:
    - Pass CPU tensors into `predict()` to avoid internal `.pin_memory()` errors.
    - `predict()` returns `list[Tensor]` with shapes that may vary:
      (pred_len,), (num_samples, pred_len), or (pred_len, num_samples).
    """

    def __init__(
        self,
        context_len=256,
        pred_len=1,
        stride=1,
        batch_size=256,
        device="cuda",
        model_name="amazon/chronos-2",
        **predict_kwargs,
    ):
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.device = device
        self.predict_kwargs = dict(predict_kwargs)

        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    def _forecast_to_pred(self, forecast) -> torch.Tensor:
        f = forecast
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        if not torch.is_tensor(f):
            raise TypeError(f"Unexpected forecast type: {type(f)}")

        # Some chronos versions return a time-axis forecast including the context:
        # (context_len + pred_len, n_quantiles, n_variates). For univariate: (..., 21, 1).
        if f.ndim == 3 and f.shape[0] == self.context_len + self.pred_len:
            n_quantiles = int(f.shape[1])
            q_idx = n_quantiles // 2
            # Take only the prediction part and flatten to (pred_len,)
            pred_part = f[-self.pred_len :, q_idx, :]  # (pred_len, n_variates)
            return pred_part.reshape(-1)[: self.pred_len]

        # Chronos2Pipeline.predict typically returns quantile forecasts:
        # (n_variates, n_quantiles, prediction_length).
        # For univariate this is often (1, n_quantiles, pred_len).
        if f.ndim == 3 and f.shape[-1] == self.pred_len:
            # Select the median quantile (0.5). With 21 quantiles, the middle index is 10.
            # We avoid relying on quantile level metadata for robustness.
            n_quantiles = int(f.shape[1])
            q_idx = n_quantiles // 2
            # (1, pred_len) -> (pred_len,)
            return f[:, q_idx, :].reshape(-1)

        # Drop a singleton variate dim if present: (1, S, L) -> (S, L)
        if f.ndim == 3 and f.shape[0] == 1:
            f = f[0]

        if f.ndim == 1:
            return f

        if f.ndim != 2:
            raise ValueError(f"Unexpected forecast ndim={f.ndim} shape={tuple(f.shape)}")

        # Backward/alternate cases (older APIs): samples layouts
        # - (num_samples, pred_len)
        # - (pred_len, num_samples)
        if f.shape[1] == self.pred_len:
            return f.median(dim=0).values
        if f.shape[0] == self.pred_len:
            return f.median(dim=1).values

        # Fallback: median over samples dim=0, then crop if needed.
        p = f.median(dim=0).values
        if p.numel() != self.pred_len:
            p = p.reshape(-1)[: self.pred_len]
        return p

    @torch.no_grad()
    def score(self, values):
        values = np.asarray(values, dtype=np.float32).ravel()
        total_len = self.context_len + self.pred_len
        if len(values) < total_len:
            return np.array([], dtype=np.float32)

        series = torch.as_tensor(values, device="cpu", dtype=torch.float32)
        windows_full = series.unfold(0, total_len, self.stride)  # (B, total_len)
        context = windows_full[:, : self.context_len].unsqueeze(1)  # (B, 1, context_len)
        gt = windows_full[:, self.context_len :]  # (B, pred_len)

        all_scores = []
        for start in range(0, context.shape[0], self.batch_size):
            batch_context = context[start : start + self.batch_size]
            batch_gt = gt[start : start + self.batch_size]

            forecasts = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                batch_size=self.batch_size,
                context_length=self.context_len,
                **self.predict_kwargs,
            )

            preds = [self._forecast_to_pred(f).to("cpu") for f in forecasts]
            pred = torch.stack(preds, dim=0)  # (B, pred_len)
            scores = (pred - batch_gt).pow(2).mean(dim=1)  # (B,)
            all_scores.append(scores)

        return torch.cat(all_scores, dim=0).to(dtype=torch.float32).numpy()


class Chronos2FastPredictMV:
    """
    Multivariate Chronos2 scorer using `Chronos2Pipeline.predict()`.

    Input: values_2d shape (T, D).
    Output: score per time step, aggregated across channels by `agg` ("mean" or "max").
    """

    def __init__(
        self,
        context_len=256,
        pred_len=1,
        stride=1,
        batch_size=256,
        device="cuda",
        model_name="amazon/chronos-2",
        agg="mean",
        **predict_kwargs,
    ):
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.device = device
        self.agg = str(agg).lower()
        if self.agg not in {"mean", "max"}:
            raise ValueError("agg must be 'mean' or 'max'")

        self.predict_kwargs = dict(predict_kwargs)

        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    def _forecast_to_pred_mv(self, forecast, n_variates: int) -> torch.Tensor:
        f = forecast
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        if not torch.is_tensor(f):
            raise TypeError(f"Unexpected forecast type: {type(f)}")

        # Format A (observed in chronos==2.2.2):
        # (context_len + pred_len, n_quantiles, n_variates_out)
        # Example: (129, 21, 1) for context_len=128, pred_len=1.
        if f.ndim == 3 and f.shape[0] == self.context_len + self.pred_len:
            n_quantiles = int(f.shape[1])
            q_idx = n_quantiles // 2  # median quantile
            pred_part = f[-self.pred_len :, q_idx, :]  # (pred_len, n_variates_out)
            pred = pred_part.transpose(0, 1).contiguous()  # (n_variates_out, pred_len)

            # If the model returns fewer variates than the input, broadcast conservatively.
            if pred.shape[0] == 1 and n_variates > 1:
                pred = pred.repeat(n_variates, 1)
            if pred.shape[0] > n_variates:
                pred = pred[:n_variates]
            return pred

        # Format B:
        # (n_variates_out, n_quantiles, pred_len)
        if f.ndim == 3 and f.shape[-1] == self.pred_len:
            if f.shape[0] not in (n_variates, n_variates + 1, 1):
                raise ValueError(
                    f"Cannot infer variate axis: forecast shape={tuple(f.shape)} n_variates={n_variates}"
                )
            n_quantiles = int(f.shape[1])
            q_idx = n_quantiles // 2
            pred = f[:, q_idx, :]  # (n_variates_out, pred_len)

            if pred.shape[0] == 1 and n_variates > 1:
                pred = pred.repeat(n_variates, 1)
            elif pred.shape[0] == n_variates + 1:
                pred = pred[:n_variates]
            return pred

        # Deterministic: (D, L) or (L, D)
        if f.ndim == 2:
            if f.shape == (n_variates, self.pred_len):
                return f
            if f.shape == (self.pred_len, n_variates):
                return f.transpose(0, 1)
            raise ValueError(f"Unexpected deterministic forecast shape={tuple(f.shape)}")

        raise ValueError(f"Unexpected forecast ndim={f.ndim} shape={tuple(f.shape)}")

    @torch.no_grad()
    def score(self, values_2d):
        values = np.asarray(values_2d, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values_2d must have shape (T, D)")
        t_len, n_variates = values.shape

        total_len = self.context_len + self.pred_len
        if t_len < total_len:
            return np.array([], dtype=np.float32)

        series = torch.as_tensor(values, device="cpu", dtype=torch.float32)  # (T, D)
        # For a 2D tensor (T, D), `unfold(0, ...)` returns shape (B, D, total_len)
        # (it appends the window dimension at the end).
        windows_full = series.unfold(0, total_len, self.stride).contiguous()  # (B, D, total_len)

        context = windows_full[:, :, : self.context_len]  # (B, D, context_len)
        gt = windows_full[:, :, self.context_len :]  # (B, D, pred_len)

        all_scores = []
        for start in range(0, context.shape[0], self.batch_size):
            batch_context = context[start : start + self.batch_size]
            batch_gt = gt[start : start + self.batch_size]

            forecasts = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                batch_size=self.batch_size,
                context_length=self.context_len,
                **self.predict_kwargs,
            )

            preds = [self._forecast_to_pred_mv(f, n_variates).to("cpu") for f in forecasts]
            pred = torch.stack(preds, dim=0)  # (B, D, pred_len)
            per_var = (pred - batch_gt).pow(2).mean(dim=2)  # (B, D)
            if self.agg == "mean":
                score = per_var.mean(dim=1)
            else:
                score = per_var.max(dim=1).values
            all_scores.append(score)

        return torch.cat(all_scores, dim=0).to(dtype=torch.float32).numpy()

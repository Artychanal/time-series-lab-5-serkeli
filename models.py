from dataclasses import dataclass
import warnings
import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class Spec:
    family: str
    params: tuple = ()

    @property
    def name(self):
        return self.family + ("(" + ",".join(map(str, self.params)) + ")" if self.params else "")


def candidates():
    result = [Spec("Naive"), Spec("SeasonalNaive", (12,))]
    result += [Spec("SMA", (w,)) for w in (2, 3, 6, 9, 12, 18)]
    for family, d in (("ARMA", 0), ("ARIMA", 1)):
        result += [Spec(family, (p, d, q)) for p in range(3) for q in range(3)
                   if 0 < p + q <= 3]
    result += [Spec("Ridge", (lag, alpha)) for lag in (3, 6, 12) for alpha in (0.1, 1.0, 10.0)]
    result += [Spec("Polynomial", (degree,)) for degree in (1, 2, 3)]
    return result


def defaults():
    return [Spec("SMA", (3,)), Spec("ARMA", (1, 0, 1)), Spec("ARIMA", (1, 1, 1)),
            Spec("Ridge", (6, 1.0)), Spec("Polynomial", (1,))]


def fit_predict(values, spec, horizon):
    started = time.perf_counter()
    y = np.asarray(values, float)
    if y.ndim != 1 or len(y) < 8 or not np.isfinite(y).all() or horizon < 1:
        raise ValueError("Finite 1D history and positive horizon required")
    info = {"converged": True, "warnings": "", "aic": np.nan, "bic": np.nan}
    family, params = spec.family, spec.params
    variance = None
    if family in ("ARMA", "ARIMA"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ARIMA(y, order=params, trend="c" if params[1] == 0 else "n").fit(
                method_kwargs={"maxiter": 200})
            prediction = model.get_forecast(horizon)
            point = np.asarray(prediction.predicted_mean)
            variance = np.asarray(prediction.var_pred_mean)
        info.update(converged=bool(model.mle_retvals.get("converged", True)),
                    warnings=" | ".join(sorted({str(w.message) for w in caught})),
                    aic=float(model.aic), bic=float(model.bic))
    elif family == "Polynomial":
        t = np.arange(len(y)) / (len(y) - 1)
        beta = np.polynomial.polynomial.polyfit(t, y, params[0])
        point = np.polynomial.polynomial.polyval(np.arange(len(y), len(y) + horizon) / (len(y) - 1), beta)
    else:
        history = list(y)
        if family == "Ridge":
            lag, alpha = params
            x = np.array([y[t-lag:t][::-1] for t in range(lag, len(y))])
            target = y[lag:]
            mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
            design = (x - mean) / scale
            beta = np.linalg.solve(design.T @ design + alpha * np.eye(lag),
                                   design.T @ (target - target.mean()))
        for _ in range(horizon):
            if family == "Naive":
                value = history[-1]
            elif family == "SeasonalNaive":
                value = history[-params[0]]
            elif family == "SMA":
                value = np.mean(history[-params[0]:])
            elif family == "Ridge":
                value = target.mean() + ((np.array(history[-lag:][::-1]) - mean) / scale) @ beta
            else:
                raise ValueError(f"Unknown model {family}")
            history.append(float(value))
        point = np.array(history[len(y):])
    if not np.isfinite(point).all():
        raise FloatingPointError("Nonfinite prediction")
    info["fit_forecast_seconds"] = time.perf_counter() - started
    return point, variance, info


def original(values):
    values = np.asarray(values, float)
    if np.max(values) > 700:
        raise FloatingPointError("Explosive extrapolation exceeds numerical range")
    return np.maximum(0, np.expm1(values))

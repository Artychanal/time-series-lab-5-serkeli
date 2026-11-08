import numpy as np
import pandas as pd

from models import fit_predict, original


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    return {"n": len(error), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "smape": float(200 * np.mean(np.divide(np.abs(error), denominator,
                         out=np.zeros_like(error), where=denominator > 0)))}


def paired_metrics(actual, predicted, unit="reviews"):
    return {**{k + "_log": v for k, v in metrics(actual, predicted).items()},
            **{k + "_" + unit: v for k, v in metrics(original(actual), original(predicted)).items()}}


def validation(y, specs, origins=range(36, 55, 3), stop=57):
    rows, predictions, failures = [], [], []
    for spec in specs:
        current, errors = [], []
        for origin in origins:
            horizon = min(6, stop-origin)
            try:
                point, _, info = fit_predict(y[:origin], spec, horizon)
                if not info["converged"]:
                    raise ValueError("Optimizer did not converge")
                for h in (1, 3, 6):
                    if h <= horizon:
                        current.append({"model": spec.name, "family": spec.family, "origin": origin,
                                        "horizon": h, "actual_log": y[origin+h-1], "predicted_log": point[h-1]})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                errors.append({"model": spec.name, "origin": origin, "error": str(exc)})
        failures.extend(errors)
        for h in (1, 3, 6):
            group = [r for r in current if r["horizon"] == h]
            if group:
                m = paired_metrics([r["actual_log"] for r in group], [r["predicted_log"] for r in group])
                rows.append({"model": spec.name, "family": spec.family, "horizon": h,
                             "eligible": not errors, **m})
        predictions.extend(current)
        print("  validation", spec.name, "ok" if not errors else "excluded", flush=True)
    frame = pd.DataFrame(rows)
    ranking = frame.groupby(["model", "family"], as_index=False).agg(
        score=("rmse_log", "mean"), eligible=("eligible", "all"), horizons=("horizon", "nunique"))
    ranking = ranking.loc[ranking.eligible & (ranking.horizons == 3)].sort_values(["score", "model"])
    lookup = {s.name: s for s in specs}
    selected = [lookup[g.iloc[0].model] for _, g in ranking.groupby("family", sort=True)]
    return frame, ranking, selected, pd.DataFrame(predictions), pd.DataFrame(failures)


def backtest(y, labelled_specs, start=57):
    rows, status = [], []
    for label, spec in labelled_specs.items():
        for origin in range(start, len(y)):
            hmax = min(12, len(y)-origin)
            try:
                pred, var, info = fit_predict(y[:origin], spec, hmax)
                status.append({"model": label, "origin": origin, **info})
                if not info["converged"]:
                    continue
                for h in range(1, hmax+1):
                    rows.append({"model": label, "spec": spec.name, "origin": origin, "horizon": h,
                                 "actual_log": y[origin+h-1], "predicted_log": pred[h-1],
                                 "variance_log": var[h-1] if var is not None else np.nan})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                status.append({"model": label, "origin": origin, "converged": False, "warnings": str(exc)})
    predictions = pd.DataFrame(rows)
    summaries = []
    for label, g in predictions.groupby("model"):
        for mode, subset in (("fixed_12", g.loc[g.origin == start]),
                             ("rolling_1", g.loc[g.horizon == 1])):
            if len(subset) == 12:
                summaries.append({"model": label, "mode": mode,
                                  **paired_metrics(subset.actual_log, subset.predicted_log)})
    by_horizon = [{"model": label, "horizon": h,
                   **paired_metrics(g.actual_log, g.predicted_log)}
                  for (label, h), g in predictions.groupby(["model", "horizon"])]
    return predictions, pd.DataFrame(summaries), pd.DataFrame(by_horizon), pd.DataFrame(status)


def interval_scores(actual, lower, upper, alpha=0.05):
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    score = upper-lower + 2/alpha * np.maximum(lower-actual, 0) + 2/alpha * np.maximum(actual-upper, 0)
    return {"coverage": float(np.mean((lower <= actual) & (actual <= upper))),
            "mean_width": float(np.mean(upper-lower)), "interval_score": float(np.mean(score))}

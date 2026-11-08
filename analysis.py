from pathlib import Path
import json
import time
import warnings
import platform
import statsmodels
import scipy

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

from data_loader import load_steam, load_sales
from models import candidates, defaults, fit_predict, original
from evaluation import validation, backtest, interval_scores, paired_metrics
from synthetic_data import experiment
from visualization import plots


def properties(y):
    rows = []
    for name, values in (("log_levels", y), ("log_differences", np.diff(y))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adf = adfuller(values, regression="c", autolag="AIC", maxlag=6)
            kp = kpss(values, regression="c", nlags="auto")
        rows.append({"series": name, "n": len(values), "mean": np.mean(values),
                     "std": np.std(values, ddof=1), "adf_stat": adf[0], "adf_p": adf[1],
                     "kpss_stat": kp[0], "kpss_p": kp[1],
                     "warnings": " | ".join(str(w.message) for w in caught)})
    return pd.DataFrame(rows)


def sales_analysis(daily, output):
    y = np.log1p(daily.revenue.to_numpy(float))
    def predict(history, method, horizon):
        if method == "Linear":
            coef = np.polyfit(np.arange(len(history)), history, 1)
            return np.polyval(coef, np.arange(len(history), len(history)+horizon))
        values = list(history)
        w = 1 if method == "Naive" else int(method[-1])
        for _ in range(horizon):
            values.append(float(np.mean(values[-w:])))
        return np.array(values[len(history):])
    val = []
    for method in ("SMA2", "SMA3"):
        actual = y[4:6]
        pred = [predict(y[:o], method, 1)[0] for o in (4, 5)]
        val.append({"model": method, **paired_metrics(actual, pred, "revenue")})
    val = pd.DataFrame(val)
    winner = val.sort_values("rmse_log").iloc[0].model
    rows, forecasts = [], []
    for method in ("Naive", "SMA2", "SMA3", "Linear"):
        pred = predict(y[:6], method, 3)
        rows.append({"model": method, "selected_sma": method == winner,
                     **paired_metrics(y[6:], pred, "revenue")})
        future = predict(y, method, 18)
        for h, value in enumerate(future, 1):
            forecasts.append({"model": method, "future_observed_date_index": h,
                              "forecast_revenue": original([value])[0], "forecast_log": value,
                              "calendar_date": "unknown"})
    val.to_csv(output / "sales_validation.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "sales_test.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output / "sales_conditional_extrapolation.csv", index=False)
    daily.to_csv(output / "sales_observed_dates.csv")
    return {"selected": winner, "validation_points": 2, "test_points": 3,
            "horizons_observed_dates": [5, 9, 14, 18],
            "interpretation": "Conditional forecast of revenue on the next observed dates, not calendar-day sales",
            "arma_arima": "Not estimated: nine irregular dates cannot support reliable identification"}


def run(data_dir, output_dir, simulations, seed):
    start_time = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    def save(frame, filename):
        frame.to_csv(output / filename, index=False)
    steam, steam_info = load_steam(Path(data_dir) / "cyberpunk_2077_monthly_reviews.csv")
    _, daily, sales_info = load_sales(Path(data_dir) / "dataset8.csv")
    y = np.log1p(steam.total_reviews.to_numpy(float))
    save(steam, "steam_observed.csv")
    save(properties(y), "series_properties.csv")
    save(pd.DataFrame({"lag": np.arange(19), "acf": acf(y, nlags=18),
                       "pacf": pacf(y, nlags=18, method="ywm")}), "acf_pacf.csv")
    smooth = steam[["date", "total_reviews"]].copy()
    for w in (3, 6, 12):
        smooth[f"causal_log_sma_{w}"] = pd.Series(y).rolling(w, min_periods=w).mean()
    save(smooth, "causal_smoothing.csv")
    print("Chronological validation", flush=True)
    scores, ranking, selected, val_predictions, failures = validation(y, candidates())
    save(scores, "validation_scores.csv")
    save(ranking, "validation_ranking.csv")
    save(val_predictions, "validation_predictions.csv")
    save(failures, "validation_failures.csv")
    labelled = {s.family: s for s in selected}
    labelled.update({s.family + "_default": s for s in defaults()})
    print("Independent final backtest", flush=True)
    predictions, comparison, by_horizon, fit_status = backtest(y, labelled)
    save(predictions, "test_predictions.csv")
    save(comparison, "test_comparison.csv")
    save(by_horizon, "test_by_horizon.csv")
    save(fit_status, "fit_status.csv")
    save(fit_status.groupby("model", as_index=False).agg(
        median_seconds=("fit_forecast_seconds", "median"), fits=("origin", "count")), "runtime_comparison.csv")
    gain_rows = []
    for mode in ("fixed_12", "rolling_1"):
        part = comparison.loc[comparison["mode"] == mode].set_index("model")
        for spec in defaults():
            a, b = spec.family + "_default", spec.family
            if a in part.index and b in part.index:
                gain_rows.append({"family": b, "mode": mode, "default_rmse": part.loc[a, "rmse_log"],
                                  "tuned_rmse": part.loc[b, "rmse_log"],
                                  "improvement_percent": 100*(1-part.loc[b, "rmse_log"]/part.loc[a, "rmse_log"])})
    save(pd.DataFrame(gain_rows), "optimization_effect.csv")
    intervals = []
    for (label, mode), g in (( (label, mode),
                              part.loc[part.origin == 57] if mode == "fixed_12" else part.loc[part.horizon == 1])
                             for label, part in predictions.groupby("model") for mode in ("fixed_12", "rolling_1")):
        if len(g) == 12 and g.variance_log.notna().all():
            sd = np.sqrt(g.variance_log.to_numpy())
            lo, hi = g.predicted_log.to_numpy()-1.95996398454*sd, g.predicted_log.to_numpy()+1.95996398454*sd
            intervals.append({"model": label, "mode": mode, "n": len(g),
                              **interval_scores(g.actual_log, lo, hi)})
    save(pd.DataFrame(intervals), "test_interval_scores.csv")
    future_rows, diagnostic_rows = [], []
    future_dates = pd.date_range(steam.date.iloc[-1] + pd.offsets.MonthBegin(), periods=138, freq="MS")
    for spec in selected:
        pred, var, info = fit_predict(y, spec, 138)
        if not info["converged"]:
            raise ValueError(f"Final model failed: {spec.name}")
        for h, value in enumerate(pred, 1):
            sd = np.sqrt(var[h-1]) if var is not None else np.nan
            future_rows.append({"model": spec.family, "spec": spec.name, "date": future_dates[h-1], "horizon": h,
                                "forecast_log": value, "forecast_reviews_median": original([value])[0],
                                "lower_95": original([value-1.95996398454*sd])[0] if var is not None else np.nan,
                                "upper_95": original([value+1.95996398454*sd])[0] if var is not None else np.nan})
        part = predictions.loc[(predictions.model == spec.family) & (predictions.horizon == 1)]
        residual = part.actual_log.to_numpy()-part.predicted_log.to_numpy()
        if len(residual) == 12:
            lag = 3
            lb = acorr_ljungbox(residual, lags=[lag], model_df=0, return_df=True)
            diagnostic_rows.append({"model": spec.family, "n": 12, "ljung_box_lag": lag,
                                    "ljung_box_p": lb.lb_pvalue.iloc[0], "bias_log": -residual.mean(), **info})
    future = pd.DataFrame(future_rows)
    save(future, "extrapolation_all.csv")
    for ratio, horizon in (("0_5", 35), ("1_0", 69), ("1_5", 104), ("2_0", 138)):
        save(future.loc[future.horizon <= horizon], f"extrapolation_{ratio}.csv")
    save(pd.DataFrame(diagnostic_rows), "forecast_diagnostics.csv")
    sales_result = sales_analysis(daily, output)
    print("Monte Carlo transfer study", flush=True)
    individual, ensemble, sim_failures, examples = experiment(selected, simulations, seed)
    save(individual, "synthetic_individual.csv")
    save(ensemble, "synthetic_summary.csv")
    save(sim_failures, "synthetic_failures.csv")
    pivot = individual.pivot(index=["scenario", "replication", "ratio"], columns="model", values="rmse")
    paired = []
    for (scenario, ratio), group in pivot.groupby(level=["scenario", "ratio"]):
        diff = (group.SMA_default-group.SMA).dropna()
        margin = 1.96*diff.std(ddof=1)/np.sqrt(len(diff))
        paired.append({"scenario": scenario, "ratio": ratio, "pairs": len(diff),
                       "default_minus_tuned_rmse": diff.mean(),
                       "mc_ci_low": diff.mean()-margin, "mc_ci_high": diff.mean()+margin})
    save(pd.DataFrame(paired), "synthetic_optimization_pairs.csv")
    best_name = ranking.iloc[0].model
    best = next(s for s in selected if s.name == best_name)
    checks = {"steam_rows_69": len(y) == 69, "sales_dates_9": len(daily) == 9,
              "validation_before_test": bool((val_predictions.origin + val_predictions.horizon <= 57).all()),
              "test_origin_not_earlier_57": bool((predictions.origin >= 57).all()),
              "all_final_models_converged": len(future) == len(selected)*138,
              "all_forecasts_finite_nonnegative": bool(np.isfinite(future.forecast_reviews_median).all()
                                                           and (future.forecast_reviews_median >= 0).all())}
    summary = {"seed": seed, "simulations_per_scenario": simulations,
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "statsmodels": statsmodels.__version__},
               "synthetic_scenarios": 3, "data_quality": {"steam": steam_info, "sales": sales_info},
               "protocol": {"initial_train": 36, "validation_stop": 57, "test_points": 12,
                            "validation_origins": list(range(36, 55, 3)), "validation_horizons": [1, 3, 6],
                            "selection": "Mean of per-horizon log RMSE; no test data in selection",
                            "extrapolation_convention": "ceil(ratio * 69 monthly observations)",
                            "point_forecast": "Nonnegative inverse-log location, interpreted as conditional median under Gaussian log errors",
                            "intervals": "ARMA/ARIMA Gaussian predictive intervals conditional on estimated parameters"},
               "selected": [{"family": s.family, "params": list(s.params), "name": s.name} for s in selected],
               "validation_winner": best.name, "sales": sales_result,
               "verification": checks, "elapsed_seconds": time.perf_counter()-start_time}
    plots(steam, smooth, scores, comparison, by_horizon, future, ensemble, daily, predictions, output)
    summary["figures"] = len(list((output / "figures").glob("*.png")))
    if not all(checks.values()):
        raise AssertionError(checks)
    (output / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete. Validation winner: {best.name}. Results: {output}", flush=True)

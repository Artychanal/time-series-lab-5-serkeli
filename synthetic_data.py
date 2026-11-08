import numpy as np
import pandas as pd

from models import Spec, fit_predict
from evaluation import metrics


def generate(scenario, seed, n=207):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.22, n+150)
    e = np.zeros(n+150)
    for t in range(1, len(e)):
        e[t] = 0.65*e[t-1] + noise[t] + 0.3*noise[t-1]
    t = np.arange(n)
    if scenario == "stationary_arma":
        return 9 + e[150:]
    if scenario == "smooth_trend":
        return 8.5 + 0.008*t + e[150:]
    if scenario == "future_break":
        return 9 + e[150:] + (t >= 81)*0.9
    raise ValueError(scenario)


def experiment(specs, simulations, seed, progress=print):
    rows, failures, examples = [], [], {}
    used = {s.family: s for s in specs}
    used["SMA_default"] = Spec("SMA", (3,))
    for index, scenario in enumerate(("stationary_arma", "smooth_trend", "future_break")):
        for replication in range(simulations):
            y = generate(scenario, seed + index*100000 + replication)
            for label, spec in used.items():
                try:
                    pred, var, info = fit_predict(y[:69], spec, 138)
                    if not info["converged"]:
                        raise ValueError("Optimizer did not converge")
                    for ratio, horizon in ((0.5, 35), (1., 69), (1.5, 104), (2., 138)):
                        row = {"scenario": scenario, "replication": replication, "model": label,
                               "ratio": ratio, "horizon": horizon, **metrics(y[69:69+horizon], pred[:horizon])}
                        if var is not None:
                            row["coverage_95"] = float(np.mean(np.abs(y[69:69+horizon]-pred[:horizon]) <=
                                                                  1.95996398454*np.sqrt(var[:horizon])))
                        rows.append(row)
                    if replication == 0:
                        examples[(scenario, label)] = (y, pred)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    failures.append({"scenario": scenario, "replication": replication,
                                     "model": label, "error": str(exc)})
            if (replication+1) % 25 == 0:
                progress(f"  synthetic {scenario}: {replication+1}/{simulations}", flush=True)
    individual = pd.DataFrame(rows)
    summaries = []
    for (scenario, model, ratio), g in individual.groupby(["scenario", "model", "ratio"]):
        se = float(g.rmse.std(ddof=1)/np.sqrt(len(g)))
        summaries.append({"scenario": scenario, "model": model, "ratio": ratio,
                          "successful": len(g), "attempted": simulations, "rmse_mean": g.rmse.mean(),
                          "rmse_mc_low": g.rmse.mean()-1.96*se, "rmse_mc_high": g.rmse.mean()+1.96*se,
                          "mae_mean": g.mae.mean(), "bias_mean": g.bias.mean(),
                          "coverage_95": g.get("coverage_95", pd.Series(dtype=float)).mean()})
    return individual, pd.DataFrame(summaries), pd.DataFrame(failures), examples

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plots(steam, smooth, scores, comparison, by_horizon, future, ensemble, daily, predictions, output):
    target = output / "figures"
    target.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(target / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steam.date, steam.total_reviews, color="0.25")
    axes[0].set_ylabel("Відгуки за місяць")
    axes[1].plot(steam.date, np.log1p(steam.total_reviews), color="0.6", label="Спостереження")
    for w in (3, 6, 12):
        axes[1].plot(steam.date, smooth[f"causal_log_sma_{w}"], label=f"SMA {w}")
    axes[1].set_ylabel("ln(1 + кількість)")
    axes[1].legend(ncol=4, fontsize=8)
    finish(fig, "01_data_smoothing.png")
    ac = pd.read_csv(output / "acf_pacf.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col in zip(axes, ("acf", "pacf")):
        ax.stem(ac.lag, ac[col])
        ax.axhline(1.96/np.sqrt(69), color="0.6", ls="--")
        ax.axhline(-1.96/np.sqrt(69), color="0.6", ls="--")
        ax.set(title=col.upper(), xlabel="Лаг, місяців")
    finish(fig, "02_acf_pacf.png")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rank = scores.loc[scores.eligible].groupby("model").rmse_log.mean().sort_values().head(12)
    ax.barh(rank.index[::-1], rank.values[::-1], color="#39758a")
    ax.set_xlabel("Валідаційний score для h = 1, 3, 6")
    ax.set_title("12 найкращих допустимих конфігурацій")
    ax.tick_params(axis="y", labelsize=10)
    finish(fig, "03_validation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ("fixed_12", "rolling_1")):
        part = comparison.loc[comparison["mode"] == mode].sort_values("rmse_log")
        ax.barh(part.model, part.rmse_log, color=["#b6c5cd" if "default" in x else "#39758a" for x in part.model])
        ax.set(title=mode, xlabel="RMSE у log-шкалі")
    finish(fig, "04_final_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, part in by_horizon.groupby("model"):
        if "default" not in label:
            ax.plot(part.horizon, part.rmse_log, marker=".", label=label)
    ax.set(xlabel="Горизонт, місяців", ylabel="RMSE у log-шкалі", title="Число доступних помилок зменшується від 12 до 1")
    ax.legend(ncol=4, fontsize=8)
    finish(fig, "05_error_horizon.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steam.date, np.log1p(steam.total_reviews), color="black", label="Історія")
    for label, part in future.groupby("model"):
        ax.plot(pd.to_datetime(part.date), part.forecast_log, label=label)
    for h in (35, 69, 104, 138):
        ax.axvline(pd.to_datetime(future.loc[future.horizon == h].date.iloc[0]), color="0.7", ls=":")
    ax.set(ylabel="Прогноз у log-шкалі", xlabel="Дата", title="Екстраполяція на 35, 69, 104 та 138 місяців")
    ax.legend(ncol=4, fontsize=8)
    finish(fig, "06_extrapolation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, label in zip(axes, ("ARMA", "ARIMA")):
        part = future.loc[future.model == label]
        dates = pd.to_datetime(part.date)
        ax.plot(steam.date, steam.total_reviews, color="0.4", lw=0.8)
        ax.plot(dates, part.forecast_reviews_median, label=label)
        ax.fill_between(dates, part.lower_95, part.upper_95, alpha=0.2, label="95% умовний інтервал")
        ax.set_yscale("symlog", linthresh=100)
        ax.set(title=label, ylabel="Відгуки, шкала symlog")
        ax.legend(fontsize=8)
    finish(fig, "07_predictive_intervals.png")
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.8), sharex=True)
    for ax, (scenario, part) in zip(axes, ensemble.groupby("scenario", sort=False)):
        for label, g in part.groupby("model"):
            ax.plot(g.ratio, g.rmse_mean, marker=".", label=label)
        ax.set(title=scenario, xlabel="Горизонт / 69", ylabel="Середній RMSE log")
    axes[-1].legend(fontsize=8, ncol=4)
    finish(fig, "08_synthetic.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].bar(daily.index, daily.revenue, width=0.7, color="#39758a")
    axes[0].set(title="Лише наявні дати", ylabel="Виручка виданих операцій")
    sf = pd.read_csv(output / "sales_conditional_extrapolation.csv")
    axes[1].plot(np.arange(-8, 1), np.log1p(daily.revenue), color="black", marker=".")
    for label, part in sf.groupby("model"):
        axes[1].plot(part.future_observed_date_index, part.forecast_log, label=label)
    axes[1].set(title="Умовна екстраполяція №8", xlabel="Індекс наступної наявної дати", ylabel="ln(1 + виручка)")
    axes[1].legend(fontsize=8)
    fig.autofmt_xdate()
    finish(fig, "09_sales.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steam.date.iloc[57:], np.log1p(steam.total_reviews.iloc[57:]), color="black", marker="o", label="Факт")
    part = predictions.loc[predictions.origin == 57]
    for label, g in part.groupby("model"):
        if "default" not in label:
            ax.plot(steam.date.iloc[57:], g.predicted_log, label=label)
    ax.set(ylabel="ln(1 + кількість)", title="Фіксований прогноз останніх 12 місяців без оновлення фактом")
    ax.legend(ncol=4, fontsize=8)
    finish(fig, "10_fixed_forecast.png")

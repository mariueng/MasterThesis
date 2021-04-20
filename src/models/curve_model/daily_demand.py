from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import random
random.seed(1)
import matplotlib.pyplot as plt
from data.data_handler import get_data
import numpy as np
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from src.system.generate_periods import get_all_2019_periods
from src.system.generate_periods import get_random_periods
import seaborn as sns
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)


def find_optimal_order(history):
    warnings.filterwarnings("ignore")
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))
    aic_results = []
    parameter = []
    for param in pdq:
        try:
            model = ARIMA(history, order=param)
            results = model.fit()
            aic_results.append(results.aic)
            parameter.append(param)
        except:
            continue
    d = dict(ARIMA=parameter, AIC=aic_results)
    results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    min_order = results_table.iloc[results_table['AIC'].argmin()]["ARIMA"]
    print("Best order this round {}".format(min_order))
    return results_table


def find_best_arima_parameters():
    demand_column = "Total Vol"
    all_dates = pd.date_range(dt(2014, 1, 1), dt(2019, 6, 1), freq="d")
    chosen_dates = random.sample([d for d in all_dates], 12)
    result_df = None
    for i in range(len(chosen_dates)):
        start = chosen_dates[i]
        print(start.date())
        end = start + timedelta(days=180)
        hist = get_data(start, end, [demand_column, "Weekday"], os.getcwd(), "d")
        decomp = seasonal_decompose(hist[demand_column], model='multiplicative', period=7)
        plot = False
        if i == 0 and plot:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            decomp.plot().suptitle('Multiplicative Seasonal Decomposition', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig("eda/seasonal_daily_demand.png")
            plt.close()
        hist["Trend"] = decomp.trend
        hist["Factor"] = decomp.seasonal
        hist["Trend"] = hist["Trend"].interpolate()
        hist['Trend'] = hist.apply(
            lambda row: row[demand_column] / row['Factor'] if np.isnan(row['Trend']) else row['Trend'], axis=1)
        df = find_optimal_order(hist["Trend"].tolist())
        df = df.rename(columns={"AIC": "AIC {}".format(start)})
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on=["ARIMA"])
    arima_df = result_df[[c for c in result_df.columns if "AIC" in c]]
    result_df["Median AIC"] = arima_df.median(axis=1)
    min_order = result_df.iloc[result_df['Median AIC'].argmin()]["ARIMA"]
    print("\nMinimum order: {}".format(min_order))
    result_df.to_csv("eda/arima_parameters.csv", index=False)


def predict_daily_demand(history, start, end):
    demand_column = history.columns[1]
    exog_column = history.columns[2]
    decomp = seasonal_decompose(history[demand_column], model='multiplicative', period=7)
    history["Factor"] = decomp.seasonal
    history["Trend"] = decomp.trend
    history["Trend"] = history["Trend"].interpolate()
    history['Trend'] = history.apply(
        lambda row: row[demand_column] / row['Factor'] if np.isnan(row['Trend']) else row['Trend'], axis=1)
    print(history)
    assert False
    exog_hist = history[exog_column].to_list()
    demand_hist = history[history.columns[1]].to_list()
    order = (4, 3, 1)
    # model = SARIMAX(demand_hist, order=order, seasonal_order=None, exog=exog_hist)
    model = ARIMA(demand_hist, order=order, seasonal_order=None, exog=exog_hist)
    model_fit = model.fit(disp=0)
    exog_t = get_data(start, end, [exog_column], os.getcwd(), "d")[exog_column].tolist()
    prediction = model_fit.get_forecast(steps=len(exog_t), exog=exog_t, maxiter=3000)
    forecast = prediction.predicted_mean.tolist()
    df = pd.DataFrame(columns=["Date", "Demand Forecast"])
    df["Date"] = [i for i in pd.date_range(start, end, freq='d')]
    df["Demand Forecast"] = forecast
    return df


def evaluate_daily_demand():
    # MAE = 53552.56, MAPE = 5.47. None seasonaly, xog = weekday. order = 4,1,4
    # MAE = 560765.65, MAPE = 60.88. None seasonaly, xog = weekend. order = 4,1,4
    warnings.filterwarnings("ignore")
    demand_column = "Total Vol"
    dummy = "Weekday"
    df = pd.DataFrame(columns=["Period", "Date", dummy, "Demand Forecast", demand_column])
    periods = get_all_2019_periods()
    window = 180
    for i in range(len(periods)):
        start_date = periods[i][0]
        print("Evaluating period starting from {}".format(start_date))
        end_date = periods[i][1]
        prev_day = start_date - timedelta(days=1)
        hist = get_data(prev_day - timedelta(days=window-1), prev_day, [demand_column, dummy], os.getcwd(), "d")
        forecast = predict_daily_demand(hist, start_date, end_date)
        true = get_data(start_date, end_date, [demand_column, dummy], os.getcwd(), "d")
        p_df = pd.DataFrame(columns=df.columns)
        p_df["Date"] = forecast["Date"]
        p_df["Demand Forecast"] = forecast["Demand Forecast"]
        p_df[demand_column] = true[demand_column].to_list()
        p_df[dummy] = true[dummy].to_list()
        p_df["Period"] = i + 1
        df = df.append(p_df, ignore_index=True)
        plot_prediction(start_date, end_date, demand_column, forecast)
    df["AE"] = abs(df[demand_column] - df["Demand Forecast"])
    df["APE"] = 100 * df["AE"] / df[demand_column]
    mae = df["AE"].mean()
    mape = df["APE"].mean()
    print("MAE = {:.2f}, MAPE = {:.2f}".format(mae, mape))


def plot_prediction(start_date, end_date, demand_column, forecast_df):
    plot_df = get_data(start_date - timedelta(days=7), end_date, [demand_column], os.getcwd(), "d")
    plot_df = plot_df.merge(forecast_df, on="Date", how="outer")
    plt.subplots(figsize=full_fig)
    plt.plot(plot_df["Date"], plot_df[demand_column], color=first_color, label="Total Volume")
    plt.plot(plot_df["Date"], plot_df["Demand Forecast"], color=sec_color, label="Demand Forecast")
    for line in plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Daily Demand Forecast from {} to {}".format(start_date, end_date), pad=title_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.ylabel("Volume [MWh]", labelpad=label_pad)
    plt.tight_layout()
    plt.savefig("daily_demand_plots/{}_to_{}.png".format(start_date, end_date))
    plt.close()


def explore_seasonality():
    demand_column = "Total Vol"
    dummy = "Weekend"
    df = get_data("01.01.2014", "31.12.2019", [demand_column, dummy], os.getcwd(), "d")
    sns.boxplot(x=dummy, y=demand_column, data=df)
    plt.show()


if __name__ == '__main__':
    # find_best_arima_parameters()
    # explore_seasonality()
    evaluate_daily_demand()

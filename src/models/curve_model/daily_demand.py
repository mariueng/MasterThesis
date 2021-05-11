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
    demand_column = "Curve Demand"
    all_dates = pd.date_range(dt(2014, 1, 1), dt(2019, 6, 1), freq="d")
    chosen_dates = random.sample([d for d in all_dates], 12)
    result_df = None
    for i in range(len(chosen_dates)):
        start = chosen_dates[i]
        print(start.date())
        end = start + timedelta(days=180)
        hist = get_data(start, end, [demand_column, "Weekday"], os.getcwd(), "d")
        d_df, decomp = get_decomp_new(hist)
        df = find_optimal_order(d_df["Trend"].tolist())
        df = df.rename(columns={"AIC": "AIC {}".format(start)})
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on=["ARIMA"])
    arima_df = result_df[[c for c in result_df.columns if "AIC" in c]]
    result_df["Median AIC"] = arima_df.median(axis=1)
    min_order = result_df.iloc[result_df['Median AIC'].argmin()]["ARIMA"]
    print("\nMinimum order: {}".format(min_order))
    result_df.to_csv("eda/arima_parameters_extrapolate.csv", index=False)


def predict_daily_demand(start, end, plot):
    window = 180
    demand_column = "Curve Demand"
    prev_day = start - timedelta(days=1)
    history = get_data(start - timedelta(days=window), prev_day, [demand_column, "Weekday"], os.getcwd(), "d")
    exog_column = history.columns[2]
    df, decomp = get_decomp_old(history)
    demand_hist = df["Trend"].to_list()
    order = (0, 3, 1)
    order = (4, 3, 1)
    #hist_temp = get_data(history.loc[0, "Date"], history.loc[len(history)-1, "Date"], ["Temp Norway"], os.getcwd(),
                         #"d")
    #hist_temp["Trend"] = seasonal_decompose(hist_temp["Temp Norway"], model='a', period=7, extrapolate_trend='freq').trend
    #hist_temp_diff = hist_temp["Trend"] - hist_temp["Trend"].shift(1)
    #hist_temp_diff[0] = hist_temp_diff[1]
    # model = ARIMA(demand_hist, order=order, exog=hist_temp_diff)
    model = ARIMA(demand_hist, order=order)
    model_fit = model.fit()
    seasonal_df = history[["Weekday", "Factor"]].head(7)
    seasonal_dict = dict(zip(seasonal_df.Weekday, seasonal_df.Factor))
    #hor_temp = get_data(start, end, ["Temp Norway"], os.getcwd(), "d")["Temp Norway"].values
    # hor_temp = get_data(start-timedelta(1), end, ["Temp Norway"], os.getcwd(), "d")
    # hor_temp["Trend"] = seasonal_decompose(hor_temp["Temp Norway"], model='a', period=7, extrapolate_trend='freq').trend
    # hor_temp_diff = hor_temp["Trend"] - hor_temp["Trend"].shift(1)
    # hor_temp_diff = hor_temp_diff[1:]
    #plt.subplots(figsize=full_fig)
    #plt.plot(range(len(hist_temp)+len(hor_temp)), np.append(hist_temp, hor_temp))
    #plt.plot(range(len(hist_temp)), hist_temp)
    #plt.show()
    # prediction = model_fit.get_forecast(steps=14, exog=hor_temp_diff)
    prediction = model_fit.get_forecast(steps=14)
    forecast = prediction.predicted_mean.tolist()
    exog_t = get_data(start, end, [exog_column], os.getcwd(), "d")[exog_column].tolist()
    adjusted_forecast = [forecast[i] * seasonal_dict[exog_t[i]] for i in range(len(forecast))]
    df = pd.DataFrame(columns=["Date", "Demand Forecast"])
    df["Date"] = [i for i in pd.date_range(start, end, freq='d')]
    df["Demand Forecast"] = adjusted_forecast
    if plot:
        plot_decomp(decomp, start, end)
        plot_prediction(start.date(), end.date(), demand_column, df)
    return df


def predict_daily_demand_diff(history, start, end, plot):
    demand_column = "Curve Demand"
    dummy = "Weekday"
    df, decomp = get_decomp_new(history)
    trend = df["Trend"].values
    last_trend = trend[-1]
    df['trend diff'] = df['Trend'] - df['Trend'].shift(1)
    df = df.dropna()
    diff_threshold = abs(df["trend diff"]).mean()
    diff = df["trend diff"].to_list()
    # order = (1, 3, 1)
    order = (4, 1, 4)
    model = ARIMA(diff, order=order, enforce_invertibility=False, enforce_stationarity=False)
    model_fit = model.fit()
    prediction = model_fit.get_forecast(steps=14)
    forecast = prediction.predicted_mean.tolist()
    forecast = [min(f, diff_threshold) if f > 0 else max(-diff_threshold, f) for f in forecast]
    result = pd.DataFrame(columns=["Date", "Demand Diff"])
    result["Date"] = [i for i in pd.date_range(start, start + timedelta(days=13), freq='d')]
    result["Demand Diff"] = forecast
    result["Trend"] = np.PZERO
    for i in range(len(result)):
        result.loc[i, "Trend"] = result.loc[i, "Demand Diff"] + last_trend
        last_trend = result.loc[i, "Trend"]
    seasonal_df = df[[dummy, "Factor"]].head(7)
    seasonal_dict = dict(zip(seasonal_df[dummy], seasonal_df["Factor"]))
    exog_t = get_data(start, start + timedelta(days=13), [dummy], os.getcwd(), "d")[dummy].tolist()
    forecast = result["Trend"].tolist()
    adjusted_forecast = [forecast[i] * seasonal_dict[exog_t[i]] for i in range(len(forecast))]
    result["Demand Forecast"] = adjusted_forecast
    if plot:
        plot_decomp(df, start, end)
        plot_prediction(start, end, demand_column, df)
    return result


def plot_decomp(decomp, start, end):
    start = start.date()
    end = end.date()
    w_start = start - timedelta(days=180)
    plt.rcParams.update({'figure.figsize': (12, 9)})
    decomp.plot().suptitle('Seasonal Decomposition {} to {}'.format(w_start, start), fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("daily_demand_plots/{}_to_{}_window.png".format(start, end))
    plt.close()


def evaluate_daily_demand(differences):
    # MAE = 53552.56, MAPE = 5.47. None seasonaly, xog = weekday. order = 4,1,4
    # MAE = 47198.10, MAPE = 4.74. Arima order 4,3,1 after deseasonlising on weekday
    warnings.filterwarnings("ignore")
    demand_column = "Curve Demand"
    dummy = "Weekday"
    df = pd.DataFrame(columns=["Period", "Date", dummy, "Demand Forecast", demand_column])
    periods = get_all_2019_periods()
    window = 300
    for i in range(len(periods)):
        start_date = periods[i][0]
        end_date = periods[i][1]
        prev_day = start_date - timedelta(days=1)
        hist = get_data(prev_day - timedelta(days=window-1), prev_day, [demand_column, dummy], os.getcwd(), "d")
        if differences:
            forecast = predict_daily_demand_diff(hist, start_date, end_date, False)
        else:
            forecast = predict_daily_demand(hist, start_date, end_date, False)
        #plot_prediction(start_date, end_date, demand_column, forecast)
        true = get_data(start_date, end_date, [demand_column, dummy], os.getcwd(), "d")
        p_df = pd.DataFrame(columns=df.columns)
        p_df["Date"] = forecast["Date"]
        p_df["Demand Forecast"] = forecast["Demand Forecast"]
        p_df[demand_column] = true[demand_column].to_list()
        p_df[dummy] = true[dummy].to_list()
        p_df["Period"] = i + 1
        df = df.append(p_df, ignore_index=True)
        period_mae = abs((p_df[demand_column] - p_df["Demand Forecast"]).mean())
        print("Period starting from {}. MAE\t{:.2f}".format(start_date, period_mae))
    df["AE"] = abs(df[demand_column] - df["Demand Forecast"])
    df["APE"] = 100 * df["AE"] / df[demand_column]
    mae = df["AE"].mean()
    mape = df["APE"].mean()
    print("MAE = {:.2f}, MAPE = {:.2f}".format(mae, mape))


def plot_prediction(start_date, end_date, demand_column, forecast_df):
    plot_df = get_data(start_date - timedelta(days=14), end_date, [demand_column], os.getcwd(), "d")
    plot_df = plot_df.merge(forecast_df, on="Date", how="outer")
    plt.subplots(figsize=full_fig)
    plt.plot(plot_df["Date"], plot_df[demand_column], color=first_color, label="Demand")
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
    demand_column = "Curve Demand"
    dummy = "Weekend"
    df = get_data("01.01.2014", "31.12.2019", [demand_column, dummy], os.getcwd(), "d")
    sns.boxplot(x=dummy, y=demand_column, data=df)
    plt.show()


def demand_is_stationary(x):
    # H0: Suggests the time series has a unit root, meaning it is non-stationary.
    # H1: Null-hyp is rejected. Suggests the time series does not have a unit root, meaning it is stationary.
    # p-value <= 0.05: Reject the null hypothesis
    from statsmodels.tsa.stattools import adfuller
    threshold = 0.05
    result = adfuller(x)
    print_values = False
    if print_values:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
    return result[1] < threshold, result[1]


def check_how_many_periods_stationary_demand():
    result = pd.DataFrame(columns=["Period", "Start date", "P value", "Stationary"])
    demand_column = "Curve Demand"
    dummy = "Weekday"
    periods = get_all_2019_periods()
    window = 180
    for i in range(len(periods)):
        start_date = periods[i][0]
        prev_day = start_date - timedelta(days=1)
        hist = get_data(prev_day - timedelta(days=window-1), prev_day, [demand_column, dummy], os.getcwd(), "d")
        df, decomp = get_decomp(hist)
        demand_hist = df["Trend"].to_list()
        is_stat, p_value = demand_is_stationary(demand_hist)
        row = {"Period": i, "Start date": start_date, "P value": p_value, "Stationary": is_stat}
        result = result.append(row, ignore_index=True)
    is_stat_df = result[result["Stationary"]==True]
    print("Validation periods stationary demand: {} of {}".format(len(is_stat_df), len(result)))
    print("Mean p value for rejecting H0: {:.2f}".format((result["P value"].mean())))


def get_decomp(hist):
    demand_column = hist.columns[1]
    decomp = seasonal_decompose(hist[demand_column], model='multiplicative', period=7, extrapolate_trend='freq')
    hist["Factor"] = decomp.seasonal
    hist["Trend"] = decomp.trend
    return hist, decomp


def get_decomp_new(hist):
    hist = hist.reset_index(drop=True)
    demand_column = "Curve Demand"
    decomp = seasonal_decompose(hist[demand_column], model='multiplicative', period=7)
    hist["Factor"] = decomp.seasonal
    hist["Trend"] = decomp.trend
    hist["Adj Trend"] = hist["Curve Demand"] / hist["Factor"]
    hist.loc[0, "Trend"] = sum(hist["Adj Trend"].head(3) * np.asarray([0.2, 0.3, 0.5]))
    hist.loc[len(hist)-1, "Trend"] = sum(hist["Adj Trend"].tail(3) * np.asarray([0.2, 0.3, 0.5]))
    hist["Trend"] = hist["Trend"].interpolate(method='quadratic', limit_area="inside")
    hist = hist.drop(columns=["Adj Trend"])
    return hist, decomp


def get_decomp_old(df):
    demand_column = df.columns[1]
    decomp = seasonal_decompose(df[demand_column], model='multiplicative', period=7)
    df["Factor"] = decomp.seasonal
    df["Trend"] = decomp.trend
    df["Trend"] = df["Trend"].interpolate()
    df['Trend'] = df.apply(
        lambda row: row[demand_column] / row['Factor'] if np.isnan(row['Trend']) else row['Trend'], axis=1)
    return df, decomp


def explore_how_to_make_period_stationary():
    warnings.filterwarnings("ignore")
    demand_column = "Curve Demand"
    dummy = "Weekday"
    period = (dt(2019, 10, 27), dt(2019, 11, 9))
    start = period[0]
    end = period[1]
    window = 180
    prev_day = start - timedelta(days=1)
    hist = get_data(prev_day - timedelta(days=window - 1), prev_day, [demand_column, dummy], os.getcwd(), "d")
    df, decomp = get_decomp_new(hist)
    trend = df["Trend"].values
    is_stat, p_val = demand_is_stationary(trend)
    print("Is stat {}, p value {}".format(is_stat, p_val))
    df['trend diff'] = df['Trend'] - df['Trend'].shift(1)
    df = df.dropna()
    fig, axs = plt.subplots(2, figsize=(13, 8))
    fig.suptitle("Demand Stationarity {}".format(start.date()))
    axs[0].plot(df["Date"], df["Curve Demand"], label="Demand", color=sec_color)
    axs[0].plot(df["Date"], df["Trend"], label="Trend", color=first_color)
    axs[1].plot(df["Date"], df["trend diff"], label="Diff trend", color=first_color)
    for ax in axs:
        for line in ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.06), fancybox=True,
                              shadow=True).get_lines():
            line.set_linewidth(2)
    # plt.show()
    # plt.close()
    diff = df["trend diff"].values
    is_stat, p_val = demand_is_stationary(diff)
    print("Is stat {}, p value {}".format(is_stat, p_val))
    #result = find_optimal_order(diff)
    #result.to_csv(r"arima_results_{}.csv".format(start.date()))
    result = predict_daily_demand_diff(df[["Date", dummy, demand_column]], start, end, plot=False)
    true = get_data(start, start+timedelta(days=13), ["Curve Demand"], os.getcwd(), "d")
    result["True"] = true["Curve Demand"]
    fig, axs = plt.subplots(2, figsize=(13, 8))
    fig.suptitle("Demand Stationarity {}".format(start.date()))
    axs[0].plot(df["Date"], df["Curve Demand"], label="Demand", color=sec_color)
    axs[0].plot(result["Date"], true["Curve Demand"], label="_nolegend_", color=sec_color)
    axs[0].plot(df["Date"], df["Trend"], label="Trend", color=first_color)
    axs[0].plot(result["Date"], result["Trend"], label="Trend forecast", color="orange")
    axs[1].plot(df["Date"], df["trend diff"], label="Diff trend", color=first_color)
    axs[1].plot(result["Date"], result["Demand Diff"], label="Diff trend forecast", color="orange")
    for ax in axs:
        for line in ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.06), fancybox=True,
                              shadow=True).get_lines():
            line.set_linewidth(2)

    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(result["Date"], result["True"], label="True demand")
    plt.plot(result["Date"], result["Trend"], label="Trend forecast")
    plt.plot(result["Date"], result["Demand Forecast"], label="Complete forecast")
    plt.title("Demand Forecast {} to {}".format(start, end), pad=title_pad)
    plt.ylabel("Date", labelpad=label_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # find_best_arima_parameters()
    # explore_seasonality()
    # evaluate_daily_demand(differences=True)
    # check_how_many_periods_stationary_demand()
    explore_how_to_make_period_stationary()

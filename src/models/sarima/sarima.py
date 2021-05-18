import os
from data import data_handler
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from src.system.generate_periods import get_random_periods
from src.system.scores import calculate_coverage_error
from src.system.scores import calculate_smape
from data.data_handler import get_data
import numpy as np
from src.preprocessing.arcsinh import arcsinh
import random
random.seed(1)


class Sarima:
    def __init__(self):
        self.name = "Sarima"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = self.get_forecast(forecast_df)
        return forecast_df

    @staticmethod
    def get_forecast(forecast_df):
        warnings.filterwarnings("ignore")
        start_date = forecast_df.at[0, "Date"]
        end_date = forecast_df.at[len(forecast_df) - 1, "Date"]
        data = get_data(start_date, end_date, ["Weekday", "Holiday"], os.getcwd(), "h")
        days_back = 14
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back - 1)
        hist = get_data(train_start_date, train_end_date, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
        hist["System Price"] = hist.apply(lambda row: row["System Price"] * get_hol_factor() if
        row["Holiday"] == 1 and row["Weekday"] != 7 else row["System Price"], axis=1)
        hist["Factor"] = [get_weekday_c()[weekday] for weekday in hist["Weekday"]]
        hist["System Price"] = hist["System Price"] / hist["Factor"]
        hist, a, b = arcsinh.to_arcsinh(hist, "System Price")
        pre_proc = True
        if pre_proc:
            history = [x for x in hist["Trans System Price"]]
        else:
            history = [x for x in hist["System Price"]]
        ses_order = (1, 1, 1, 24)
        order = (1, 0, 2)
        model = ARIMA(history, order=order, seasonal_order=ses_order, trend="n", enforce_stationarity=False)
        model_fit = model.fit()
        prediction = model_fit.get_forecast(steps=len(forecast_df))
        forecast = prediction.predicted_mean.tolist()
        if pre_proc:
            forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
            conf_int = prediction.conf_int(alpha=0.0015)
            predictionslower = arcsinh.from_arcsin_to_original([conf_int[i][0] for i in range(len(conf_int))], a, b)
            predictionsupper = arcsinh.from_arcsin_to_original([conf_int[i][1] for i in range(len(conf_int))], a, b)
        else:
            conf_int = prediction.conf_int(alpha=0.1)
            predictionslower = [conf_int[i][0] for i in range(len(conf_int))]
            predictionsupper = [conf_int[i][1] for i in range(len(conf_int))]
        d = dict(Forecast=forecast, Upper=predictionsupper, Lower=predictionslower)
        results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        results_table["Date"] = data["Date"]
        results_table["Hour"] = data["Hour"]
        for i in range(len(results_table)):
            if data.loc[i, "Holiday"] == 1 and data.loc[i, "Weekday"] != 7:
                for col in ["Forecast", "Upper", "Lower"]:
                    results_table.loc[i, col] /= get_hol_factor()
            for col in ["Forecast", "Upper", "Lower"]:
                results_table.loc[i, col] *= hist.loc[i, "Factor"]
        return results_table


def find_optimal_order(history):
    print("Finding optimal order")
    warnings.filterwarnings("ignore")
    p = d = q = range(0, 4)
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
    assert False
    return min_order


# methods for internal testing ___________________________________________________________
def stat_test(df):
    x = df["System Price"]
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def plot(train, results, path):
    df = train.append(results, ignore_index=True)
    plt.subplots(figsize=(13, 7))
    plt.plot(df.index, df["System Price"], label="True")
    plt.plot(df.index, df["Forecast"], label="Forecast")
    # plt.plot(df.index, df["Upper"], label="Upper")
    # plt.plot(df.index, df["Lower"], label="Lower")
    plt.legend()
    ymax = max(max(df["System Price"]), max(df["Forecast"]))
    ymin = min(min(df["System Price"]), min(df["Forecast"]))
    plt.ylim(ymin, ymax)
    plt.savefig(path + ".png")


def tune_best_alpha():
    alphas = [0.05, 0.1, 0.15, 0.20, 0.25]
    periods = get_random_periods(10)
    orders = get_orders(periods)
    results = {}
    sarimax = Sarima()
    for a in alphas:
        scores = []
        for period in periods:
            order = orders[period]
            time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
            time_df["Forecast"] = np.nan
            time_df["Upper"] = np.nan
            time_df["Lower"] = np.nan
            result = sarimax.forecast(time_df, a, order)
            true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
            result = true_price_df.merge(result, on=["Date", "Hour"], how="outer")
            cov, score = calculate_coverage_error(result)
            scores.append(score)
        s = sum(scores) / len(scores)
        print("Alpha {}, error {}".format(a, s))
        result[a] = s
    print("Min ACE up and down: " + str(min(results, key=results.get)))


def get_orders(periods):
    result = {}
    for period in periods:
        start_date = period[0]
        days_back = 7
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back - 1)
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price", "Total Vol"], os.getcwd(), "h")
        history = [x for x in train["System Price"]]
        order = find_optimal_order(history)
        print("Period starting from {} has order {}".format(period[0], order))
        result[period] = order
    return result


def run():
    # stat_test(train_)
    start_date_ = dt(2019, 1, 3)
    end_date_ = start_date_ + timedelta(days=13)
    forecast_df = get_empty_forecast(start_date_, end_date_)
    model = Sarima()
    result = model.forecast(forecast_df)
    days_back = 14
    train_start_date = forecast_df.loc[0, "Date"].date() - timedelta(days=days_back)
    train_end_date = train_start_date + timedelta(days=days_back - 1)
    train = get_data(train_start_date, train_end_date, ["System Price"], os.getcwd(), "h")
    true_test = get_data(start_date_, end_date_, ["System Price"], os.getcwd(), "h")
    result = result.merge(true_test, on=["Date", "Hour"], how="outer")
    pre_proc = True
    if pre_proc:
        path = dt.strftime(start_date_, "%d_%m_%Y") + "_trans"
        print("Trans SMAPE {}".format(calculate_smape(result)))
    else:
        path = dt.strftime(start_date_, "%d_%m_%Y") + "_orig"
        print("Orig SMAPE {}".format(calculate_smape(result)))
    plot(train, result, path)
    # result.to_csv(path + '.csv')


def get_empty_forecast(start_date_, end_date_):
    time_df = get_data(start_date_, end_date_, [], os.getcwd(), "h")
    if len(time_df) != 336:
        print("Length of horizon: {}".format(len(time_df)))
        print("Prediction horizon must be length 336")
        assert False
    time_df["Forecast"] = np.nan
    time_df["Upper"] = np.nan
    time_df["Lower"] = np.nan
    return time_df


def find_optimal_order_2(history):
    warnings.filterwarnings("ignore")
    p = q = range(6)
    d = range(2)
    pdq = list(itertools.product(p, d, q))
    aic_results = []
    parameter = []
    for param in pdq:
        try:
            model = ARIMA(history, order=param, seasonal_order=(1, 1, 1, 24))
            results = model.fit()
            aic_results.append(results.aic)
            print("{} {:.2f}".format(param, results.aic))
            parameter.append(param)
        except:
            continue
    d = dict(ARIMA=parameter, AIC=aic_results)
    results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    min_order = results_table.iloc[results_table['AIC'].argmin()]["ARIMA"]
    print("Best order this round {}".format(min_order))
    return results_table


def get_hol_factor():
    f = 1.1158
    return f


def get_weekday_c():
    d = {1: 1.028603, 2: 1.034587, 3: 1.0301834, 4: 1.033991, 5: 1.014928, 6: 0.941950, 7: 0.915758}
    return d


def find_best_arima_parameters():
    all_dates = pd.date_range(dt(2014, 7, 1), dt(2019, 5, 19), freq="d")
    chosen_dates = random.sample([d for d in all_dates], 30)
    result_df = None
    c = get_weekday_c()
    for i in range(len(chosen_dates)):
        start = chosen_dates[i]
        print(start.date())
        end = start + timedelta(days=10)
        hist = get_data(start, end, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
        hist["System Price"] = hist.apply(lambda row: row["System Price"] * get_hol_factor() if
        row["Holiday"] == 1 and row["Weekday"] != 7 else row["System Price"], axis=1)
        hist["Factor"] = [c[weekday] for weekday in hist["Weekday"]]
        hist["System Price"] = hist["System Price"] / hist["Factor"]
        hist, a, b = arcsinh.to_arcsinh(hist, "System Price")
        df = find_optimal_order_2(hist["Trans System Price"])
        df = df.rename(columns={"AIC": "AIC {}".format(start)})
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on=["ARIMA"])
    arima_df = result_df[[c for c in result_df.columns if "AIC" in c]]
    result_df["Median AIC"] = arima_df.median(axis=1)
    min_order = result_df.iloc[result_df['Median AIC'].argmin()]["ARIMA"]
    print("\nMinimum order: {}".format(min_order))
    result_df = result_df.round(3)
    result_df.to_csv("optimal_parameters.csv", index=False)


if __name__ == '__main__':
    find_best_arima_parameters()
    assert False
    run()

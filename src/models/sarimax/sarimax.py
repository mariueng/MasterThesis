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


class Sarimax:
    def __init__(self):
        self.name = "Sarimax"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast = self.get_forecast(forecast_df)
        forecast_df["Forecast"] = forecast["forecast"]
        forecast_df["Upper"] = forecast["upper"]
        forecast_df["Lower"] = forecast["lower"]
        return forecast_df

    @staticmethod
    def get_forecast(forecast_df):
        warnings.filterwarnings("ignore")
        start_date = forecast_df.at[0, "Date"]
        end_date = forecast_df.at[len(forecast_df)-1, "Date"]
        days_back = 14
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back-1)
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price"], os.getcwd())
        train, a, b = arcsinh.to_arcsinh(train, "System Price")
        history = [x for x in train["System Price"]]
        pre_proc = True
        if pre_proc:
            history = [x for x in train["Trans System Price"]]
        ex_col = "Weekday"
        exog = data_handler.get_data(train_start_date, train_end_date, [ex_col], os.getcwd())[ex_col].tolist()
        order = find_optimal_order(history)
        ses_order = (1, 1, 1, 24)
        model = SARIMAX(history, order=order, seasonal_order=ses_order, exog=exog)
        exog_t = data_handler.get_data(start_date, end_date, [ex_col], os.getcwd())[ex_col].tolist()
        model_fit = model.fit(disp=0)
        prediction = model_fit.get_forecast(steps=len(forecast_df), exog=exog_t)
        forecast = prediction.predicted_mean.tolist()
        if pre_proc:
            forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
            conf_int = prediction.conf_int(alpha=0.02)
            predictionslower = arcsinh.from_arcsin_to_original([conf_int[i][0] for i in range(len(conf_int))], a, b)
            predictionsupper = arcsinh.from_arcsin_to_original([conf_int[i][1] for i in range(len(conf_int))], a, b)
        else:
            conf_int = prediction.conf_int(alpha=0.1)
            predictionslower = [conf_int[i][0] for i in range(len(conf_int))]
            predictionsupper = [conf_int[i][1] for i in range(len(conf_int))]
        d = dict(forecast=forecast, upper=predictionsupper, lower=predictionslower)
        results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        return results_table


def find_optimal_order(history):
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
    #plt.plot(df.index, df["Upper"], label="Upper")
    #plt.plot(df.index, df["Lower"], label="Lower")
    plt.legend()
    ymax = max(max(df["System Price"]), max(df["Forecast"]))
    ymin = min(min(df["System Price"]), min(df["Forecast"]))
    plt.ylim(ymin, ymax)
    plt.savefig(path+".png")


def tune_best_alpha():
    alphas = [0.05, 0.1, 0.15, 0.20, 0.25]
    periods = get_random_periods(10)
    orders = get_orders(periods)
    results = {}
    sarimax = Sarimax()
    for a in alphas:
        scores = []
        for period in periods:
            order = orders[period]
            time_df = get_data(period[0], period[1], [], os.getcwd())
            time_df["Forecast"] = np.nan
            time_df["Upper"] = np.nan
            time_df["Lower"] = np.nan
            result = sarimax.forecast(time_df, a, order)
            true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd())
            result = true_price_df.merge(result, on=["Date", "Hour"], how="outer")
            cov, score = calculate_coverage_error(result)
            scores.append(score)
        s = sum(scores)/len(scores)
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
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price", "Total Vol"], os.getcwd())
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
    model = Sarimax()
    result = model.forecast(forecast_df)
    days_back = 7
    train_start_date = forecast_df.loc[0, "Date"].date() - timedelta(days=days_back)
    train_end_date = train_start_date + timedelta(days=days_back - 1)
    train = get_data(train_start_date, train_end_date, ["System Price"], os.getcwd())
    true_test = get_data(start_date_, end_date_, ["System Price"], os.getcwd())
    result = result.merge(true_test, on=["Date", "Hour"], how="outer")
    pre_proc = False
    if pre_proc:
        path = dt.strftime(start_date_, "%d_%m_%Y") + "_trans"
        print("Trans SMAPE {}".format(calculate_smape(result)))
    else:
        path = dt.strftime(start_date_, "%d_%m_%Y") + "_orig"
        print("Orig SMAPE {}".format(calculate_smape(result)))
    plot(train, result, path)
    #result.to_csv(path + '.csv')


def get_empty_forecast(start_date_, end_date_):
    time_df = get_data(start_date_, end_date_, [], os.getcwd())
    if len(time_df) != 336:
        print("Length of horizon: {}".format(len(time_df)))
        print("Prediction horizon must be length 336")
        assert False
    time_df["Forecast"] = np.nan
    time_df["Upper"] = np.nan
    time_df["Lower"] = np.nan
    return time_df


if __name__ == '__main__':
    run()

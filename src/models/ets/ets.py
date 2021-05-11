# grid search exponential smoothing
from data import data_handler
import os
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
import warnings
from src.system.generate_periods import get_random_periods
from src.system.scores import calculate_coverage_error
from data.data_handler import get_data
import numpy as np
from src.preprocessing.arcsinh import arcsinh
from src.system.scores import calculate_smape


class Ets:
    def __init__(self):
        self.name = "ETS"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        filterwarnings("ignore")
        start_date = forecast_df.at[0, "Date"]
        data = get_data(start_date, start_date+timedelta(days=13), ["Weekday", "Holiday"], os.getcwd(), "h")
        pre_proc = True
        train, a, b, hist = get_training_data(start_date, pre_proc, days_back=14)
        ets = ExponentialSmoothing(train, seasonal=24)
        model_fit = ets.fit(disp=0, optimized=True, use_boxcox=True, remove_bias=True)
        forecast = model_fit.get_prediction(start=len(train), end=len(train) + len(forecast_df) - 1)
        prediction = forecast.predicted_mean
        if pre_proc:
            prediction = arcsinh.from_arcsin_to_original(prediction, a, b)
            conf_int = forecast.conf_int(alpha=0.12)
            lowers = arcsinh.from_arcsin_to_original([conf_int[i][0] for i in range(len(conf_int))], a, b)
            uppers = arcsinh.from_arcsin_to_original([conf_int[i][1] for i in range(len(conf_int))], a, b)
        else:
            conf_int = forecast.conf_int(alpha=0.1)
            lowers = [conf_int[i][0] for i in range(len(conf_int))]
            uppers = [conf_int[i][1] for i in range(len(conf_int))]
        forecast_df["Forecast"] = prediction
        forecast_df["Upper"] = uppers
        forecast_df["Lower"] = lowers
        for i in range(len(forecast_df)):
            if data.loc[i, "Holiday"] == 1 and data.loc[i, "Weekday"] != 7:
                for col in ["Forecast", "Upper", "Lower"]:
                    forecast_df.loc[i, col] /= get_hol_factor()
            for col in ["Forecast", "Upper", "Lower"]:
                forecast_df.loc[i, col] *= hist.loc[i, "Factor"]
        return forecast_df


def get_training_data(start_date, pre_proc, days_back):
    train_start_date = start_date - timedelta(days=days_back)
    train_end_date = train_start_date + timedelta(days=days_back - 1)
    hist = get_data(train_start_date, train_end_date, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
    hist["System Price"] = hist.apply(lambda row: row["System Price"] * get_hol_factor() if
    row["Holiday"] == 1 and row["Weekday"] != 7 else row["System Price"], axis=1)
    hist["Factor"] = [get_weekday_c()[weekday] for weekday in hist["Weekday"]]
    hist["System Price"] = hist["System Price"] / hist["Factor"]
    if pre_proc:
        hist, a, b = arcsinh.to_arcsinh(hist, "System Price")
        data = hist["Trans System Price"].tolist()
        return data, a, b, hist
    else:
        return hist["System Price"].tolist(), None, None, hist


# Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
    warnings.filterwarnings("ignore")
    t, d, s, p, b, r = config
    # define model
    history = array(history)
    model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s)
    # fit model
    model_fit = model.fit(disp=0, optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    warnings.filterwarnings("ignore")
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = exp_smoothing_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    return cfg, result


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    warnings.filterwarnings("ignore")
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of exponential smoothing configs to try
def exp_smoothing_configs():
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = [None]
    p_params = [24]
    b_params = [True]
    r_params = [True]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
    return models


# Local methods used for internal testing ---------------------------------------------------------------------
def get_train_data():
    start_date_ = "15.01.2019"
    end_date_ = "29.01.2019"
    train = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd(), "h")["System Price"].tolist()
    return train


def get_test_data():
    start_date_ = "30.01.2019"
    end_date_ = "12.02.2019"
    test = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd(), "h")["System Price"].tolist()
    return test


def get_best_params(train):
    n_test = 4  # data split
    cfg_list = exp_smoothing_configs()  # model configs
    scores = grid_search(train, cfg_list, n_test) # grid search
    best = None
    for cfg, error in scores[:1]:
        best = cfg
    configs = {"Trend": best[0], "Damped": best[1], "Seasonal": best[2], "Periods": best[3],
               "Box": best[4], "Remove": best[5]}
    return configs


def get_forecast(train, conf, test):
    ets = ExponentialSmoothing(train, trend=conf["Trend"], seasonal=24, damped_trend=conf["Damped"])
    model_fit = ets.fit(disp=0, optimized=True, use_boxcox=conf["Box"], remove_bias=conf["Remove"])
    forecast = model_fit.forecast(steps=len(test))
    return forecast


def plot(df):
    df["Hour"] = pd.to_datetime(df['Hour'], format="%H").dt.time
    df["DateTime"] = df.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    plt.subplots(figsize=(13, 7))
    plt.plot(df["DateTime"], df["System Price"], label="True")
    plt.plot(df["DateTime"], df["Forecast"], label="Forecast")
    plt.legend()
    plt.show()
    plt.close()


def tune_best_alpha():
    alphas = [0.05, 0.1, 0.15, 0.20, 0.25]
    periods = get_random_periods(10)
    results = {}
    ets = Ets()
    for a in alphas:
        scores = []
        for period in periods:
            time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
            time_df["Forecast"] = np.nan
            time_df["Upper"] = np.nan
            time_df["Lower"] = np.nan
            result = ets.forecast(time_df, a)
            true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
            result = true_price_df.merge(result, on=["Date", "Hour"], how="outer")
            cov, score = calculate_coverage_error(result)
            scores.append(score)
        s = sum(scores)/len(scores)
        print("Alpha {}, error {}".format(a, s))
        result[a] = s
    print("Min ACE up and down: " + str(min(results, key=results.get)))


def run():
    # stat_test(train_)
    start_date = dt(2019, 1, 3)
    end_date = start_date + timedelta(days=13)
    forecast_df = get_empty_forecast(start_date, end_date)
    model = Ets()
    result = model.forecast(forecast_df)
    true_test = get_data(start_date, end_date, ["System Price"], os.getcwd(), "h")
    result = result.merge(true_test, on=["Date", "Hour"], how="outer")
    pre_proc = False
    if pre_proc:
        path = dt.strftime(start_date, "%d_%m_%Y") + "_trans"
        print("Trans SMAPE {}".format(calculate_smape(result)))
    else:
        path = dt.strftime(start_date, "%d_%m_%Y") + "_orig"
        print("Orig SMAPE {}".format(calculate_smape(result)))
    plot(result)
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


def get_hol_factor():
    f = 1.1158
    return f


def get_weekday_c():
    d = {1: 1.028603, 2: 1.034587, 3: 1.0301834, 4: 1.033991, 5: 1.014928, 6: 0.941950, 7: 0.915758}
    return d


if __name__ == '__main__':
    run()
    # tune_best_alpha()

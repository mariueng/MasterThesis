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
        forecast = self.get_forecast(forecast_df)
        forecast_df["Forecast"] = forecast["forecast"]
        forecast_df["Upper"] = forecast["upper"]
        forecast_df["Lower"] = forecast["lower"]
        return forecast_df

    @staticmethod
    def get_forecast(forecast_df):
        warnings.filterwarnings("ignore")
        start_date = forecast_df.at[0, "Date"]
        days_back = 14
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back-1)
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price", "Total Vol"], os.getcwd())
        history = [x for x in train["System Price"]]
        exog = [x for x in train["Total Vol"]]
        order = find_optimal_order(train)
        ses_order = (1, 1, 1, 24)
        model = SARIMAX(history, order=order, seasonal_order=ses_order, exog=exog)
        exog_t = data_handler.get_data(start_date, start_date+timedelta(days=13), ["Total Vol"], os.getcwd())["Total Vol"].tolist()
        model_fit = model.fit(disp=0)
        prediction = model_fit.get_forecast(steps=len(forecast_df), exog=exog_t)
        forecast = prediction.predicted_mean
        conf_int = prediction.conf_int(alpha=0.15)
        predictions = forecast.tolist()
        predictionslower = [conf_int[i][0] for i in range(len(conf_int))]
        predictionsupper = [conf_int[i][1] for i in range(len(conf_int))]
        d = dict(forecast=predictions, upper=predictionsupper, lower=predictionslower)
        results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        return results_table


def find_optimal_order(train):
    warnings.filterwarnings("ignore")
    history = [x for x in train["System Price"]]
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
def SARIMA_pre(train, test):
    warnings.filterwarnings("ignore")
    history = [x for x in train["System Price"]]
    x = test["System Price"].tolist()
    order = find_optimal_order(train)
    ses_order = (1, 1, 0, 24)
    model = SARIMAX(history, order=order, seasonal_order=ses_order)
    model_fit = model.fit(disp=0)
    prediction = model_fit.get_forecast(steps=len(x))
    forecast = prediction.predicted_mean
    conf_int = prediction.conf_int(alpha=0.1)
    predictions = forecast.tolist()
    predictionslower = [conf_int[i][0] for i in range(len(conf_int))]
    predictionsupper = [conf_int[i][1] for i in range(len(conf_int))]
    d = dict(data=x, forecast=predictions, lower=predictionslower, upper=predictionsupper)
    results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    plot(train, results_table)
    results_table.to_csv(r'sarima_results.csv')

def stat_test(df):
    x = df["System Price"]
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def plot(train, resultstable):
    df = pd.DataFrame(columns = ["Hour", "True", "Forecast", "Upper", "Lower"])
    for i in range(len(train)):
        row = {"Hour": i, "True": train_.at[i, "System Price"], "Forecast": None, "Upper": None, "Lower": None}
        df = df.append(row, ignore_index=True)
    for index, row in resultstable.iterrows():
        true = row["data"]
        forecast = row["forecast"]
        upper = row["upper"]
        lower = row["lower"]
        hour = len(train) + index
        row = {"Hour": hour, "True": true, "Forecast": forecast, "Upper": upper, "Lower": lower}
        df = df.append(row, ignore_index=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(df["Hour"], df["True"], label="True")
    plt.plot(df["Hour"], df["Forecast"], label="Forecast")
    plt.plot(df["Hour"], df["Upper"], label="Upper")
    plt.plot(df["Hour"], df["Lower"], label="Lower")
    plt.legend()
    ymax = max(df["True"])*1.2
    ymin = min(df["True"])*0.8
    plt.ylim(ymin, ymax)
    plt.savefig(r'sarima_results.png')


if __name__ == '__main__':
    start_date_ = "24.01.2019"
    end_date_ = "27.01.2019"
    train_ = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd())
    #stat_test(train_)
    # ------------
    start_date_ = "28.01.2019"
    end_date_ = "05.02.2019"
    test_ = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd())
    SARIMA_pre(train_, test_)

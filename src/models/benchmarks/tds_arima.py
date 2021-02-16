import os
from data import data_handler
import pmdarima as pm
from pmdarima.arima import StepwiseContext
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools
from statsmodels.tsa.arima.model import ARIMA


class TdsArima:
    def __init__(self):
        self.name = "TDS Arima"
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
        return forecast_df


def run():
    model_ = TdsArima()
    start_date_ = "30.01.2019"
    end_date_ = "12.02.2019"
    time_list_ = data_handler.get_data(start_date_, end_date_, [], os.getcwd())
    time_list_["Forecast"] = np.nan
    time_list_["Upper"] = np.nan
    time_list_["Lower"] = np.nan
    forecast_ = model_.forecast(time_list_)
    print(forecast_)


def stat_test(df):
    x = df["System Price"]
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def AIC_iteration(train):
    warnings.filterwarnings("ignore")
    history = [x for x in train["System Price"]]
    p = d = q = range(0, 6)
    pdq = list(itertools.product(p, d, q))
    aic_results = []
    parameter = []
    for param in pdq:
        try:
            model = ARIMA(history, order=param)
            results = model.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            aic_results.append(results.aic)
            parameter.append(param)
        except:
            continue
    d = dict(ARIMA=parameter, AIC=aic_results)
    results_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    results_table.to_csv(r'AIC.csv')

def ARIM_pre():
    warnings.filterwarnings("ignore")
    df1 = df['price_sterling']
    df2 = df1.loc[:300]
    X = df2.values
    size = int(len(X) * 0.66)
    train, test = X[0:size] , X[size:len(X)]
    history = [x for x in train]
    predictions = []
    predictionslower = []
    predictionsupper = []
    for k in range(len(test)):
        model = ARIMA(history, order=(3, 1, 5))
        model_fit = model.fit()
        forecast, stderr, conf_int = model_fit.forecast()
        yhat = forecast[0]
        yhatlower = conf_int[0][0]
        yhatupper = conf_int[0][1]
        predictions.append(yhat)
        predictionslower.append(yhatlower)
        predictionsupper.append(yhatupper)
        obs = test[k]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        print('95 prediction interval: %f to %f' % (yhatlower, yhatupper))
    error = mean_squared_error(test, predictions)
    RMSE = np.sqrt(error)
    print('TEST MSE: %.3f' % error)
    print('RMSE: %.3f' % (RMSE))
    d = dict(data=X, forecast=predictions, lower=predictionslower, upper=predictionsupper)
    results_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items()]))
    results_table.to_csv(r'arima_forecasting.csv')


if __name__ == '__main__':
    start_date_ = "23.01.2019"
    end_date_ = "29.01.2019"
    train_ = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd())
    stat_test(train_)
    # ------------
    start_date_ = "30.01.2019"
    end_date_ = "12.02.2019"
    test = data_handler.get_data(start_date_, end_date_, ["System Price"], os.getcwd())
    AIC_iteration(train_)

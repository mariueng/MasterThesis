import os
from data import data_handler
from datetime import datetime as dt
from datetime import timedelta
import warnings
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing


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
        forecast, uppers, lowers = self.get_forecast(forecast_df)
        forecast_df["Forecast"] = forecast
        forecast_df["Upper"] = uppers
        forecast_df["Lower"] = lowers
        return forecast_df

    @staticmethod
    def get_forecast(forecast_df):
        warnings.filterwarnings("ignore")
        start_date = forecast_df.at[0, "Date"]
        days_back = 14
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back - 1)
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price"], os.getcwd())
        model, model_fit = fit(train["System Price"].tolist())
        forecast, uppers, lowers = predict(model_fit, len(forecast_df))
        return forecast, uppers, lowers


def fit(train, verbose=False):
    ets = ExponentialSmoothing(train, trend='add', seasonal=24, damped_trend=True)
    model_fit = ets.fit(disp=0)
    if verbose:
        print(model_fit.summary())
    return ets, model_fit


# Use model to predict
def predict(model_fit, steps):
    forecast = model_fit.get_forecast(steps=steps)
    prediction = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.15)
    predictionsupper = [conf_int[i][1] for i in range(len(conf_int))]
    predictionslower = [conf_int[i][0] for i in range(len(conf_int))]
    return prediction, predictionsupper, predictionslower




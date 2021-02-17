import os
from data import data_handler
import pmdarima as pm
from pmdarima.arima import StepwiseContext
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd


class ArimaModel:
    def __init__(self):
        self.name = "AutoArima"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        start_date = forecast_df.at[0, "Date"]
        days_back = 14
        train_start_date = start_date - timedelta(days=days_back)
        train_end_date = train_start_date + timedelta(days=days_back-1)
        train = data_handler.get_data(train_start_date, train_end_date, ["System Price"], os.getcwd())
        model = self.fit(train[["System Price"]])
        forecast = self.predict(model, len(forecast_df))
        forecast_df["Forecast"] = forecast["Forecast"]
        forecast_df["Upper"] = forecast["Upper"]
        forecast_df["Lower"] = forecast["Lower"]
        return forecast_df

    # Fit new model to new data
    @staticmethod
    def fit(train, verbose=False):
        # with StepwiseContext(max_dur=100):
        arima_model = pm.auto_arima(train, seasonal=True, m=24, stepwise=True)
        if verbose:
            print(arima_model.summary())
        return arima_model

    # Use model to predict
    @staticmethod
    def predict(arima_, forecast_size):
        forecast, conf_int = arima_.predict(forecast_size, return_conf_int=True)
        results = np.column_stack((forecast, conf_int))  # Combine results into one array
        dataframe = pd.DataFrame(results)  # Add datetime period as index for forecast data
        dataframe.columns = ['Forecast', 'Lower', 'Upper']
        return dataframe

    # Save model to file
    # @staticmethod
    # def save_model(id_, name, model, forecast, scores, folder):
    #    save_pickle(id_, name, model, forecast, scores, folder)


if __name__ == '__main__':
    model_ = ArimaModel()
    start_date_ = "04.02.2019"
    end_date_ = "17.02.2019"
    time_list_ = data_handler.get_data(start_date_, end_date_, [], os.getcwd())
    time_list_["Forecast"] = np.nan
    time_list_["Upper"] = np.nan
    time_list_["Lower"] = np.nan
    forecast_ = model_.forecast(time_list_)
    print(forecast_)

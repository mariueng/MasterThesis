import os
from data import data_handler
from src.models.model_handler import *
from data import data_formatter
import pmdarima as pm
import pandas as pd
import pickle

start_time = "01.01.2018"
end_time = "31.01.2018"

data = data_handler.get_data(start_time, end_time, ["System Price"], os.getcwd())
data = data_formatter.combine_hour_day_month_year_to_datetime_index(data)
split = 150
train, test = data[:split], data[split:]


arima_model = pm.auto_arima(train, seasonal=True, m=12)

# Forecast
forecasts = arima_model.predict(test.shape[0])  # predict N (len(test)) steps into the future
# Add datetime period as index for forecast data
forecasts = data_formatter.array_to_dataframe_with_datetime_as_index(forecasts, test)
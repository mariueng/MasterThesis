# script for the most naive model: copy the 24 hours of last day for the following 14 days
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
from pathlib import Path
import numpy as np
import math
from src.system.generate_periods import get_random_periods
from data.data_handler import get_data
from src.models.naive_day.naive_day import get_prob_forecast
from src.preprocessing.arcsinh import arcsinh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid warnings from Tensorflow or any {'0', '1', '2'}
from keras.models import load_model
import statsmodels.api as sm
import matplotlib.pyplot as plt


class ExpertDay:
    def __init__(self):
        self.name = "Expert Day"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast = self.get_forecast(forecast_df)
        forecast_df["Forecast"] = forecast
        forecast_df["Upper"] = forecast_df["Forecast"] * 1.15
        forecast_df["Lower"] = forecast_df["Forecast"] * 0.9
        return forecast_df

    def get_forecast(self, forecast_df):
        model_fit = self.get_model()
        start_date = forecast_df.at[0, "Date"]
        train_start = start_date - timedelta(days=14)
        train_end = train_start + timedelta(days=13)
        df = get_data(train_start, train_end, ["System Price", "Weekday", "Week"], os.getcwd(), "d")
        #df, a, b = arcsinh.to_arcsinh(df, "System Price")
        col = "System Price"
        past_prices = df[col].tolist()
        weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        weeks = ["w{}".format(w) for w in range(1, 54)]
        forecast = []
        last_week_in_training = df.loc[len(df)-1, "Week"]
        last_day_in_training = df.loc[len(df)-1, "Weekday"]
        for i in range(14):
            x = {"1 day lag": past_prices[-1], "2 day lag": past_prices[-2], "3 day lag": past_prices[-3],
                 "1 week lag": past_prices[-7], "2 week lag": past_prices[-14]}
            weekday = df.loc[i % 7, "Weekday"]
            for val in weekdays.values():
                x[val] = 0
            day = weekdays[weekday]
            x[day] = 1
            for week in weeks:
                x[week] = 0
            current_week = last_week_in_training + math.floor((last_day_in_training+i) / 7)
            string_week = "w{}".format(current_week)
            x[string_week] = 1
            x = pd.DataFrame.from_dict(x, orient="index").transpose()
            prediction = model_fit.get_prediction(exog=x)
            forecasted_value = prediction.predicted_mean[0]
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
        hour_forecast = self.get_hour_forecast_from_mlp(forecast, start_date)
        #forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
        return hour_forecast

    @staticmethod
    def get_model():
        dir_path = str(Path(__file__).parent)
        m_path = dir_path + "\\expert_day.pickle"
        model_exists = os.path.isfile(m_path)
        if model_exists:
            model = sm.load(m_path)
        else:
            train_model(dir_path)
            model = sm.load(m_path)
        return model

    @staticmethod
    def get_hour_forecast_from_mlp(forecast, start):
        day_df = get_data(start, start+timedelta(days=13), ["Weekday", "Month"], os.getcwd(), "d")
        hour_forecast = []
        model = load_model(r"../models/mlp_day_profile/first_model")
        weekdays = ["d{}".format(i) for i in range(1, 8)]
        months = ["m{}".format(i) for i in range(1, 13)]
        col = ["Price"] + [w for w in weekdays] + [m for m in months]
        data = pd.DataFrame(columns=col)
        for i in range(len(day_df)):
            row = {"Price": forecast[i]}
            weekday = day_df.loc[i, "Weekday"]
            month = day_df.loc[i, "Month"]
            for j in range(1, 8):
                if weekday == j:
                    row["d{}".format(j)] = 1
                else:
                    row["d{}".format(j)] = 0
            for j in range(1, 13):
                if month == j:
                    row["m{}".format(j)] = 1
                else:
                    row["m{}".format(j)] = 0
            data = data.append(row, ignore_index=True)
        rows = []
        for index, row in data.iterrows():
            r = row.tolist()
            rows.append(r)
        test_x = np.asarray(rows).astype('float32')
        for i in range(len(test_x)):
            day_mean = forecast[i]
            x = test_x[i]
            x = x.reshape(1, len(x))
            prediction = model.predict(x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                                       workers=1, use_multiprocessing=False)
            hour_forecast.extend([day_mean + prediction[0][j] for j in range(len(prediction[0]))])
        return hour_forecast



def train_model(dir_path):
    start_date = dt(2014, 1, 1)
    end_date = dt(2019, 12, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday", "Week"], os.getcwd(), "d")
    #training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    col = "System Price"
    training_data["1 day lag"] = training_data[col].shift(1)
    training_data["2 day lag"] = training_data[col].shift(2)
    training_data["3 day lag"] = training_data[col].shift(3)
    training_data["1 week lag"] = training_data[col].shift(7)
    training_data["2 week lag"] = training_data[col].shift(14)
    training_data = training_data.dropna()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for index, row in training_data.iterrows():
        for j in range(1, 8):
            day = days[j - 1]
            if training_data.loc[index, "Weekday"] == j:
                training_data.loc[index, day] = 1
            else:
                training_data.loc[index, day] = 0
    for index, row in training_data.iterrows():
        for j in range(1, 54):
            if training_data.loc[index, "Week"] == j:
                training_data.loc[index, "w{}".format(j)] = 1
            else:
                training_data.loc[index, "w{}".format(j)] = 0
    #drop_cols = ["Date", "System Price", "Trans System Price", "Weekday", "Week"]
    drop_cols = ["Date", "System Price", "Weekday", "Week"]
    x_columns = [col for col in training_data.columns if col not in drop_cols]
    x = training_data[x_columns]
    col = "System Price"
    #col = "Trans System Price"
    y = training_data[[col]]
    model = sm.OLS(y, x).fit()
    print_model = model.summary()
    with open(dir_path + '\\expert_fit.txt', 'w') as fh:
        fh.write(print_model.as_text())
    model.save(dir_path + "\\expert_day.pickle")


def run(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0]-timedelta(days=21), period[1], ["System Price"], os.getcwd(), "d")
        result_df = true_price_df.merge(forecast_df, on=["Date"], how="outer")
        plt.subplots(figsize=(13, 7))
        plt.plot(result_df["Date"], result_df["Forecast"], label="Forecast")
        plt.plot(result_df["Date"], result_df["System Price"], label="True")
        plt.legend()
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.title("Expert day")
        plt.tight_layout()
        plt.show()
        result_list.append(result_df)
        print(result_df)


if __name__ == '__main__':
    # train_model(str(Path(__file__).parent))
    model_ = ExpertDay()
    periods_ = get_random_periods(1)
    run(model_, periods_)

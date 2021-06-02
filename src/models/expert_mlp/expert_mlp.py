from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
from pathlib import Path
import shutil
import numpy as np
from src.system.generate_periods import get_random_periods
from src.system.scores import get_all_point_metrics
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid warnings from Tensorflow or any {'0', '1', '2'}
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import operator
import pickle


class ExpertMLP:
    def __init__(self):
        self.name = "Expert MLP Sinh"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))
        self.prev_workdir = os.getcwd()
        os.chdir("\\".join(self.prev_workdir.split("\\")[:6]) + "\\models\\expert_mlp")
        self.model_fits = [load_model("h_models/expert_mlp_{}.pickle".format(h)) for h in range(24)]
        self.scalers = [pickle.load(open("h_scalers/scaler_{}.pickle".format(h), 'rb')) for h in range(24)]
        with open("h_columns/mlp_input_0.txt", 'r') as fh:
            self.columns = fh.readline().replace("\n", "").split(",")
            fh.close()

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = self.get_forecast(forecast_df)
        up = 1.15
        down = 0.90
        forecast_df = get_prob_forecast(forecast_df,  up, down)
        return forecast_df

    def get_forecast(self, forecast_df):
        start = forecast_df.loc[0, "Date"]
        end = forecast_df.loc[len(forecast_df)-1, "Date"]
        data = get_data(start, end, ["Weekday", "Holiday"], os.getcwd(), "h")
        train_start = start - timedelta(days=7)
        train_end = start - timedelta(days=1)
        df = get_data(train_start, train_end, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
        df = adjust_for_holiday(df, "System Price")
        df = df.append(data, ignore_index=True)
        idx = {'1 hour lag': 1, '2 hour lag': 2, '1 day lag': 24, '2 day lag': 48, '1 week lag': 168}
        weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        for i in range(168, 168 + len(forecast_df)):
            #print("i {}".format(i))
            x = pd.DataFrame(columns=self.columns)
            for key, value in idx.items():
                x.loc[0, key] = df.loc[i-value, "System Price"]
            yest_df = df[df["Date"] == df.loc[i, "Date"] - timedelta(days=1)]
            x.loc[0, "Max Yesterday"] = max(yest_df["System Price"])
            x.loc[0, "Min Yesterday"] = min(yest_df["System Price"])
            x.loc[0, "Midnight Yesterday"] = yest_df["System Price"].values[-1]
            for key, value in weekdays.items():
                x.loc[0, value] = 1 if df.loc[i, "Weekday"] == key else 0
            #for h in range(24):
                #x.loc[0, "h{}".format(h)] = 1 if df.loc[i, "Hour"] == h else 0
            #print(x.transpose())
            scaler = self.scalers[i%24]
            model_fit = self.model_fits[i%24]
            x = scaler.transform(x)
            #print(x)
            df.loc[i, "System Price"] = model_fit.predict(x)[0][0]
            #print("Predicted value {}".format(df.loc[i, "System Price"]))
            #print("-------------------------------\n")
        result = df.tail(336).reset_index(drop=True)
        result = result.rename(columns={"System Price": "Forecast"})
        result = adjust_for_holiday(result, "Forecast")
        forecast_df["Forecast"] = result["Forecast"]
        os.chdir(self.prev_workdir)
        return forecast_df


def get_prob_forecast(forecast_df, up, down):
    for index, row in forecast_df.iterrows():
        point_f = row["Forecast"]
        factor_up = get_factor_up(index, len(forecast_df), up)
        upper_f = point_f * factor_up
        forecast_df.at[index, "Upper"] = upper_f
        factor_down = get_factor_down(index, len(forecast_df), down)
        lower_f = point_f * factor_down
        forecast_df.at[index, "Lower"] = lower_f
    return forecast_df


def get_factor_up(index, horizon, up_factor):
    return up_factor + ((index / horizon) / 2) * up_factor


def get_factor_down(index, horizon, down_factor):
    return down_factor - ((index / horizon) / 2) * down_factor


def train_model():
    df = get_data("01.07.2014", "02.06.2019", ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
    df = adjust_for_holiday(df, "System Price")
    col = "System Price"
    df["1 hour lag"] = df[col].shift(1)
    df["2 hour lag"] = df[col].shift(2)
    df["1 day lag"] = df[col].shift(24)
    df["2 day lag"] = df[col].shift(48)
    df["1 week lag"] = df[col].shift(168)
    all_dates = pd.date_range(df.loc[0, "Date"], df.loc[len(df)-1, "Date"], freq="d")
    for date in all_dates:
        yesterday_df = df[df["Date"] == date - timedelta(days=1)]
        if not yesterday_df.empty:
            max_yesterday = max(yesterday_df[col])
            min_yesterday = min(yesterday_df[col])
            midnight_price = yesterday_df.tail(1)[col].values[0]
            todays_df = df[df["Date"] == date]
            for index in todays_df.index:
                df.loc[index, "Max Yesterday"] = max_yesterday
                df.loc[index, "Min Yesterday"] = min_yesterday
                df.loc[index, "Midnight Yesterday"] = midnight_price
    df = df.dropna().reset_index(drop=True)
    weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    for key, value in weekdays.items():
        df[value] = df.apply(lambda row: 1 if key == row["Weekday"] else 0, axis=1)
    #for i in range(24):
        #df["h{}".format(i)] = df.apply(lambda row: 1 if row["Hour"] == i else 0, axis=1)
    drop_cols = ["Date", "Hour", "System Price", "Weekday"]
    for h in range(0, 24):
        sub_df = df[df["Hour"]==h]
        x_columns = [col for col in df.columns if col not in drop_cols]
        y = df[[col]].values
        scaler.fit(df[x_columns])
        pickle.dump(scaler, open("h_scalers/scaler_{}.pickle".format(h), 'wb'))
        with open("h_columns/mlp_input_{}.txt".format(h), 'w') as fh:
            fh.write(",".join(x_columns) + "\n")
            fh.close()
        x = scaler.transform(df[x_columns])
        model = get_mlp_model(x.shape[1])
        print(model.summary())
        epocks = 25
        earlystopping = EarlyStopping(monitor="loss", mode="min", patience=5, restore_best_weights=True)
        model.fit(x, y, epochs=epocks, batch_size=32,  callbacks=[earlystopping])
        model.save("h_models/expert_mlp_{}.pickle".format(h))


def adjust_for_holiday(df, col):
    operation = operator.mul if col == "System Price" else operator.truediv
    df[col] = df.apply(lambda row: operation(row[col], get_hol_factor()) if
    row["Holiday"] == 1 and row["Weekday"] != 7 else row[col], axis=1)
    return df.drop(columns=["Holiday"])


def get_hol_factor():
    f = 1.1158
    return f


def get_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(units=input_dim, input_dim=input_dim))
    model.add(Dense(math.ceil(input_dim * 2)))
    model.add(Dense(math.ceil(input_dim / 2)))
    model.add(Dense(1, activation="relu"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def validate_day_model(input_col):
    model = ExpertMLP()
    model_fit = model.get_model(input_col)
    periods = get_all_2019_periods()
    # periods = get_random_periods(10)
    forecasts = pd.DataFrame(columns=["Period", "Date", "System Price", "Forecast"])
    results = pd.DataFrame(columns=["Period", "mape", "smape", "mae", "rmse"])
    result_path = reset_result_directory(input_col)
    for j in range(len(periods)):
        start = periods[j][0]
        end = periods[j][1]
        print("Forecasting from {} to {}".format(start, end))
        forecast_df = get_data(start, end, ["System Price"], os.getcwd(), "d")
        data_df = get_data(start, end, input_col, os.getcwd(), "d")
        past_prices = get_data(start - timedelta(days=14), start - timedelta(days=1), ["System Price"], os.getcwd(),
                               "d")["System Price"].tolist()
        plot_df = get_data(start - timedelta(days=7), end, ["System Price"], os.getcwd(), "d")
        forecast = []
        for i in range(len(forecast_df)):
            x = get_x_input(i, past_prices, data_df, input_col)
            x = scaler.transform(x)
            prediction = model_fit.predict(x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                                           workers=1, use_multiprocessing=False)
            forecasted_value = prediction[0][0]
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
            past_prices.pop(0)
        forecast_df["Forecast"] = forecast
        plot_df = pd.merge(plot_df, forecast_df[["Date", "Forecast"]], on="Date", how="outer")
        plot_forecast(plot_df, j + 1, result_path)
        forecast_df["Period"] = j + 1
        forecasts = forecasts.append(forecast_df, ignore_index=True)
        point_performance = get_all_point_metrics(forecast_df)
        point_performance["Period"] = j + 1
        results = results.append(point_performance, ignore_index=True)
    calculate_performance(forecasts, results, model.get_name(), model.get_time().replace("_", " "), input_col,
                          result_path)


def reset_result_directory(input_col):
    res_dir = "results/{}".format("_".join(input_col))
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir, ignore_errors=True)
    os.makedirs(res_dir)
    os.makedirs(res_dir + "/plots")
    return res_dir


def calculate_performance(forecasts, results, m_name, m_time, inputs, res_path):
    forecasts.to_csv(res_path + "/forecast.csv", index=False, float_format='%.3f')
    results.to_csv(res_path + "/metrics.csv", index=False, float_format='%.3f')
    avg_mape = round(results["mape"].mean(), 3)
    avg_smape = round(results["smape"].mean(), 3)
    avg_mae = round(results["mae"].mean(), 3)
    avg_rmse = round(results["rmse"].mean(), 3)
    if os.path.exists('input_results.csv'):
        input_res = pd.read_csv('input_results.csv', header=0)
    else:
        columns = ['inputs', 'mape', 'smape', 'mae', 'rmse']
        input_res = pd.DataFrame(columns=columns)
    row = {'inputs': " - ".join(inputs), 'mape': avg_mape, 'smape': avg_smape, 'mae': avg_mae, 'rmse': avg_rmse}
    input_res = input_res.append(row, ignore_index=True)
    input_res.to_csv('input_results.csv', index=False, float_format='%.3f')
    summary = open(res_path + "/performance.txt", "w")
    summary.write("-- Performance Summary for '{}', created {} --\n\n".format(m_name, m_time))
    line = "Point performance:\nMape:\t {}\nSmape:\t{}\nMae:\t{}\nRmse:\t{}\n".format(
        avg_mape, avg_smape, avg_mae, avg_rmse)
    summary.write(line)
    if len(results) > 5:
        results = results.set_index("Period")
        best = results.sort_values(by="smape").head(3)[["smape"]].to_dict()["smape"]
        best = {key: round(best[key], 2) for key in best}
        summary.write("\n\nThree best SMAPE performances:\t{}".format(best))
        worst = results.sort_values(by="smape").tail(3)[["smape"]].to_dict()["smape"]
        worst = {key: round(worst[key], 2) for key in worst}
        summary.write("\nThree worst SMAPE performances:\t{}".format(worst))
    summary.close()



def plot_forecast(df, period_no, result_path):
    label_pad = 12
    title_pad = 20
    fig, ax = plt.subplots(figsize=(13, 7))
    plt.plot(df["Date"], df["System Price"], label="True", color="steelblue", linewidth=2)
    plt.plot(df["Date"], df["Forecast"], label="Forecast", color="firebrick")
    for line in plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ymax = max(max(df["System Price"]), max(df["Forecast"])) * 1.1
    ymin = min(min(df["System Price"]), min(df["Forecast"])) * 0.95
    plt.ylim(ymin, ymax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    end_date = df.loc[len(df) - 1, "Date"]
    start_date = end_date - timedelta(days=13)
    start_day_string = dt.strftime(start_date, "%d %b")
    end_day_string = dt.strftime(end_date, "%d %b")
    plt.title("Result from '{}' - {} to {}".format("Expert MLP", start_day_string, end_day_string), pad=title_pad)
    plt.tight_layout()
    plot_path = str(period_no) + "_" + start_day_string.replace(" ", "") + "_" + end_day_string.replace(" ",
                                                                                                        "") + ".png"
    out_path = result_path + "/plots/" + plot_path
    plt.savefig(out_path)
    plt.close()



def run(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0] - timedelta(days=21), period[1], ["System Price"], os.getcwd(), "d")
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
    print("Running script")
    # train_model()
    model_ = ExpertMLP()
    periods_ = get_random_periods(1)
    run(model_, periods_)

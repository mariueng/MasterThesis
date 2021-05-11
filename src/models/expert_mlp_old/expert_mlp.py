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


class ExpertMLP:
    def __init__(self):
        self.name = "Expert MLP"
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
        params = get_parameters_dataframe(model_fit).columns.tolist()
        input_cols = [i for i in params if i != "Period" and "lag" not in i]
        start_date = forecast_df.at[0, "Date"]
        train_start = start_date - timedelta(days=14)
        train_end = train_start + timedelta(days=13)
        # df, a, b = arcsinh.to_arcsinh(df, "System Price")
        col = "System Price"
        past_prices = get_data(train_start, train_end, ["System Price"], os.getcwd(), "d")[col].tolist()
        data_df = get_data(start_date, start_date + timedelta(days=13), input_cols, os.getcwd(), "d")
        forecast = []
        for i in range(14):
            x = get_x_input(i, model_fit.params, past_prices, data_df, input_cols)
            prediction = model_fit.get_prediction(exog=x)
            forecasted_value = prediction.predicted_mean[0]
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
        hour_forecast = self.get_hour_forecast_from_mlp(forecast, start_date)
        # forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
        return hour_forecast

    @staticmethod
    def get_model(input_cols):
        dir_path = str(Path(__file__).parent)
        model_name = dir_path + "\\models\\" + "_".join(input_cols)
        model_exists = os.path.exists(model_name)
        if model_exists:
            print("Retrieving model from {}".format(model_name))
            model = load_model(model_name)
        else:
            print("Training model from {}".format(input_cols))
            train_model(dir_path, input_cols)
            model = load_model(model_name)
        return model

    @staticmethod
    def get_hour_forecast_from_mlp(forecast, start):
        day_df = get_data(start, start + timedelta(days=13), ["Weekday", "Month", "Temp Norway"], os.getcwd(), "d")
        hour_forecast = []
        model = load_model(r"../models/mlp_day_profile/first_model")
        # model = load_model(r"../mlp_day_profile/first_model")
        col = ["Price"] + [w for w in weekdays] + [m for m in months] + ["Temp Norway"]
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
            row["Temp Norway"] = day_df.loc[i, "Temp Norway"]
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


weekdays = ["d{}".format(d) for d in range(1, 8)]
months = ["m{}".format(m) for m in range(1, 13)]


def train_model(dir_path, input_col):
    print("Training model for {}".format(", ".join(input_col)))
    model_name = dir_path + "\\models\\" + "_".join(input_col)
    if not os.path.exists(model_name):
        start_date = dt(2014, 1, 1)
        end_date = dt(2018, 12, 31)
        columns = ["System Price"] + input_col
        x, y = get_x_and_y_dataset(start_date, end_date, columns)
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        model = get_mlp_model(len(x.columns), 2 * math.ceil(len(x.columns) / 3))
        callback = EarlyStopping(monitor="loss", patience=20)
        # model.fit(x, y, epochs=1000, batch_size=32, callbacks=[callback])
        # epocks = 100 + 50 * (len(input_col)-2)
        epocks = 1000
        model.fit(x_scaled, y, epochs=epocks, batch_size=32, callbacks=[callback])
        model.save(model_name)


def get_mlp_model(input_layer, hidden_layer):
    model = Sequential()
    model.add(Dense(units=hidden_layer, input_dim=input_layer, activation='tanh'))
    model.add(Dense(math.ceil(hidden_layer / 2)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
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


def get_x_input(id_x, past_prices, data_df, input_col):
    lags = ["1 day lag", "2 day lag", "3 day lag", "1 week lag", "2 week lag"]
    idx_back = [-1, -2, -3, -7, -14]
    data_cols = [i for i in input_col if i not in ["Weekday", "Month"]]
    variables = data_cols + lags + weekdays + months
    x_df = pd.DataFrame(columns=variables)
    x_dict = {}
    for c in data_cols:
        x_dict[c] = data_df.loc[id_x, c]
    time_cols = {"Weekday": "d", "Month": "m"}
    time_list = {"Weekday": weekdays, "Month": months}
    for j in range(len(lags)):
        x_dict[lags[j]] = past_prices[idx_back[j]]
    for col in time_cols.keys():
        if col in input_col:
            current = data_df.loc[id_x, col]
            for val in time_list[col]:
                x_dict[val] = 0
            x_dict["{}{}".format(time_cols[col], current)] = 1
    x_df = x_df.append(x_dict, ignore_index=True)
    return x_df


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


def get_parameters_dataframe(model_fit):
    columns_to_dataframe = ["Period"]
    p = model_fit.params
    keys = p.keys().tolist()
    for k in keys:
        if k not in weekdays and k not in weeks and k not in months:
            columns_to_dataframe.append(k)
    if all(x in keys for x in weekdays):
        columns_to_dataframe.append("Weekday")
    if all(x in keys for x in weeks):
        columns_to_dataframe.append("Week")
    if all(x in keys for x in months):
        columns_to_dataframe.append("Month")
    df = pd.DataFrame(columns=columns_to_dataframe)
    return df


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


def get_x_and_y_dataset(start, end, columns):
    if start > dt(2014, 1, 14):
        start = start - timedelta(days=13)
    data = get_data(start, end, columns, os.getcwd(), "d")
    col = "System Price"
    data["1 day lag"] = data[col].shift(1)
    data["2 day lag"] = data[col].shift(2)
    data["3 day lag"] = data[col].shift(3)
    data["1 week lag"] = data[col].shift(7)
    data["2 week lag"] = data[col].shift(14)
    data = data.dropna().reset_index()
    drop_cols = ["index", "Date", "System Price", "Month", "Weekday"]
    if "Weekday" in columns:
        data = one_hot_encoding("Weekday", "d", 7, data)
    if "Month" in columns:
        data = one_hot_encoding("Month", "m", 12, data)
    x_columns = [col for col in data.columns if col not in drop_cols]
    x = data[x_columns]
    y = data[[col]]
    return x, y


def one_hot_encoding(col, prefix, length, data):
    for index, row in data.iterrows():
        for j in range(1, length + 1):
            if data.loc[index, col] == j:
                data.loc[index, "{}{}".format(prefix, j)] = 1
            else:
                data.loc[index, "{}{}".format(prefix, j)] = 0
    return data


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
    data_sources = ["Temp Norway", "Total Hydro Dev", "Wind DK", "Prec Forecast"]
    time_dummies = ["Weekday", "Month"]
    length = len(data_sources)
    all_inputs = {frozenset({e for e, b in zip(data_sources, f'{i:{length}b}') if b == '1'}) for i in
                  range(2 ** length)}
    for i_set in all_inputs:
        # inputs = time_dummies + list(i_set)
        inputs = time_dummies + ["Demand", "Coal"]
        train_model(str(Path(__file__).parent), inputs)
        validate_day_model(inputs)
        assert False
    # model_ = ExpertDay()
    # periods_ = get_random_periods(1)
    # run(model_, periods_)

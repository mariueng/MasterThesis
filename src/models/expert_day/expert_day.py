# script for the most naive model: copy the 24 hours of last day for the following 14 weekdays
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
from pathlib import Path
import numpy as np
import math
from src.system.generate_periods import get_random_periods
from src.system.scores import get_all_point_metrics
from src.system.generate_periods import get_all_2019_periods
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
        temp_df = get_data(start_date, start_date + timedelta(days=13), ["T Nor"], os.getcwd(), "d")
        # df, a, b = arcsinh.to_arcsinh(df, "System Price")
        col = "System Price"
        past_prices = df[col].tolist()
        forecast = []
        last_week_in_training = df.loc[len(df) - 1, "Week"]
        last_day_in_training = df.loc[len(df) - 1, "Weekday"]
        for i in range(14):
            x = {"Temp. Norway": temp_df.loc[i, "T Nor"], "1 day lag": past_prices[-1], "2 day lag": past_prices[-2],
                 "3 day lag": past_prices[-3],
                 "1 week lag": past_prices[-7], "2 week lag": past_prices[-14]}
            weekday = df.loc[i % 7, "Weekday"]
            for val in weekdays.values():
                x[val] = 0
            day = weekdays[weekday]
            x[day] = 1
            for week in weeks:
                x[week] = 0
            current_week = last_week_in_training + math.floor((last_day_in_training + i) / 7)
            string_week = "w{}".format(current_week)
            x[string_week] = 1
            x = pd.DataFrame.from_dict(x, orient="index").transpose()
            prediction = model_fit.get_prediction(exog=x)
            forecasted_value = prediction.predicted_mean[0]
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
        hour_forecast = self.get_hour_forecast_from_mlp(forecast, start_date)
        # forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
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
        day_df = get_data(start, start + timedelta(days=13), ["Weekday", "Month", "T Nor"], os.getcwd(), "d")
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
            row["Temp Norway"] = day_df.loc[i, "T Nor"]
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
weeks = ["w{}".format(w) for w in range(1, 54)]
months = ["m{}".format(m) for m in range(1, 13)]
divide_on_thousand_list = ["Total Hydro Dev", "Total Hydro", "Wind DK", "Total Vol"]


def train_model(dir_path, input_col):
    start_date = dt(2014, 1, 1)
    end_date = dt(2018, 12, 31)
    columns = ["System Price"] + input_col
    x, y = get_x_and_y_dataset(start_date, end_date, columns)
    model = sm.OLS(y, x).fit()
    print_model = model.summary()
    with open(dir_path + '\\expert_fit.txt', 'w') as fh:
        fh.write(print_model.as_text())
    model.save(dir_path + "\\expert_day.pickle")


def validate_day_model(input_col):
    model = ExpertDay()
    model_fit = model.get_model()
    periods = get_all_2019_periods()
    # periods = get_random_periods(10)
    forecasts = pd.DataFrame(columns=["Period", "Date", "System Price", "Forecast"])
    results = pd.DataFrame(columns=["Period", "mape", "smape", "mae", "rmse"])
    weights_df = get_parameters_dataframe(model_fit)
    [f.unlink() for f in Path("validation/plots").glob("*") if f.is_file()]
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
            x = get_x_input(i, model_fit.params, past_prices, data_df, input_col)
            prediction = model_fit.get_prediction(exog=x)
            forecasted_value = prediction.predicted_mean[0]
            weights_df = append_day_weight_to_df(j+1, weights_df, x, model_fit, input_col)
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
            past_prices.pop(0)
        forecast_df["Forecast"] = forecast
        plot_df = pd.merge(plot_df, forecast_df[["Date", "Forecast"]], on="Date", how="outer")
        plot_forecast(plot_df, j + 1)
        forecast_df["Period"] = j + 1
        forecasts = forecasts.append(forecast_df, ignore_index=True)
        point_performance = get_all_point_metrics(forecast_df)
        point_performance["Period"] = j + 1
        results = results.append(point_performance, ignore_index=True)
    calculate_performance(forecasts, results, model.get_name(), model.get_time().replace("_", " "), input_col)
    plot_weights(weights_df)


def get_x_input(id, params, past_prices, data_df, input_col):
    variables = params.keys().tolist()
    x_df = pd.DataFrame(columns=variables)
    x_dict = {}
    for c in divide_on_thousand_list:
        if c in variables:
            x_dict[c] = data_df.loc[id, c] / 1000
    if "Temp. Norway" in variables:
        x_dict["Temp. Norway"] = data_df.loc[id, "T Nor"]
    lags = ["1 day lag", "2 day lag", "3 day lag", "1 week lag", "2 week lag"]
    idx_back = [-1, -2, -3, -7, -14]
    for j in range(len(lags)):
        x_dict[lags[j]] = past_prices[idx_back[j]]
    if "Weekday" in input_col:
        current_day = data_df.loc[id, "Weekday"]
        for val in weekdays:
            x_dict[val] = 0
        x_dict["d{}".format(current_day)] = 1
    if "Week" in input_col:
        current_week = data_df.loc[id, "Week"]
        for week in weeks:
            x_dict[week] = 0
        x_dict["w{}".format(current_week)] = 1
    if "Month" in input_col:
        current_month = data_df.loc[id, "Month"]
        for month in months:
            x_dict[month] = 0
        x_dict["m{}".format(current_month)] = 1
    x_df = x_df.append(x_dict, ignore_index=True)
    return x_df


def plot_weights(weights_df):
    for col in weights_df.columns:
        weights_df[col] = abs(weights_df[col])
    row = {}
    for col in weights_df.columns:
        row[col] = weights_df[col].mean()
    row.pop("Period")
    keys = row.keys()
    values = row.values()
    plt.figure(figsize=(10, 7))
    plt.pie(values, startangle=90, frame=True, textprops=None,
            autopct=lambda p: '{:.0f}'.format((p / 100) * sum(values), p) if p > 0 else "", pctdistance=0.53)
    centre_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
    plt.gca().add_artist(centre_circle)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(keys, loc="upper right", fancybox=True, shadow=True)
    plt.axis('equal')
    plt.title("Model Influence per Parameter in OLS")
    plt.tight_layout()
    plt.savefig("validation/parameter_influence.png")
    plt.close()


def append_day_weight_to_df(period, weights_df, x, model_fit, input_col):
    row = {}
    base_columns = [i for i in x.columns.tolist() if i not in weekdays + weeks + months]
    params = model_fit.params
    for col in base_columns:
        row[col] = abs(x.loc[0, col] * params[col])
    if "Weekday" in input_col:
        for col in weekdays:
            if x.loc[0, col] == 1:
                row["Weekday"] = abs(x.loc[0, col] * params[col])
    if "Week" in input_col:
        for col in weeks:
            if x.loc[0, col] == 1:
                row["Week"] = abs(x.loc[0, col] * params[col])
    if "Month" in input_col:
        for col in months:
            if x.loc[0, col] == 1:
                row["Month"] = abs(x.loc[0, col] * params[col])
    sum_values = sum(row.values())
    for key in row.keys():
        row[key] = (100 * row[key]) / sum_values
    assert round(sum(row.values()), 1) == 100
    row["Period"] = int(period)
    weights_df = weights_df.append(row, ignore_index=True)
    return weights_df


def calculate_performance(forecasts, results, m_name, m_time, inputs):
    forecasts.to_csv(r"validation/forecast.csv", index=False, float_format='%.3f')
    results.to_csv(r"validation/metrics.csv", index=False, float_format='%.3f')
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
    summary = open(r"validation/performance.txt", "w")
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


def plot_forecast(df, period_no):
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
    plt.title("Result from '{}' - {} to {}".format("Expert Day", start_day_string, end_day_string), pad=title_pad)
    plt.tight_layout()
    plot_path = str(period_no) + "_" + start_day_string.replace(" ", "") + "_" + end_day_string.replace(" ",
                                                                                                        "") + ".png"
    out_path = "validation/plots/" + plot_path
    plt.savefig(out_path)
    plt.close()


def get_x_and_y_dataset(start, end, columns):
    if start > dt(2014, 1, 14):
        start = start - timedelta(days=13)
    data = get_data(start, end, columns, os.getcwd(), "d")
    if "T Nor" in columns:
        data = data.rename(columns={"T Nor": "Temp. Norway"})
    for c in divide_on_thousand_list:
        if c in columns:
            data[c] = data[c] / 1000
    # training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    col = "System Price"
    data["1 day lag"] = data[col].shift(1)
    data["2 day lag"] = data[col].shift(2)
    data["3 day lag"] = data[col].shift(3)
    data["1 week lag"] = data[col].shift(7)
    data["2 week lag"] = data[col].shift(14)
    data = data.dropna()
    drop_cols = ["Date", "System Price"]
    if "Weekday" in columns:
        data = one_hot_encoding("Weekday", "d", 7, data)
        drop_cols.append("Weekday")
    if "Week" in columns:
        data = one_hot_encoding("Week", "w", 53, data)
        drop_cols.append("Week")
    if "Month" in columns:
        data = one_hot_encoding("Month", "m", 12, data)
        drop_cols.append("Month")
    x_columns = [col for col in data.columns if col not in drop_cols]
    x = data[x_columns]
    y = data[[col]]
    return x, y


def one_hot_encoding(col, prefix, length, data):
    for index, row in data.iterrows():
        for j in range(1, length+1):
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
    col_1 = ["Weekday", "Month"]
    col_2 = ["Weekday", "Week"]
    col_3 = ["Weekday", "Month", "T Nor"]
    col_4 = ["Weekday", "Week", "T Nor"]
    col_5 = ["Weekday", "Month", "Total Hydro"]
    col_6 = ["Weekday", "Month", "Total Hydro Dev"]
    col_7 = ["Weekday", "Month", "Wind DK"]
    col_8 = ["Weekday", "Month", "T Nor", "Wind DK"]
    col_9 = ["Weekday", "Month", "Total Vol"]
    col_10 = ["Weekday", "Month", "Total Vol", "T Nor"]
    col_11 = ["Total Hydro Dev"]
    inputs = [col_11]
    # --------------------------------------------
    for col_ in inputs:
        input_col_ = col_
        # --------------------------------------------
        train_model(str(Path(__file__).parent), input_col_)
        validate_day_model(input_col_)
    # model_ = ExpertDay()
    # periods_ = get_random_periods(1)
    # run(model_, periods_)

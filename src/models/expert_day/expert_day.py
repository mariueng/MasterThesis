from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
from pathlib import Path
import numpy as np
from src.system.generate_periods import get_random_periods
from src.system.scores import get_all_point_metrics
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from src.preprocessing.arcsinh import arcsinh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid warnings from Tensorflow or any {'0', '1', '2'}
from keras.models import load_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import operator
import warnings


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

    @staticmethod
    def forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = get_forecast(forecast_df)
        up = 1.15
        down = 0.90
        forecast_df = get_prob_forecast(forecast_df,  up, down)
        return forecast_df


def get_forecast(forecast_df):
    prev_workdir = os.getcwd()
    os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\expert_day")
    profiles = pd.read_csv("hourly_decomp.csv")
    start = forecast_df.loc[0, "Date"].date()
    end = forecast_df.loc[len(forecast_df)-1, "Date"].date()
    h_data = get_data(start, end, ["Weekday", "Month", "Holiday"], os.getcwd(), "h")
    data = get_data(start, end, ["Weekday", "Holiday"], os.getcwd(), "d")
    train_start = start - timedelta(days=7)
    train_end = start - timedelta(days=1)
    hist = get_data(train_start, train_end, ["System Price", "Weekday", "Holiday"], os.getcwd(), "d")
    hist = adjust_for_holiday(hist, "System Price")
    norm_hist = get_normalised_last_week(hist)
    min_last_week = min(norm_hist["System Price"])
    max_last_week = max(norm_hist["System Price"])
    hist = hist.append(data, ignore_index=True)
    daily_models = get_daily_models_as_dict()
    for i in range(7, len(hist)):
        weekday = hist.loc[i, "Weekday"]
        model_fit = daily_models[weekday]
        if i == 14:
            norm_last_week = get_normalised_last_week(hist.loc[7:13])
            min_last_week, max_last_week = (min(norm_last_week["System Price"]), max(norm_last_week["System Price"]))
        one_day_lag = hist.loc[i-1, "System Price"]
        two_day_lag = hist.loc[i-2, "System Price"]
        one_week_lag = hist.loc[i-7, "System Price"]
        last_sunday = hist[hist["Weekday"] == 7]["System Price"].dropna("").values[-1]
        x = {'1 day lag':one_day_lag, '2 day lag': two_day_lag, '1 week lag': one_week_lag, 'Max Last Week':
            max_last_week, 'Min Last Week': min_last_week, 'Last Sunday': last_sunday}
        x = pd.DataFrame.from_dict(x, orient="index").transpose()
        x = x.assign(intercept=1)
        prediction = model_fit.get_prediction(x)
        hist.loc[i, "System Price"] = prediction.predicted_mean[0]
    df = hist.loc[7:, :].reset_index(drop=True)
    df = df.rename(columns={"System Price": "Forecast"})
    df = adjust_for_holiday(df, "Forecast")
    for i in range(len(forecast_df)):
        date = forecast_df.loc[i, "Date"]
        day_prediction = df[df["Date"] == date]["Forecast"].values[0]
        hour = forecast_df.loc[i, "Hour"]
        weekday = h_data.loc[i, "Weekday"] if h_data.loc[i, "Holiday"] != 1 else 7
        month = h_data.loc[i, "Month"]
        profile = profiles.iloc[(month-1)*24*7 + (weekday-1)*24 + hour]
        forecast_df.loc[i, "Forecast"] = profile["Factor"] * day_prediction
    os.chdir(prev_workdir)
    return forecast_df


def get_daily_models_as_dict():
    daily_models = {}
    for i in range(1, 8):
        daily_models[i] = sm.load(os.getcwd() + '\\d_models\\expert_model_{}.pickle'.format(i))
    return daily_models


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
    start_date = dt(2014, 7, 1)
    first_year = start_date.year
    end_date = dt(2019, 6, 2)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday", "Holiday", "Week"], os.getcwd(), "d")
    training_data = adjust_for_holiday(training_data, "System Price")
    col = "System Price"
    training_data["1 day lag"] = training_data[col].shift(1)
    training_data["2 day lag"] = training_data[col].shift(2)
    training_data["1 week lag"] = training_data[col].shift(7)
    training_data["Week"] = (training_data["Date"].dt.year - first_year) * 52 + training_data["Week"]
    all_weeks = training_data["Week"].unique()
    for week in all_weeks:
        last_week_df = training_data[training_data["Week"] == week - 1]
        if len(last_week_df) == 7:
            normalised_last_week = get_normalised_last_week(last_week_df)
            this_week_df = training_data[training_data["Week"] == week]
            max_last_week = max(normalised_last_week[col])
            min_last_week = min(normalised_last_week[col])
            sunday_price = last_week_df.tail(1)[col].values[0]
            for index in this_week_df.index:
                training_data.loc[index, "Max Last Week"] = max_last_week
                training_data.loc[index, "Min Last Week"] = min_last_week
                training_data.loc[index, "Last Sunday"] = sunday_price
    training_data = training_data.dropna().reset_index(drop=True)
    drop_cols = ["Date", "System Price", "Weekday", "Week"]
    x_columns = [col for col in training_data.columns if col not in drop_cols]
    for i in range(1, 8):
        day_data = training_data.loc[training_data["Weekday"] == i]
        y = day_data[[col]]
        x = day_data[x_columns]
        x = x.assign(intercept=1)
        model = sm.OLS(y, x, hasconst=True).fit()
        print_model = model.summary()
        with open(os.getcwd() + '\\d_fits\\expert_fit_{}.txt'.format(i), 'w') as fh:
            fh.write(print_model.as_text())
            fh.close()
        model.save(os.getcwd() + "\\d_models\\expert_model_{}.pickle".format(i))


def get_normalised_last_week(last_week_df):
    warnings.filterwarnings("ignore")
    df = last_week_df.copy()
    d = {1: 1.028603, 2: 1.034587, 3: 1.0301834, 4: 1.033991, 5: 1.014928, 6: 0.941950, 7: 0.915758}
    df["Factor"] = [d[weekday] for weekday in df["Weekday"]]
    df["System Price"] = df["System Price"] / df["Factor"]
    return df[["Date", "System Price", "Weekday"]]


def adjust_for_holiday(df, col):
    operation = operator.mul if col == "System Price" else operator.truediv
    df[col] = df.apply(lambda row: operation(row[col], get_hol_factor()) if
    row["Holiday"] == 1 and row["Weekday"] != 7 else row[col], axis=1)
    return df.drop(columns=["Holiday"])


def get_hol_factor():
    f = 1.1158
    return f


def train_hour_coefficients():
    start = "01.07.2014"
    df = get_data(start, "02.06.2019", ["System Price", "Weekday", "Month", "Holiday"], os.getcwd(), "h")
    df = adjust_for_holiday(df, "System Price")
    daily = get_data(start, "02.06.2019", ["System Price", "Weekday", "Holiday"], os.getcwd(), "d")
    daily = adjust_for_holiday(daily, "System Price")
    daily = daily.rename(columns={"System Price": "Daily Price"})
    daily = daily.drop(columns=["Weekday"])
    df = df.merge(daily, on="Date")
    df["Factor"] = df["System Price"] / df["Daily Price"]
    df = df.drop(columns=["System Price", "Date", "Daily Price"])
    grouped = df.groupby(by=["Month", "Weekday", "Hour"]).median().reset_index()
    grouped.to_csv("hourly_decomp.csv", index=False, float_format="%g")


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
    print("Running methods")
    train_model()
    train_hour_coefficients()
    # model_ = ExpertDay()
    # periods_ = get_random_periods(1)
    # run(model_, periods_)

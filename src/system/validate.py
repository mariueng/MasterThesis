# script for validation test
from generate_periods import get_four_periods_median_method
from generate_periods import get_one_period
from src.models.sarima import sarima
from src.models.ets import ets
from data.data_handler import get_data
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import datetime as dt
import math
import time


def validate(model, periods, plot):  # plot is boolean value, if you want to plot results
    print("-- Running validation using Model '{}' --".format(model.get_name()))
    start_time = time.time()
    result_list = []
    for i in range(len(periods)):
        forecast_df = get_forecast(model, periods[i])
        true_price_df = get_data(periods[i][0], periods[i][1], ["System Price"], os.getcwd())
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_list.append(result_df)
    result_path = get_result_folder(model)
    if plot:
        for i in range(len(result_list)):
            plot_result(result_list[i], periods[i], result_path, model.get_name(), str(i + 1))
    save_forecast(result_list, result_path)
    calculate_performance(result_list, result_path)
    analyze_performance(result_path, model)
    print("\nResults are saved to 'src/" + result_path[3:] + "'")
    elapsed_min = int((time.time() - start_time) // 60)
    elapsed_sec = math.ceil((time.time() - start_time) % 60)
    print("-- Validation time:\t{0:02d}:{1:02d} --".format(elapsed_min, elapsed_sec))


def get_forecast(model, period):
    print("Forecasting from {} to {}".format(period[0], period[1]))
    time_df = get_data(period[0], period[1], [], os.getcwd())
    time_df["Forecast"] = np.nan
    time_df["Upper"] = np.nan
    time_df["Lower"] = np.nan
    forecast_df = model.forecast(time_df)
    return forecast_df


def get_result_folder(model):
    name = model.get_name()
    # time = model.get_time() REMOVE "#" if you want to keep old folder with same model name
    # folder_name = (name+"_"+time).replace(" ", "")
    folder_name = name.replace(" ", "")
    folder_path = "../results/validation/" + folder_name
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # delete old
    os.makedirs(folder_path)  # create new
    return folder_path


def plot_result(result, period, dir_path, model_name, period_no):
    label_pad = 12
    title_pad = 20
    img_size = (13, 7)
    true_color = "steelblue"
    fc_color = "firebrick"
    result["Hour"] = pd.to_datetime(result['Hour'], format="%H").dt.time
    result["DateTime"] = result.apply(lambda r: dt.datetime.combine(r['Date'], r['Hour']), 1)
    print("Plotting from {} to {}".format(period[0], period[1]))
    fig, ax = plt.subplots(figsize=img_size)
    plt.plot(result["DateTime"], result["System Price"], label="True", color=true_color, linewidth=2)
    plt.plot(result["DateTime"], result["Forecast"], label="Forecast", color=fc_color)
    plt.gca().fill_between(result["DateTime"], result["Upper"], result["Lower"],
                           facecolor='gainsboro', interpolate=True, label="Interval")
    for line in plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ymax = max(result["System Price"])*1.4
    ymin = min(result["System Price"])*0.7
    plt.ylim(ymin, ymax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    start_day_string = dt.datetime.strftime(period[0], "%d %b")
    end_day_string = dt.datetime.strftime(period[1], "%d %b")
    plt.title("Result from '{}' - {} to {}".format(model_name, start_day_string, end_day_string), pad=title_pad)
    plt.tight_layout()
    plot_path = period_no + "_" + start_day_string.replace(" ", "") + "_" + end_day_string.replace(" ", "") + ".png"
    out_path = dir_path + "/" + plot_path
    plt.savefig(out_path)
    plt.close()


def save_forecast(result_list, result_path):
    for i in range(len(result_list)):
        result = result_list[i]
        result["Period"] = i+1
    all_forecasts = pd.concat(result_list, ignore_index=True)
    all_forecasts = all_forecasts[["Period", "DateTime", "System Price", "Forecast", "Upper", "Lower"]]
    path = result_path + "/forecast.csv"
    all_forecasts.to_csv(path, index=False, float_format='%.3f')


def calculate_performance(result_list, dir_path):
    print("Calculating performance for all periods")
    performance_df = pd.DataFrame(columns=["Period", "From Date", "To Date", "MAPE", "SMAPE", "MAE", "RMSE", "COV",
                                           "CE", "IS"])
    for i in range(len(result_list)):
        result = result_list[i]
        period = i + 1
        from_date = result.at[0, "Date"]
        to_date = result.at[len(result) - 1, "Date"]
        mape = calculate_mape(result)
        smape = calculate_smape(result)
        mae = calculate_mae(result)
        rmse = calculate_rmse(result)
        cov, ce = calculate_coverage_error(result)
        int_score = calculate_interval_score(result)
        row = {"Period": period, "From Date": from_date, "To Date": to_date, "MAPE": mape, "SMAPE": smape,
               "MAE": mae, "RMSE": rmse, "COV": cov, "CE": ce, "IS": int_score}
        performance_df = performance_df.append(row, ignore_index=True)
    path = dir_path + "/performance.csv"
    performance_df.to_csv(path, index=False, sep=",", float_format='%.3f')


def analyze_performance(result_path, model):
    results = pd.read_csv(result_path + "/performance.csv")
    avg_mape = round(results["MAPE"].mean(), 2)
    std_mape = round(results["MAPE"].std(), 2)
    avg_smape = round(results["SMAPE"].mean(), 2)
    std_smape = round(results["SMAPE"].std(), 2)
    avg_mae = round(results["MAE"].mean(), 2)
    std_mae = round(results["MAE"].std(), 2)
    avg_rmse = round(results["RMSE"].mean(), 2)
    std_rmse = round(results["RMSE"].std(), 2)
    avg_cov = round(results["COV"].mean(), 2)
    std_cov = round(results["COV"].std(), 2)
    avg_ce = round(results["CE"].mean(), 2)
    std_ce = round(results["CE"].std(), 2)
    avg_is = round(results["IS"].mean(), 2)
    std_is = round(results["IS"].std(), 2)

    summary = open(result_path + "/performance.txt", "w")
    summary.write("-- Performance Summary for '{}', created {} --\n\n".format(model.get_name(),
                                                                              model.get_time().replace("_", " ")))
    line = "Point performance:\nMape:\t {} ({})\nSmape:\t{} ({})\nMae:\t{} ({})\nRmse:\t{} ({})\n\n".format(
        avg_mape, std_mape, avg_smape, std_smape, avg_mae, std_mae, avg_rmse, std_rmse)
    summary.write(line)
    summary.write("Interval performance: \nCov:\t{:.1f}% ({})\nACE:\t{:.1f}% ({})\nMIS:\t{:.1f} ({})".format(
        avg_cov, std_cov, avg_ce, std_ce, avg_is, std_is))

    summary.close()


def calculate_coverage_error(result):
    result['Hit'] = result.apply(lambda row: 1 if row["Lower"] <= row["System Price"] <=
                                                  row["Upper"] else 0, axis=1)
    coverage = sum(result["Hit"]) / len(result)
    cov_error = abs(0.95 - coverage)
    return coverage * 100, cov_error * 100


def calculate_interval_score(result):
    t_values = []
    for index, row in result.iterrows():
        u = row["Upper"]
        l = row["Lower"]
        y = row["System Price"]
        func_l = 1 if y < l else 0
        func_u = 1 if y > u else 0
        t = (u - l) + (2 / 0.05) * (l - y) * func_l + (2 / 0.05) * (y - u) * func_u
        t_values.append(t)
    interval_score = sum(t_values) / len(t_values)
    return interval_score


def calculate_mape(result):
    apes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        ape = (abs(true - forecast) / true) * 100
        apes.append(ape)
    mape = sum(apes) / len(apes)
    return mape


def calculate_smape(result):
    sapes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        sape = 100 * (abs(true - forecast) / (true + forecast / 2))
        sapes.append(sape)
    smape = sum(sapes) / len(sapes)
    return smape


def calculate_mae(result):
    aes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        ae = abs(true - forecast)
        aes.append(ae)
    mae = sum(aes) / len(aes)
    return mae


def calculate_rmse(result):
    ses = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        se = (true - forecast) ** 2
        ses.append(se)
    rmse = math.sqrt(sum(ses) / len(ses))
    return rmse


if __name__ == '__main__':
    # model_ = copy_last_day.CopyLastDayModel()
    # model_ = sarima.Sarima()
    model_ = ets.Ets()
    periods_ = get_four_periods_median_method(write_summary=False)
    # periods_ = get_all_2019_periods()
    #periods_ = get_one_period()
    validate(model_, periods_, plot=True)

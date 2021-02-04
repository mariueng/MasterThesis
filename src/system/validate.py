# script for validation test
from generate_periods import get_four_periods_median_method
from src.models.benchmarks import copy_last_day
from data.data_handler import get_data
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import datetime as dt
import math


def validate(model, periods, plot):  # plot is boolean value, if you want to plot results
    print("-- Running validation using Model '{}' --".format(model.get_name()))
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
    calculate_point_performance(result_list, result_path, model)
    calculate_interval_performance(result_list, result_path, model)
    print("\nResults are saved to 'src/" + result_path[3:] + "'")


def get_forecast(model, period):
    print("Forecasting from {} to {}".format(period[0], period[1]))
    time_df = get_data(period[0], period[1], [], os.getcwd())
    time_df["Forecast"] = np.nan
    point_forecast = model.get_point_forecast(time_df)
    point_forecast["Upper"] = np.nan
    point_forecast["Lower"] = np.nan
    prob_forecast = model.get_prob_forecast(point_forecast)
    return prob_forecast


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
    # plt.plot(result["DateTime"], result["Upper"], "--", label="Interval", color=fc_color)
    # plt.plot(result["DateTime"], result["Lower"], "--", color=fc_color)
    plt.gca().fill_between(result["DateTime"], result["Upper"], result["Lower"],
                           facecolor='gainsboro', interpolate=True, label="Interval")
    for line in plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    start_day_string = dt.datetime.strftime(period[0], "%d %b")
    end_day_string = dt.datetime.strftime(period[1], "%d %b")
    plt.title("Result from '{}' - {} to {}".format(model_name, start_day_string, end_day_string), pad=title_pad)
    plt.tight_layout()
    plot_path = period_no + "_" + start_day_string.replace(" ", "") + "_" + end_day_string.replace(" ", "") + ".png"
    out_path = dir_path + "/" + plot_path
    plt.savefig(out_path)
    plt.close()


def calculate_point_performance(result_list, dir_path, model):
    print("Calculating point performance for all periods")
    mapes = []
    smapes = []
    maes = []
    rmses = []
    for result in result_list:
        mapes.append(calculate_mape(result))
        smapes.append(calculate_smape(result))
        maes.append(calculate_mae(result))
        rmses.append(calculate_rmse(result))
    avg_mape = round(sum(mapes) / len(mapes), 2)
    avg_smape = round(sum(smapes) / len(smapes), 2)
    avg_mae = round(sum(maes) / len(maes), 2)
    avg_rmse = round(sum(rmses) / len(rmses), 2)
    summary = open(dir_path + "/performance.txt", "w")
    summary.write("-- Performance Summary for '{}', created {} --\n\n".format(model.get_name(),
                                                                              model.get_time().replace("_", " ")))
    line = "Point performance:\nMape:\t {}\nSmape:\t{}\nMae:\t{}\nRmse:\t{}\n\n".format(avg_mape, avg_smape, avg_mae,
                                                                                        avg_rmse)
    summary.write(line)
    lines = ["Mapes: " + ", ".join(format(x, ".2f") for x in mapes) + "\n",
             "Smapes: " + ", ".join(format(x, ".2f") for x in smapes) + "\n",
             "Maes: " + ", ".join(format(x, ".2f") for x in maes) + "\n",
             "Rmses: " + ", ".join(format(x, ".2f") for x in rmses) + "\n"]
    for line in lines:
        summary.write(line)
    summary.close()


def calculate_interval_performance(result_list, dir_path, model):
    print("Calculating point performance for all periods")
    coverages = []
    cov_errors = []
    for result in result_list:
        result['Hit'] = result.apply(lambda row: 1 if row["Lower"] <= row["System Price"] <=
                                                      row["Upper"] else 0, axis=1)
        coverage = sum(result["Hit"]) / len(result)
        cov_error = abs(coverage - 0.95)
        coverages.append(coverage)
        cov_errors.append(cov_error)
    a_coverage = sum(coverages) / len(coverages)
    ace = abs(a_coverage - 0.95)
    summary = open(dir_path + "/performance.txt", "a")
    summary.write("\nInterval performance: \nCov:\t{:.1f}%\nACE:\t{:.1f}%".format(a_coverage * 100, ace * 100))
    int_scores = get_interval_scores(result_list)
    mean_int_score = sum(int_scores) / len(int_scores)
    summary.write("\nMIS:\t{:.2f}\n".format(mean_int_score))
    summary.write("\nCoverages:\t{}".format(get_lst_as_str(coverages, 3)))
    summary.write("\nI. scores:\t{}".format(get_lst_as_str(int_scores, 2)))
    summary.close()


def get_lst_as_str(input_list, decimals):
    string = str([round(a, decimals) for a in input_list])[1:-1]
    return string


def get_interval_scores(result_list):
    interval_scores = []
    for result in result_list:
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
        interval_scores.append(interval_score)
    return interval_scores


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
    model_ = copy_last_day.CopyLastDayModel()
    periods_ = get_four_periods_median_method(write_summary=False)
    validate(model_, periods_, plot=True)

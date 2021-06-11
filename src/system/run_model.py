#  script for running model
from generate_periods import get_four_periods_median_method
from generate_periods import get_one_period
from generate_periods import get_random_periods
from generate_periods import get_all_2019_periods
from generate_periods import get_validation_periods
from generate_periods import get_testing_periods
from scores import calculate_interval_score
from scores import calculate_coverage_error
from scores import get_all_point_metrics
from src.models.naive_day import naive_day
from src.models.naive_week import naive_week
from src.models.sarima import sarima
from src.models.expert_model import expert_model
from src.models.expert_day import expert_day
from src.models.expert_mlp import expert_mlp
from src.models.curve_model import curve_model
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
import warnings
warnings.filterwarnings("ignore")


def run(model, periods, result_folder, plot):  # plot is boolean value, if you want to plot results
    print("-- Running periods using Model '{}' --".format(model.get_name()))
    start_time = time.time()
    result_list = []
    for i in range(len(periods)):
        forecast_df = get_forecast(model, periods[i])
        true_price_df = get_data(periods[i][0], periods[i][1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_list.append(result_df)
    result_path = get_result_folder(model, result_folder)
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
    time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
    if len(time_df) != 336:
        print("Length of horizon: {}".format(len(time_df)))
        print("Prediction horizon must be length 336")
        assert False
    time_df["Forecast"] = np.nan
    time_df["Upper"] = np.nan
    time_df["Lower"] = np.nan
    forecast_df = model.forecast(time_df)
    return forecast_df


def get_result_folder(model, result_folder):
    name = model.get_name()
    # time = model.get_time() REMOVE "#" if you want to keep old folder with same model name
    # folder_name = (name+"_"+time).replace(" ", "")
    folder_name = name.replace(" ", "")
    folder_path = "../results/{}/".format(result_folder) + folder_name
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # delete old
    os.makedirs(folder_path)  # create new
    os.makedirs(folder_path+"/plots")  # create plot folder
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
    ymax = max(max(result["System Price"]), max(result["Forecast"]))*1.1
    ymin = min(min(result["System Price"]), min(result["Forecast"]))*0.95
    plt.ylim(ymin, ymax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    start_day_string = dt.datetime.strftime(period[0], "%d %b")
    end_day_string = dt.datetime.strftime(period[1], "%d %b")
    plt.title("Result from '{}' - {} to {}".format(model_name, start_day_string, end_day_string), pad=title_pad)
    plt.tight_layout()
    plot_path = period_no + "_" + start_day_string.replace(" ", "") + "_" + end_day_string.replace(" ", "") + ".png"
    out_path = dir_path + "/plots/" + plot_path
    plt.savefig(out_path)
    plt.close()


def save_forecast(result_list, result_path):
    for i in range(len(result_list)):
        result = result_list[i]
        result["Period"] = i+1
    all_forecasts = pd.concat(result_list, ignore_index=True)
    all_forecasts = all_forecasts[["Period", "Date", "Hour", "System Price", "Forecast", "Upper", "Lower"]]
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
        point_metrics = get_all_point_metrics(result)
        mape = point_metrics["mape"]
        smape = point_metrics["smape"]
        mae = point_metrics["mae"]
        rmse = point_metrics["rmse"]
        cov, ce = calculate_coverage_error(result)
        int_score = calculate_interval_score(result)
        row = {"Period": period, "From Date": from_date, "To Date": to_date, "MAPE": mape, "SMAPE": smape,
               "MAE": mae, "RMSE": rmse, "COV": cov, "CE": ce, "IS": int_score}
        performance_df = performance_df.append(row, ignore_index=True)
    path = dir_path + "/performance.csv"
    performance_df.to_csv(path, index=False, sep=",", float_format='%.3f')


def analyze_performance(result_path, model):
    results = pd.read_csv(result_path + "/performance.csv")
    avg_mape = round(results["MAPE"].mean(), 3)
    std_mape = round(results["MAPE"].std(), 3)
    avg_smape = round(results["SMAPE"].mean(), 3)
    std_smape = round(results["SMAPE"].std(), 3)
    avg_mae = round(results["MAE"].mean(), 3)
    std_mae = round(results["MAE"].std(), 3)
    avg_rmse = round(results["RMSE"].mean(), 3)
    std_rmse = round(results["RMSE"].std(), 3)
    avg_cov = round(results["COV"].mean(), 3)
    std_cov = round(results["COV"].std(), 3)
    avg_ce = round(results["CE"].mean(), 3)
    std_ce = round(results["CE"].std(), 3)
    avg_is = round(results["IS"].mean(), 3)
    std_is = round(results["IS"].std(), 3)

    summary = open(result_path + "/performance.txt", "w")
    summary.write("-- Performance Summary for '{}', created {} --\n\n".format(model.get_name(),
                                                                              model.get_time().replace("_", " ")))
    line = "Point performance:\nMape:\t {} ({})\nSmape:\t{} ({})\nMae:\t{} ({})\nRmse:\t{} ({})\n\n".format(
        avg_mape, std_mape, avg_smape, std_smape, avg_mae, std_mae, avg_rmse, std_rmse)
    summary.write(line)
    summary.write("Interval performance: \nCov:\t{:.2f}% ({})\nACE:\t{:.2f}% ({})\nMIS:\t{:.2f} ({})".format(
        avg_cov, std_cov, avg_ce, std_ce, avg_is, std_is))
    if len(results) > 5:
        results = results.set_index("Period")
        three_best_smape = results.sort_values(by="SMAPE").head(3)[["SMAPE"]].to_dict()["SMAPE"]
        summary.write("\n\nThree best SMAPE performances:\t{}".format(three_best_smape))
        three_worst_smape = results.sort_values(by="SMAPE").tail(3)[["SMAPE"]].to_dict()["SMAPE"]
        summary.write("\nThree worst SMAPE performances:\t{}".format(three_worst_smape))
        three_best_mis = results.sort_values(by="IS").head(3)[["IS"]].to_dict()["IS"]
        summary.write("\nThree best IS performances:\t{}".format(three_best_mis))
        three_worst_mis = results.sort_values(by="IS").tail(3)[["IS"]].to_dict()["IS"]
        summary.write("\nThree worst IS performances:\t{}".format(three_worst_mis))
    summary.close()


def get_periods_and_result_folder_path(mode):
    if mode == "v":
        periods = get_validation_periods()
        result_folder = "validation"
    elif mode == "t":
        periods = get_testing_periods()
        result_folder = "test"
    elif mode == "short":
        periods = [(dt.datetime(2019, 4, 11).date(), dt.datetime(2019, 4, 24).date())]
        periods = get_random_periods(1)
        result_folder = "short"
    elif mode == "old":
        # periods = get_all_2019_periods()
        periods = get_random_periods(1)
        #periods = [(dt.datetime(2019, 3, 22).date(), dt.datetime(2019, 4, 4).date())]
        result_folder = "old"
    else:
        assert False
    return periods, result_folder


if __name__ == '__main__':
    # model_ = naive_day.NaiveDay()
    # model_ = naive_week.NaiveWeek()
    # model_ = sarima.Sarima()
    # model_ = ets.Ets()
    # model_ = expert_model.ExpertModel()
    # model_ = expert_day.ExpertDay()
    # model_ = expert_mlp.ExpertMLP()
    model_ = curve_model.CurveModel()
    # periods_ = get_four_periods_median_method(write_summary=False)
    # periods_ = get_random_periods(30)
    # periods_, result_folder_ = get_periods_and_result_folder_path("short")
    # periods_, result_folder_ = get_periods_and_result_folder_path("old")
    # periods_, result_folder_ = get_periods_and_result_folder_path("v")
    periods_, result_folder_ = get_periods_and_result_folder_path("short")
    # periods_ = get_testing_periods()
    # periods_ = get_one_period()
    # periods_ = [(dt.datetime(2019, 5, 27), dt.datetime(2019, 6, 9))]
    run(model_, periods_, result_folder_, plot=True)

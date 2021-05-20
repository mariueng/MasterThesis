import pandas as pd
import calendar
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from data.data_handler import get_data

from pyparsing import col

label_pad = 12
title_pad = 20
main_col = "steelblue"
sec_col = "firebrick"
third_col = "darkorange"
fourth_color = "mediumseagreen"
fig_size = (13, 7)


def plot_monthly_error(folder_path):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    result = pd.DataFrame(columns=["Month", "E", "AE", "APE", "Pos error", "Neg error"])
    df = pd.read_csv(folder_path + "/forecast.csv")
    if "DateTime" not in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Hour"] = pd.to_datetime(df['Hour'].astype(str).str[0:2].astype(int), format="%H").dt.time
        df["DateTime"] = df.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    else:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    for month in months:
        month_int = months.index(month) + 1
        sub_df = df[df["Date"].dt.month == month_int].copy()
        pos_mask = (sub_df['Forecast'] > sub_df["System Price"])
        pos_error = pos_mask.value_counts()[True] / len(sub_df)
        sub_df["E"] = sub_df["Forecast"] - sub_df["System Price"]
        sub_df["AE"] = abs(sub_df["Forecast"] - sub_df["System Price"])
        sub_df["APE"] = 100 * (sub_df["AE"] / sub_df["System Price"])
        e = sub_df["E"].mean()
        ae = sub_df["AE"].mean()
        ape = sub_df["APE"].mean()
        row = {"Month": month, "E": e, "AE": ae, "APE": ape, "Pos error": pos_error, "Neg error": 1 - pos_error}
        result = result.append(row, ignore_index=True)
    n = 12
    width = 0.35  # the width of the bars
    columns = ["E", "AE"]
    legends = ["Mean error", "Mean abs. error"]
    labels = ["Month", 'Error [€]']
    colors = [main_col, sec_col]
    title = "Monthly Errors"
    bar_plot(result, n, width, columns, legends, labels, colors, title, months, folder_path)
    width = 0.5  # the width of the bars
    columns = ["Pos error", "Neg error"]
    legends = ["Forecast > SYS", "Forecast <= SYS"]
    labels = ["Month", "Proportion"]
    colors = [main_col, sec_col]
    title = "Monthly Positive/Negative Error Frequency Ratio"
    ratio_plot(result, width, columns, legends, labels, colors, title, months, folder_path)


def bar_plot(result, n, width, columns, legends, labels, colors, title, x_ticks, folder_path):
    ind = np.arange(n)  # the x locations for the groups
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for i in range(len(columns)):
        ax.bar(ind + i * width, result[columns[i]].tolist(), width, color=colors[i], label=legends[i])
    ax.set_xlabel(labels[0], labelpad=label_pad)
    ax.set_ylabel(labels[1], labelpad=label_pad)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_ticks)
    model = folder_path.split("/")[-1]
    plt.title(title + " for {}".format(model), pad=title_pad)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    dir_path = '../analysis/{}'.format(model)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(dir_path + "/monthly_errors.png")
    print("Saved to {}/monthly_errors.png".format(dir_path))


def ratio_plot(result, width, columns, legends, labels, colors, title, months, folder_path):
    result[columns].plot(kind="bar", stacked=True, color=colors, figsize=fig_size, width=width)
    for line in plt.legend(labels=legends, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylim(0, 1.1)
    model = folder_path.split("/")[-1]
    dir_path = '../analysis/{}'.format(model)
    plt.xticks(range(len(months)), months)
    plt.xlabel(labels[0], labelpad=label_pad)
    plt.ylabel(labels[1], labelpad=label_pad)
    plt.title(title + " for {}".format(model), pad=title_pad)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.tight_layout()
    plt.savefig(dir_path + "/monthly_error_ratio.png")
    print("Saved to {}/monthly_error_ratio.png".format(dir_path))


def compare_results_two_models(path_names):
    m_1_name = path_names[0]
    m_2_name = path_names[1]
    p_1 = pd.read_csv("../results/validation/" + path_names[0] + "/performance.csv")
    p_2 = pd.read_csv("../results/validation/" + path_names[1] + "/performance.csv")
    for c in ["MAPE", "SMAPE", "MAE", "RMSE"]:
        m_1 = p_1[c]
        m_2 = p_2[c]
        m_3 = pd.DataFrame(data={"M1": m_1.values, "M2": m_2.values})
        best = m_3.min(axis=1).mean()
        m_1_best = 0
        for i in range(len(m_1)):
            m_1_best += 1 if m_1[i] < m_2[i] else 0
        m_1_best = 100 * m_1_best / len(m_1)
        winner = m_1_name if m_1_best > 50 else m_2_name
        print("{}:\t{} is best in {:.2f}% periods".format(c, winner, max(m_1_best, 100 - m_1_best)))
        print("\t\t{}: {:.2f}, {}: {:.2f}. Best: {:.2f}".format(m_1_name, m_1.mean(), m_2_name, m_2.mean(), best))


def plot_all_monthly_errors_all_models():
    all_folders = ["CurveModel", "ETS", "ExpertDay", "ExpertMLP", "ExpertModel", "NaiveDay", "NaiveWeek", "Sarima"]
    for f in all_folders:
        path = "../results/test/{}".format(f)
        plot_monthly_error(path)


def find_best_and_worst_periods_curve_model():
    folder_path = "../results/test/CurveModel"
    perf_df = pd.read_csv(folder_path + "/performance.csv")
    df = perf_df.sort_values(by="MAE")
    best = df[["Period", "From Date", "To Date", "MAE"]].head(3)
    print(best)
    worst = df[["Period", "From Date", "To Date", "MAE"]].tail(3)
    print(worst)
    int_df = perf_df[["CE", "IS"]]
    norm_int = (int_df - int_df.min()) / (int_df.max() - int_df.min())
    norm_int["Sum"] = norm_int.sum(axis=1)
    norm_int = norm_int.sort_values(by="Sum")
    print(int_df.loc[norm_int.head(3).index])
    print(int_df.loc[norm_int.tail(3).index])


def demand_and_supply_errors_curve_model():
    folder_path = "../models/curve_model/demand_scores_test_period"
    for res in ["day", "hour"]:
        df = pd.read_csv(folder_path + "/{}_demand_results.csv".format(res))
        mae = df["AE"].mean()
        mape = df["APE"].mean()
        print("{}: MAE demand forecast {:.2f}, MAPE demand forecast {:.2f}".format(res, mae, mape))


def t_test():
    cm_perf = pd.read_csv("../results/test/CurveModel/performance.csv")
    sarima_perf = pd.read_csv("../results/test/Sarima/performance.csv")
    for metric in ["MAPE", "SMAPE", "MAE", "RMSE"]:
        cm, naive = cm_perf[metric], sarima_perf[metric]
        t2, p2 = stats.ttest_ind(cm, naive)
        print("{}: t = {:.2f} with a p vale {:.7f}".format(metric, t2, p2))
    assert False
    cm, naive = cm_perf["IS"], naive_perf["IS"]
    t2, p2 = stats.ttest_ind(naive, cm)
    print("MIS: t = {:.2f} with a p vale {:.7f}".format(t2, p2))


def monthly_error_double():
    already_calc = True
    if already_calc:
        result = pd.read_csv("../analysis/CurveModel/season_res.csv")
    else:
        result = get_season_result("CurveModel")
    naive_result = get_season_result("NaiveDay")
    print(naive_result)
    print("Curve Model")
    print(result)
    print("Mean error: {:.2f}".format(result["Error"].mean()))
    print("Mean pos freq: {:.2f}".format(result["Pos"].mean()))
    for i in range(1, 5):
        print("Season {}: diff {:.3f}".format(i, result.loc[i-1, "MAE"]-naive_result.loc[i-1, "MAE"]))
    assert False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax3 = ax1.twinx()
    width = 0.5
    space = 2
    for i in range(len(result)):
        val = result.loc[i, "Error"]
        ln1 = ax1.bar(space * i, val, width=width, color=third_col)
        ln2 = ax1.bar(space * i + width, result.loc[i, "MAE"], width=width, color=main_col)
        ln3 = ax3.bar(space * i + width * 2, result.loc[i, "MAPE"], width=width, color=sec_col)
        if i == 0:
            lines = [ln1, ln2, ln3]
            labs = ["Mean error", "MAE", "MAPE"]
    ax1.set_ylim(result["Error"].min() * 1.2, result["MAE"].max() * 1.1)
    ax3.set_ylim(-6.9, 45)
    ax3.set_ylabel("Error [%]", labelpad=label_pad, color=sec_col)
    ax3.tick_params(colors=sec_col)
    ax1.set_xlabel("Season", labelpad=label_pad)
    ax1.set_ylabel("Error [€]", labelpad=label_pad)
    ax1.set_xticks([i * space + width for i in range(4)])
    ax1.set_xticklabels(["Winter", "Spring", "Summer", "Fall"])
    ax3.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
               fancybox=True, shadow=True, handles=lines, labels=labs)

    result = result.rename(columns={"Pos": "Forecast > SYS", "Neg": "Forecast ≤ SYS"})
    result[["Forecast > SYS", "Forecast ≤ SYS"]].plot(kind="bar", stacked=True, color=[main_col, sec_col],
                                                       figsize=fig_size, width=width, ax=ax2)
    ax2.set_ylim(-0.16, 1.1)
    for line in ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    ax2.set_xlabel("Season", labelpad=label_pad)
    ax2.set_ylabel("Proportion", labelpad=label_pad)
    ax2.set_xticks([i for i in range(4)])
    ax2.set_xticklabels(["Winter", "Spring", "Summer", "Fall"], rotation=0)
    ax2.set_title("Curve Model Positive/Negative Error Frequency per Season", pad=title_pad)
    ax3.set_title("Curve Model Point Error Metrics per Season", pad=title_pad)
    plt.tight_layout(pad=3.0)
    plt.savefig("../analysis/CurveModel/season_res.png")
    plt.show()


def get_season_result(model_name):  # Helping method
    df = pd.read_csv("../results/test/{}/forecast.csv".format(model_name))
    df["Date"] = pd.to_datetime(df["Date"])
    if "DateTime" not in df.columns:
        df["Hour"] = pd.to_datetime(df['Hour'].astype(str).str[0:2].astype(int), format="%H").dt.time
        df["DateTime"] = df.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    data = get_data("03.06.2019", "31.05.2020", ["Season"], os.getcwd(), "d")
    df = df.merge(data, on="Date")
    result = pd.DataFrame(columns=["Season", "Error", "MAE", "MAPE", "Pos"])
    for i in range(1, 5):
        sub_df = df[df["Season"] == i][["Forecast", "System Price", "Season"]]
        sub_df["Error"] = sub_df["Forecast"] - sub_df["System Price"]
        sub_df["Pos"] = sub_df.apply(lambda r: 1 if r["Error"] > 0 else 0, axis=1)
        sub_df["AE"] = abs(sub_df["Error"])
        sub_df["APE"] = 100 * sub_df["AE"] / sub_df["System Price"]
        row = {"Season": sub_df["Season"].mean(), "Error": sub_df["Error"].mean(), "MAE": sub_df["AE"].mean(),
               "MAPE": sub_df["APE"].mean(), "Pos": sub_df["Pos"].sum() / len(sub_df)}
        result = result.append(row, ignore_index=True)
    result["Neg"] = 1 - result["Pos"]
    result = result.round(2)
    result.to_csv("../analysis/{}/season_res.csv".format(model_name), index=False)
    return result


def error_distributions():
    df = pd.read_csv("../results/test/CurveModel/performance.csv")
    scores = ["MAPE", "SMAPE", "MAE", "RMSE"]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 7))
    axes = [ax1, ax2, ax3, ax4]
    colors = [main_col, sec_col, third_col, fourth_color]
    for i in range(4):
        avg = df[scores[i]].mean()
        counts, bins, bars = axes[i].hist(df[scores[i]], bins=35, density=1, color=colors[i], label="{} distribution ({:.2f})".format(scores[i], avg))
        axes[i].set_xlabel(scores[i])
    for ax in axes:
        for line in ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.03),
                               fancybox=True, shadow=True).get_lines():
            line.set_linewidth(2)
    ax1.set_ylabel("Proportion", labelpad=label_pad)
    fig.suptitle("Point Metric Error Distribution - Curve Model", y=0.98, size=15)
    plt.tight_layout(pad=2)
    plt.savefig("../analysis/CurveModel/error_dist.png")


def prob_dist():
    df = pd.read_csv("../results/test/CurveModel/performance.csv")
    scores = ["COV", "CE", "IS"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    axes = [ax1, ax2, ax3]
    colors = [main_col, sec_col, third_col]
    for i in range(3):
        print(scores[i])
        avg = df[scores[i]].mean()
        (counts, bins, patches) = axes[i].hist(df[scores[i]].values, bins=77, color=colors[i], align="mid",
                                               label="{} distribution ({:.2f})".format(scores[i], avg))
        #count_sum = counts.sum()
        #counts = np.append(counts, 1 - count_sum)
        axes[i].set_xlabel(scores[i])
        for j in range(len(counts)):
            print("Between {:.2f} and {:.2f}: {:.3f}".format(bins[j], bins[j+1], counts[j]))
        if i == 0:
            axes[i].set_xlim(df[scores[i]].mean()-1.96*df[scores[i]].std(), 102)
        else:
            axes[i].set_xlim(df[scores[i]].min()-2, df[scores[i]].mean()+1.96*df[scores[i]].std())
    for ax in axes:
        for line in ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.03),
                               fancybox=True, shadow=True).get_lines():
            line.set_linewidth(2)
    ax1.set_ylabel("Proportion", labelpad=label_pad)
    fig.suptitle("Probabilistic Metric Error Distribution - Curve Model", y=0.97, size=14)
    plt.tight_layout(pad=2)
    plt.savefig("../analysis/CurveModel/prob_error_dist.png")

if __name__ == '__main__':
    # plot_all_monthly_errors_all_models()
    # path_names_ = ["CurveModel_mean_supply", "CurveModel_351_1"]
    # compare_results_two_models(path_names_)
    # find_best_and_worst_periods_curve_model()
    # demand_and_supply_errors_curve_model()
    # t_test()
    # monthly_error_double()
    # error_distributions()
    prob_dist()

import pandas as pd
import calendar
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os

from pyparsing import col

label_pad = 12
title_pad = 20
main_col = "steelblue"
sec_col = "firebrick"
fig_size = (13, 7)


def plot_monthly_error(folder_path):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    result = pd.DataFrame(columns=["Month", "E", "AE", "APE", "Pos error", "Neg error"])
    df = pd.read_csv(folder_path+"/forecast.csv")
    if "DateTime" not in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Hour"] = pd.to_datetime(df['Hour'], format="%H").dt.time
        df["DateTime"] = df.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    else:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    for month in months:
        month_int = months.index(month) + 1
        last_date = calendar.monthrange(2019, month_int)[1]
        start = dt(2019, month_int, 1)
        end = dt(2019, month_int, last_date, 23)
        mask = (df['DateTime'] >= start) & (df['DateTime'] <= end)
        sub_df = df.loc[mask].reset_index()
        pos_mask = (sub_df['Forecast'] > sub_df["System Price"])
        pos_error = pos_mask.value_counts()[True]/len(sub_df)
        sub_df["E"] = sub_df["Forecast"] - sub_df["System Price"]
        sub_df["AE"] = abs(sub_df["Forecast"] - sub_df["System Price"])
        sub_df["APE"] = 100 * (sub_df["AE"] / sub_df["System Price"])
        e = sub_df["E"].mean()
        ae = sub_df["AE"].mean()
        ape = sub_df["APE"].mean()
        row = {"Month": month, "E": e, "AE": ae, "APE": ape, "Pos error": pos_error, "Neg error": 1-pos_error}
        result = result.append(row, ignore_index=True)
    n = 12
    width = 0.35  # the width of the bars
    columns = ["E", "AE"]
    legends = ["Mean error", "Mean abs. error"]
    labels = ["Month", 'Error [€]']
    colors = [main_col, sec_col]
    title = "Monthly Validation Errors"
    bar_plot(result, n, width, columns, legends, labels, colors, title, months, folder_path)
    width = 0.5  # the width of the bars
    columns = ["Pos error", "Neg error"]
    legends = ["Forecast > SYS", "Forecast <= SYS"]
    labels = ["Month", "Proportion"]
    colors = [main_col, sec_col]
    title = "Monthly Positive/Negative Validation Error Ratio"
    ratio_plot(result, width, columns, legends, labels, colors, title, months, folder_path)


def bar_plot(result, n, width, columns, legends, labels, colors, title, x_ticks, folder_path):
    ind = np.arange(n)  # the x locations for the groups
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for i in range(len(columns)):
        ax.bar(ind + i*width, result[columns[i]].tolist(), width, color=colors[i], label=legends[i])
    ax.set_xlabel(labels[0], labelpad=label_pad)
    ax.set_ylabel(labels[1], labelpad=label_pad)
    ax.set_xticks(ind + width/2)
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
    plt.savefig(dir_path+"/monthly_errors.png")
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
    plt.savefig(dir_path+"/monthly_error_ratio.png")
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
        print("{}:\t{} is best in {:.2f}% periods".format(c, winner, max(m_1_best, 100-m_1_best)))
        print("\t\t{}: {:.2f}, {}: {:.2f}. Best: {:.2f}".format(m_1_name, m_1.mean(), m_2_name, m_2.mean(), best))


if __name__ == '__main__':
    path = "../results/validation/CurveModel_351_1"
    plot_monthly_error(path)
    # path_names_ = ["CurveModel_mean_supply", "CurveModel_351_1"]
    # compare_results_two_models(path_names_)

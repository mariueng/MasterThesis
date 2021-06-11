import pandas as pd
import calendar
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import os
from scipy import stats
from data.data_handler import get_data
from data.data_handler import get_auction_data
from src.models.curve_model.curve_model import CurveModel
from src.system.generate_periods import get_testing_periods
from dm_test import dm_test
import seaborn as sns
from src.models.curve_model.supply_curve import get_supply_curve
from shapely.geometry import LineString

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
    models = ["Curve Model", "ETS", "Expert Day", "Expert MLP", "Expert Model", "Naive Day", "Naive Week", "Sarima"]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))
    axs = [ax1, ax2, ax3]
    metrics = ["MAE", "RMSE", "MAPE"]
    custom_lines = [Rectangle((0,0),1,1,facecolor="#da3b46"), Rectangle((0,0),1,1,facecolor="#3f7f93")]
    for m in range(len(metrics)):
        metric = metrics[m]
        ax = axs[m]
        result = pd.DataFrame(columns=["Model"])
        result["Model"] = models
        for i in range(len(models)):
            result[models[i]] = np.ones(shape=len(models))
        for i in range(len(models)):
            model = models[i]
            performance = pd.read_csv("../results/test/{}/performance.csv".format(model.replace(" ", "")))[metric]
            for j in range(len(models)):
                if j != i:
                    other_model = models[j]
                    other_perf = pd.read_csv("../results/test/{}/performance.csv".format(other_model.replace(" ", "")))[metric]
                    t, p = stats.ttest_ind(performance, other_perf)
                    result.iloc[i, j+1] = p
        result = result.iloc[:, 1:]
        for i in range(len(result.columns)):
            for j in range(len(result.columns)):
                if j != i:
                    p = result.iloc[i, j]
                    if p < 0.05:
                        result.iloc[i, j] = 0.05
                    else:
                        result.iloc[i, j] = 1
        ticks = [col for col in result.columns]
        sns.heatmap(result.values, cmap=sns.diverging_palette(10, 220, as_cmap=False), cbar=False,
                    square=True, ax=ax, linewidths=1, linecolor="white")
        ax.set_xticks(np.arange(len(result.columns)) + 0.5)
        ax.set_xticklabels(ticks, rotation=45, size=9)
        ax.set_yticks(np.arange(len(result.columns)) + 0.5)
        ax.set_yticklabels([t.replace(" ", "\n") for t in ticks], rotation='horizontal', size=9)
        ax.legend(custom_lines, ["Diff. at p < 0.05", "No difference"], loc=(0.065, 1.02), ncol=2,
                 handlelength=1, handleheight=1, fancybox=True, shadow=True)
        ax.set_title("{}. Student t-test on {}".format(m+1, metrics[m]), y=1.12)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/stat_t_test.png")



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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
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
    ax3.set_ylabel("Error [%]", labelpad=label_pad, size=11)
    ax1.set_xlabel("Season", labelpad=label_pad, size=11)
    ax1.set_ylabel("Error [€]", labelpad=label_pad, size=11)
    ax1.set_xticks([i * space + width for i in range(4)])
    ax1.set_xticklabels(["Winter", "Spring", "Summer", "Fall"])
    ax3.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
               fancybox=True, shadow=True, handles=lines, labels=labs, prop={"size":11})

    result = result.rename(columns={"Pos": "Forecast > SYS", "Neg": "Forecast ≤ SYS"})
    result[["Forecast > SYS", "Forecast ≤ SYS"]].plot(kind="bar", stacked=True, color=[main_col, sec_col],
                                                       figsize=fig_size, width=width, ax=ax2)
    ax2.set_ylim(-0.16, 1.1)
    for line in ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True, prop={"size":11}).get_lines():
        line.set_linewidth(2)
    ax2.set_xlabel("Season", labelpad=label_pad, size=11)
    ax2.set_ylabel("Proportion", labelpad=label_pad, size=11)
    ax2.set_xticks([i for i in range(4)])
    ax2.set_xticklabels(["Winter", "Spring", "Summer", "Fall"], rotation=0)
    ax2.set_title("Positive/Negative Error Frequency per Season", pad=title_pad, size=14)
    ax3.set_title("Error Metrics per Season", pad=title_pad, size=14)
    plt.tight_layout(pad=1.5)
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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(13, 5.5))
    axes = [ax1, ax2, ax3, ax4]
    colors = [main_col, sec_col, third_col, fourth_color]
    for i in range(4):
        avg = df[scores[i]].mean()
        counts, bins, bars = axes[i].hist(df[scores[i]], bins=35, density=1, color=colors[i], label="Period {} ({:.2f})".format(scores[i][1:], avg))
        axes[i].set_xlabel(scores[i], size=11)
    for ax in axes:
        for line in ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.04),
                               fancybox=True, shadow=True, prop={"size":11}).get_lines():
            line.set_linewidth(2)
    ax1.set_ylabel("Proportion", labelpad=label_pad, size=11)
    fig.suptitle("Point Metric Error Distribution - Curve Model", y=0.98, size=14)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/error_dist.png")


def prob_dist():
    df = pd.read_csv("../results/test/CurveModel/performance.csv")
    scores = ["COV", "CE", "IS"]
    labs = {"COV": "Coverage", "CE": "Coverage error", "IS": "Interval score"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5.5))
    axes = [ax1, ax2, ax3]
    colors = [main_col, sec_col, third_col]
    for i in range(3):
        print(scores[i])
        avg = df[scores[i]].mean()
        (counts, bins, patches) = axes[i].hist(df[scores[i]].values, bins=77, color=colors[i], align="mid",
                                               label="Periods {} ({:.2f})".format(scores[i], avg))
        #count_sum = counts.sum()
        #counts = np.append(counts, 1 - count_sum)
        axes[i].set_xlabel(labs[scores[i]], size=11)
        for j in range(len(counts)):
            print("Between {:.2f} and {:.2f}: {:.3f}".format(bins[j], bins[j+1], counts[j]))
        if i == 0:
            axes[i].set_xlim(df[scores[i]].mean()-1.96*df[scores[i]].std(), 102)
        else:
            axes[i].set_xlim(df[scores[i]].min()-2, df[scores[i]].mean()+1.96*df[scores[i]].std())
    for ax in axes:
        for line in ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.04),
                               fancybox=True, shadow=True, prop={"size":11}).get_lines():
            line.set_linewidth(2)
    ax1.set_ylabel("Proportion", labelpad=label_pad, size=11)
    fig.suptitle("Probabilistic Metric Error Distribution - Curve Model", y=0.97, size=14)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/prob_error_dist.png")


def example_periods():
    df = pd.read_csv("../results/test/CurveModel/performance.csv")
    metrics = df.columns[3:].tolist()
    metrics.pop(metrics.index("COV"))
    find_periods = False
    if find_periods:
        best = {}
        worst = {}
        for m in metrics:
            sub_df = df[["Period", m]].sort_values(by=m)
            best[m] = sub_df.head(10)["Period"].to_list()
            worst[m] = sub_df.tail(10)["Period"].to_list()
        print("BEST")
        for key, values in best.items():
            print("{}: {}".format(key, values))
        print("WORST")
        for key, values in worst.items():
            print("{}: {}".format(key, values))
    final = {"MAPE": [77, 351], "SMAPE": [164, 336], "MAE": [307, 146], "RMSE": [313, 178], "CE": [220, 38], "IS": [312, 2]}
    all_periods = [77, 164, 307, 313, 220, 312, 351, 336, 146, 178, 38, 2]
    titles = ["Good {}".format(i) for i in metrics] + ["Bad {}".format(i) for i in metrics]
    data = pd.read_csv("../results/test/CurveModel/forecast.csv")
    data = data[data["Period"].isin(all_periods)].reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").dt.date
    data["Hour"] = pd.to_datetime(data['Hour'].astype(str).str[0:2].astype(int), format="%H").dt.time
    data["DateTime"] = data.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    fig, axs = plt.subplots(4, 3, figsize=(14, 17))
    positions = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        pos = positions[i]
        col = pos // 4
        row = pos % 4
        ax = axs[row][col]
        frame = "green" if i <6 else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame)
        sub_df = data[data["Period"] == all_periods[i]]
        if len(sub_df) != 336:
            print(all_periods[i])
            print(data)
            assert False
        from_date = dt.strftime(sub_df.head(1)["Date"].values[0], "%d %b")
        to_date = dt.strftime(sub_df.tail(1)["Date"].values[0], "%d %b")
        ax.plot(sub_df["DateTime"], sub_df["System Price"], color=main_col, label="T")
        ax.plot(sub_df["DateTime"], sub_df["Forecast"], color=sec_col, label="F")
        ax.set_xticks([])
        metric = titles[i].split()[1]
        score = df[df["Period"]==all_periods[i]][metric].values[0]
        ax.set_title("{}. {} ({:.2f}) {} - {}".format(i+1, titles[i], score, from_date, to_date))
        if metric in ["CE", "IS"]:
            ax.fill_between(sub_df["DateTime"], sub_df["Upper"], sub_df["Lower"],
                                   facecolor='gainsboro', interpolate=True, label="I")
        ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/12_periods.png")


def example_curves():
    d_col = plt.get_cmap("tab10")(0)
    s_col = plt.get_cmap("tab10")(1)
    all_periods = [77, 164, 307, 313, 220, 312, 351, 336, 146, 178, 38, 2]
    df = pd.read_csv("../results/test/CurveModel/performance.csv")
    df["From Date"] = pd.to_datetime(df["From Date"], format="%Y-%m-%d").dt.date
    metrics = df.columns[3:].tolist()
    metrics.pop(metrics.index("COV"))
    titles = ["Good {}".format(i) for i in metrics] + ["Bad {}".format(i) for i in metrics]
    dates = [df[df["Period"] == p].head(1)["From Date"].values[0] for p in all_periods]
    s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105,
                 165, 210]
    d_classes = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
    supply = pd.DataFrame(columns=["Period", "Date", "Hour"] + ["s {}".format(i) for i in s_classes])
    demand = pd.DataFrame(columns=["Period", "Date", "Hour", "Demand Forecast"])
    calculate_curves = False
    if calculate_curves:
        forecast = pd.read_csv("../results/test/CurveModel/forecast.csv")
        forecast = forecast[forecast["Period"].isin(all_periods)].reset_index(drop=True)
        forecast["Hour"] = forecast['Hour'].astype(str).str[0:2].astype(int)
        model = CurveModel()
        for i in range(len(dates)):
            d = dates[i]
            period = all_periods[i]
            print(period)
            f = get_data(d, d+timedelta(days=13), [], os.getcwd(), "h")
            for col in ["Forecast", "Upper", "Lower"]:
                f[col] = np.NAN
            _, demand_df, supply_df = model.forecast(f)
            for df in [demand_df, supply_df]:
                df["Period"] = period
            demand = demand.append(demand_df, ignore_index=True)
            supply = supply.append(supply_df, ignore_index=True)
        demand.to_csv("../analysis/CurveModel/example_periods/demand.csv", index=False)
        supply.to_csv("../analysis/CurveModel/example_periods/supply.csv", index=False)
        forecast.to_csv("../analysis/CurveModel/example_periods/forecast.csv", index=False)
    else:
        demand = pd.read_csv("../analysis/CurveModel/example_periods/demand.csv")
        supply = pd.read_csv("../analysis/CurveModel/example_periods/supply.csv")
        forecast = pd.read_csv("../analysis/CurveModel/example_periods/forecast.csv")
    fig, axs = plt.subplots(4, 3, figsize=(14, 17))
    positions = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        pos = positions[i]
        col = pos // 4
        row = pos % 4
        ax = axs[row][col]
        frame = "green" if i <6 else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame)
        period = all_periods[i]
        s_d = dates[i]
        s_d_title = dt.strftime(s_d, "%d %b")
        t_d_title = dt.strftime(s_d+timedelta(days=13), "%d %b")
        true_d = get_auction_data(s_d, s_d+timedelta(days=13), "d", os.getcwd()).iloc[:, 2:].mean(axis=0).values
        true_s = get_auction_data(s_d, s_d+timedelta(days=13), "s", os.getcwd()).iloc[:, 2:].mean(axis=0).values
        ax.plot(true_d, d_classes, color=d_col, linestyle="dotted")
        ax.plot(true_s, s_classes, color=s_col, linestyle="dotted")
        y_min, y_max = ax.get_ylim()
        f = forecast[forecast["Period"] == period]
        sys, for_c, u, l = (f["System Price"].mean(), f["Forecast"].mean(), f["Upper"].mean(), f["Lower"].mean())
        ax.set_ylim(0, max(70, max(sys, for_c)*1.1))
        d = demand[demand["Period"] == period]["Demand Forecast"].mean()
        ax.axvline(d, y_min, y_max, color=d_col)
        s = supply[supply["Period"] == period].iloc[:, 3:].mean(axis=0).values
        ax.set_xlim(min(true_s[3], s[3]), max(true_s[25], s[25]))
        ax.plot(s, s_classes, color=s_col)
        metric = titles[i].split()[1]
        score = df[df["Period"] == all_periods[i]][metric].values[0]
        ax.set_title("{}. {} ({:.2f}) {} - {}".format(i+1, titles[i], score, s_d_title, t_d_title))
        if metric in ["CE", "IS"]:
            if l < 0:
                ax.set_ylim(l-2, ax.get_ylim()[1])
            if u > ax.get_ylim()[1]:
                ax.set_ylim(0, u)
            ax.axhline(l, color="black")
            ax.axhline(u, color="black")
            custom_lines = [Line2D([0], [0], color="grey", linestyle="dotted"), Line2D([0], [0], color="grey"), Line2D([0], [0], color="black")]
            ax.legend(custom_lines, ["T ({:.1f})".format(sys), 'F ({:.1f})'.format(for_c), "Bound"], loc=0)
        else:
            custom_lines = [Line2D([0], [0], color="grey", linestyle="dotted"), Line2D([0], [0], color="grey")]
            ax.legend(custom_lines, ["T ({:.1f})".format(sys), 'F ({:.1f})'.format(for_c)], loc=0)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/12_curves.png")


def combined_period_curves():
    import matplotlib.dates as mdates
    performance = pd.read_csv("../results/test/CurveModel/performance.csv")
    performance["From Date"] = pd.to_datetime(performance["From Date"], format="%Y-%m-%d").dt.date
    metrics = performance.columns[3:].tolist()
    metrics.pop(metrics.index("COV"))
    final = {"MAPE": [77, 351], "SMAPE": [164, 336], "MAE": [307, 146], "RMSE": [313, 178], "CE": [220, 38], "IS": [312, 2]}
    all_periods = [77, 164, 307, 313, 220, 312, 351, 336, 146, 178, 38, 2]
    periods = [[77, 164], [307, 313], [220, 312], [351, 336], [146, 178], [38, 2]]
    titles = ["Good {}".format(i) for i in metrics] + ["Bad {}".format(i) for i in metrics]
    pred = pd.read_csv("../results/test/CurveModel/forecast.csv")
    pred = pred[pred["Period"].isin(all_periods)].reset_index(drop=True)
    pred["Date"] = pd.to_datetime(pred["Date"], format="%Y-%m-%d").dt.date
    pred["Hour"] = pd.to_datetime(pred['Hour'].astype(str).str[0:2].astype(int), format="%H").dt.time
    pred["DateTime"] = pred.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    demand = pd.read_csv("../analysis/CurveModel/example_periods/demand.csv")
    supply = pd.read_csv("../analysis/CurveModel/example_periods/supply.csv")
    forecast = pd.read_csv("../analysis/CurveModel/example_periods/forecast.csv")
    s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105,
                 165, 210]
    d_classes = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
    d_col, s_col = (plt.get_cmap("tab10")(0), plt.get_cmap("tab10")(1))
    y_tops = [70, 70, 16, 16, 65, 25, 40, 60, 200, 200, 60, 60]
    for j in range(len(periods)):
        fig, (ax_a, ax_b) = plt.subplots(2, 2, figsize=(13, 6))
        ax1, ax2 = (ax_a[0], ax_a[1])
        ax3, ax4 = (ax_b[0], ax_b[1])
        axs = [ax1, ax2, ax3, ax4]
        #frame = "green" if j <3 else "red"
        #for ax in axs:
            #for spine in ax.spines.values():
                #spine.set_edgecolor(frame)
        for i in range(2):
            period = periods[j][i]
            b_lim = y_tops[j*2+i]
            ax = ax_a[i]
            if i == 0:
                ax.set_ylabel("Price [€]", size=11)
            #ax.set_xlabel("Time", size=11)
            title = titles[j*2+i]
            metric = title.split()[1]
            score = performance[performance["Period"] == all_periods[j*2+i]][metric].values[0]
            sub_pred = pred[pred["Period"] == periods[j][i]]
            assert len(sub_pred) == 336
            from_date = dt.strftime(sub_pred.head(1)["Date"].values[0], "%d %b")
            to_date = dt.strftime(sub_pred.tail(1)["Date"].values[0], "%d %b")
            ax.plot(sub_pred["DateTime"], sub_pred["System Price"], color=main_col, label="T")
            ax.plot(sub_pred["DateTime"], sub_pred["Forecast"], color=sec_col, label="F")
            dates = sub_pred["Date"].unique()
            ax.set_xlim(dates[0], dates[len(dates)-1]+timedelta(days=1))
            ax.xaxis.set_ticks(dates)
            dtFmt = mdates.DateFormatter('%d')  # define the formatting
            ax.xaxis.set_major_formatter(dtFmt)  # apply the format to the desired axis
            #ax.set_xticklabels([])
            ax.set_title("{}) Forecast {} - {}. {} ({:.2f}) ".format((j*2+i) + 1, from_date, to_date, title, score), size=14)
            if metric in ["CE", "IS"]:
                ax.fill_between(sub_pred["DateTime"], sub_pred["Upper"], sub_pred["Lower"],
                                facecolor='gainsboro', interpolate=True, label="I")
            ax.legend(loc="upper left")
            ax = ax_b[i]
            ax.set_title("{}) Mean Curves".format(j*2+i+1), size=14)
            if i == 0:
                ax.set_ylabel("Price [€]", size=11)
            ax.set_xlabel("Volume [MWh]", size=11)
            s_d = sub_pred.head(1)["Date"].values[0]
            print(s_d)
            true_d = get_auction_data(s_d, s_d + timedelta(days=13), "d", os.getcwd()).iloc[:, 2:].mean(axis=0).values
            true_s = get_auction_data(s_d, s_d + timedelta(days=13), "s", os.getcwd()).iloc[:, 2:].mean(axis=0).values
            ax.plot(true_d, d_classes, color=d_col, linestyle="dotted")
            ax.plot(true_s, s_classes, color=s_col, linestyle="dotted")
            y_min, y_max = ax.get_ylim()
            f = forecast[forecast["Period"] == period]
            sys, for_c, u, lower = (f["System Price"].mean(), f["Forecast"].mean(), f["Upper"].mean(), f["Lower"].mean())
            max_sys, max_pred = (sub_pred["System Price"].max(), sub_pred["Forecast"].max())
            #ax.set_ylim(0, max(70, max(sys, for_c) * 1.1))
            s = supply[supply["Period"] == period].iloc[:, 3:].mean(axis=0).values
            ax.set_ylim(0, b_lim)
            if max(max_sys, max_pred) < 60:
                #ax.set_ylim(0, max(max_sys, max_pred)*1.5)
                ax.set_xlim(min(true_s[3], s[3]), max(20, max(true_s[25], s[25])))
            else:
                #ax.set_ylim(0, b_lim)
                ax.set_xlim(min(true_s[3], s[3]), max(true_s[27], s[27]))
            d = demand[demand["Period"] == period]["Demand Forecast"].mean()
            ax.axvline(d, y_min, y_max, color=d_col)
            ax.plot(s, s_classes, color=s_col)
            if metric in ["CE", "IS"]:
                if lower < 0:
                    ax.set_ylim(lower - 2, ax.get_ylim()[1])
                ax.set_ylim(ax.get_ylim()[0], b_lim)
                ax.axhline(lower, color="black")
                ax.axhline(u, color="black")
                custom_lines = [Line2D([0], [0], color="grey", linestyle="dotted"), Line2D([0], [0], color="grey"),
                                Line2D([0], [0], color="black")]
                ax.legend(custom_lines, ["T ({:.1f})".format(sys), 'F ({:.1f})'.format(for_c), "Bound"], loc="upper left")
            else:
                custom_lines = [Line2D([0], [0], color="grey", linestyle="dotted"), Line2D([0], [0], color="grey")]
                ax.legend(custom_lines, ["T ({:.1f})".format(sys), 'F ({:.1f})'.format(for_c)], loc="upper left")
        plt.tight_layout()
        #plt.show()
        #assert False
        plt.savefig("../analysis/CurveModel/result_{}.png".format(j+1))



def t_test_2():
    models = ["Curve Model", "ETS", "Expert Day", "Expert MLP", "Expert Model", "Naive Day", "Naive Week", "Sarima"]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 7))
    axs = [ax1, ax2, ax3]
    metrics = ["MAD", "MSE", "MAPE"]
    custom_lines = [Rectangle((0,0),1,1,facecolor="#da3b46"), Rectangle((0,0),1,1,facecolor="#3f7f93")]
    for m in range(len(metrics)):
        ax = axs[m]
        result = pd.DataFrame(columns=["Model"])
        result["Model"] = models
        for i in range(len(models)):
            result[models[i]] = np.ones(shape=len(models))
        for i in range(len(models)):
            model = models[i]
            forecast = pd.read_csv("../results/test/{}/forecast.csv".format(model.replace(" ", "")))
            for j in range(len(models)):

                if j != i:
                    other_model = models[j]
                    other_forecast = pd.read_csv("../results/test/{}/forecast.csv".format(other_model.replace(" ", "")))
                    dm = dm_test(forecast["System Price"], forecast["Forecast"], other_forecast["Forecast"], 336, metrics[m])
                    result.iloc[i, j+1] = dm[1]
        result = result.iloc[:, 1:]
        print(result)
        for i in range(len(result.columns)):
            for j in range(len(result.columns)):
                if j != i:
                    p = result.iloc[i, j]
                    if p < 0.05:
                        result.iloc[i, j] = 0.05
                    else:
                        result.iloc[i, j] = 1
        ticks = [col for col in result.columns]

        sns.heatmap(result.values, cmap=sns.diverging_palette(10, 220, as_cmap=False), cbar=False,
                    square=True, ax=ax, linewidths=1, linecolor="white")
        ax.set_xticks(np.arange(len(result.columns)) + 0.5)
        ax.set_xticklabels(ticks, rotation=45, size=9)
        ax.set_yticks(np.arange(len(result.columns)) + 0.5)
        ax.set_yticklabels([t.replace(" ", "\n") for t in ticks], rotation='horizontal', size=9)
        ax.set_title("{}. Diebold-Mariano on {}".format(m+1, metrics[m]), y=1.02)
        ax.legend(custom_lines, ["Diff. at p < 0.05", "No difference"], loc=(0.065, 1.02), ncol=2,
                 handlelength=1, handleheight=1, fancybox=True, shadow=True)
        ax.set_title("{}. Diebold-Mariano on {}".format(m+1, metrics[m]), y=1.12)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/dm.png")


def val_vs_test():
    models = ["Curve Model", "ETS", "Expert Day", "Expert MLP", "Expert Model", "Naive Day", "Naive Week", "Sarima"]
    metrics = ["MAPE", "SMAPE", "MAE", "RMSE"]
    result = pd.DataFrame(index=models)
    for m in metrics:
        result["Val {}".format(m)] = np.NAN
        result["Test {}".format(m)] = np.NAN
    for m in result.index:
        test = pd.read_csv("../results/test/{}/performance.csv".format(m.replace(" ", "")))
        val = pd.read_csv("../results/validation/{}/performance.csv".format(m.replace(" ", "")))
        for metric in metrics:
            result.loc[m, "Test {}".format(metric)] = test[metric].mean()
            result.loc[m, "Val {}".format(metric)] = val[metric].mean()
    print(result)
    for m in metrics:
        diff = (100 * result["Test {}".format(m)] / result["Val {}".format(m)]).mean()
        print("{} increase: {:.1f}%".format(m, diff))
    print("\n")
    for m in metrics:
        diff = (100 * result.head(1)["Test {}".format(m)] / result.head(1)["Val {}".format(m)]).mean()
        print("Curve Model {} increase: {:.1f}%".format(m, diff))


def check_data_test_val():
    training = get_data("01.07.2014", "02.06.2019", ["System Price"], os.getcwd(), "h")
    testing = get_data("03.06.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    print("Mean val price: {:.2f}".format(training["System Price"].mean()))
    print("Mean test price: {:.2f}".format(testing["System Price"].mean()))
    print("Stdev val price: {:.2f}".format(training["System Price"].std()))
    print("Stdev test price: {:.2f}".format(testing["System Price"].std()))
    for df in [training, testing]:
        for i in range(len(df)):
            price_this_hour = df.loc[i, "System Price"]
            df.loc[i, "Spike"] = check_if_spike(price_this_hour, i, df["System Price"])
        share_pos_spike_three = 100 * len(df[df["Spike"] == 1]) / len(df)
        share_neg_spike_three = 100 * len(df[df["Spike"] == -1]) / len(df)
        print("Spike occurrence:\tpos {:.2f}%, neg {:.2f}%".format(share_pos_spike_three, share_neg_spike_three))


def check_if_spike(price, i, df):  # Helping method
    spike = 0
    number_of_days_in_window = min(60, len(df) // 24)
    threshold = 1.96
    number_of_hours_in_window = 24 * number_of_days_in_window
    if i < number_of_hours_in_window:
        window_prices = df[0:number_of_hours_in_window]
    else:
        window_prices = df[i - number_of_hours_in_window:i]
    mean = window_prices.mean()
    std_dev = window_prices.std()
    if price > mean + threshold * std_dev:
        spike = 1
    elif price < mean - threshold * std_dev:
        spike = -1
    return spike


def assess_spike_detection():
    data = get_data("01.04.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    models = ["NaiveDay", "NaiveWeek", "Sarima", "ETS", "ExpertModel", "ExpertDay", "ExpertMLP", "CurveModel"]
    for m in models:
        forecast = pd.read_csv("../results/test/{}/forecast.csv".format(m))
        forecast["Date"] = pd.to_datetime(forecast["Date"], format="%Y-%m-%d")
        result = pd.DataFrame(columns=["Pos", "Neg"], index=forecast["Period"].unique())
        mod_pos_spikes = []
        test = True
        if test:
            for period in forecast["Period"].unique():
                sub_df = forecast[forecast["Period"] == period][["Date", "Hour", "Forecast"]].reset_index(drop=True)
                start_date = sub_df.loc[0, "Date"]
                window = data[(data["Date"] >= start_date - timedelta(days=60)) & (data["Date"] < start_date)]
                assert len(window) == 60 * 24
                mean = window["System Price"].mean()
                std_dev = window["System Price"].std()
                sub_df["Pos"] = sub_df.apply(lambda r: 1 if r["Forecast"] > mean+1.96*std_dev else 0, axis=1)
                sub_df["Neg"] = sub_df.apply(lambda r: 1 if r["Forecast"] < mean-1.96*std_dev else 0, axis=1)
                result.loc[period, "Pos"] = sub_df["Pos"].sum()/len(sub_df)
                result.loc[period, "Neg"] = sub_df["Neg"].sum()/len(sub_df)
                mod_pos_spikes += sub_df[sub_df["Pos"] == 1]["Forecast"].tolist()
            result["Spike"] = result["Pos"] + result["Neg"]
            print("{} total spike freq {:.2f}%".format(m, 100*result["Spike"].sum() / len(result)))
            #print("Pos spike freq {:.2f}%".format(100*result["Pos"].sum() / len(result)))
            #print("Neg spike freq {:.2f}%".format(100*result["Neg"].sum() / len(result)))
    assert False
    print("Mean price of model positive spikes {:.2f}".format(sum(mod_pos_spikes) / len(mod_pos_spikes)))
    print("\n")
    data["Pos"] = np.NAN
    data["Neg"] = np.NAN
    for i in range(len(data)):
        date = data.loc[i, "Date"]
        if date > dt(2019, 6, 2):
            window = data[(data["Date"] >= date - timedelta(days=60)) & (data["Date"] < date)]
            assert len(window) == 60 * 24
            mean = window["System Price"].mean()
            stdev = window["System Price"].std()
            data.loc[i, "Pos"] = 1 if data.loc[i, "System Price"] > mean + 1.96 * stdev else 0
            data.loc[i, "Neg"] = 1 if data.loc[i, "System Price"] < mean - 1.96 * stdev else 0
    data = data.dropna()
    data["Spike"] = data["Pos"] + data["Neg"]
    print("True spike freq {:.2f}%".format(100*data["Spike"].sum()/len(data)))
    print("True pos spike freq {:.2f}%".format(100*data["Pos"].sum()/len(data)))
    print("True neg spike freq {:.2f}%".format(100*data["Neg"].sum()/len(data)))
    pos_spike_df = data[data["Pos"] == 1]
    print("True mean price of model positive spikes {:.2f}".format(pos_spike_df["System Price"].mean()))


def performance_without_spikes():
    import math
    models = ["NaiveDay", "NaiveWeek", "Sarima", "ETS", "ExpertModel", "ExpertDay", "ExpertMLP", "CurveModel"]
    for m in models:
        print("{}: ".format(m))
        forecast = pd.read_csv("../results/test/{}/forecast.csv".format(m))
        forecast["Date"] = pd.to_datetime(forecast["Date"], format="%Y-%m-%d")
        data = get_data("01.04.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
        result = pd.DataFrame(columns=["MAE", "MAPE", "SMAPE", "RMSE"], index=forecast["Period"].unique())
        for period in forecast["Period"].unique():
            sub_df = forecast[forecast["Period"]==period].reset_index(drop=True)
            start_date = sub_df.loc[0, "Date"]
            window = data[(data["Date"] >= start_date - timedelta(days=60)) & (data["Date"] < start_date)]
            assert len(window) == 60 * 24
            mean = window["System Price"].mean()
            std_dev = window["System Price"].std()
            sub_df["Pos"] = sub_df.apply(lambda r: 1 if r["Forecast"] > mean + 1.96 * std_dev else 0, axis=1)
            sub_df["Neg"] = sub_df.apply(lambda r: 1 if r["Forecast"] < mean - 1.96 * std_dev else 0, axis=1)
            sub_df["Spike"] = sub_df["Pos"] + sub_df["Neg"]
            no_spike_df = sub_df[sub_df["Spike"] == 0]
            no_spike_df["Error"] = no_spike_df["System Price"] - no_spike_df["Forecast"]
            no_spike_df["AE"] = abs(no_spike_df["Error"])
            no_spike_df["APE"] = 100 * no_spike_df["AE"] / no_spike_df["System Price"]
            no_spike_df["SAPE"] = 100 * no_spike_df["AE"] / ((no_spike_df["System Price"] + no_spike_df["Forecast"])/2)
            no_spike_df["SE"] = no_spike_df["Error"]**2
            result.loc[period, "MAE"] = no_spike_df["AE"].mean()
            result.loc[period, "MAPE"] = no_spike_df["APE"].mean()
            result.loc[period, "SMAPE"] = no_spike_df["SAPE"].mean()
            div_factor = 1 if len(no_spike_df) == 0 else len(no_spike_df)
            result.loc[period, "RMSE"] = math.sqrt(no_spike_df["SE"].sum() / div_factor)
        print("MAPE if forecasted spikes are removed {:.2f}".format(result["MAPE"].mean()))
        print("SMAPE if forecasted spikes are removed {:.2f}".format(result["SMAPE"].mean()))
        print("MAE if forecasted spikes are removed {:.2f}".format(result["MAE"].mean()))
        print("RMSE if forecasted spikes are removed {:.2f}".format(result["RMSE"].mean()))


def trend_detection():
    forecast = pd.read_csv("../results/test/CurveModel/forecast.csv")
    forecast["Date"] = pd.to_datetime(forecast["Date"], format="%Y-%m-%d")
    data = get_data("01.05.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    result = pd.DataFrame(columns=["Trend", "Trend w1", "Trend w2", "Model", "Model w1", "Model w2"], index=forecast["Period"].unique())
    for period in forecast["Period"].unique():
        sub_df = forecast[forecast["Period"] == period].reset_index(drop=True)
        s_date = sub_df.loc[0, "Date"]
        true_df = data[(data["Date"] >= s_date - timedelta(days=7)) & (data["Date"] <= s_date+timedelta(days=13))].reset_index(drop=True)
        prev_lvl = true_df.head(168)["System Price"].median()
        result.loc[period, "Trend"] = true_df.tail(336)["System Price"].median() - prev_lvl
        result.loc[period, "Trend w1"] = true_df.loc[168:335, "System Price"].median() - prev_lvl
        result.loc[period, "Trend w2"] = true_df.tail(168)["System Price"].median() - prev_lvl
        result.loc[period, "Model"] = sub_df["Forecast"].median() - prev_lvl
        result.loc[period, "Model w1"] = sub_df.head(168)["Forecast"].median() - prev_lvl
        result.loc[period, "Model w2"] = sub_df.tail(168)["Forecast"].median() - prev_lvl
    pos_trend = result.sort_values(by="Trend").tail(50).reset_index(drop=True)
    neg_trend = result.sort_values(by="Trend").head(50).reset_index(drop=True)
    print("Number of times model predicts pos trend {} of 50".format(len(pos_trend[pos_trend["Model"]>0.5])))
    print("True mean trend increase {:.2f}, model trend increase {:.2f}".format(pos_trend["Trend"].mean(), pos_trend["Model"].mean()))
    print("Number of times model predicts neg trend {} of 50".format(len(pos_trend[neg_trend["Model"]<-0.5])))
    print("True mean trend increase {:.2f}, model trend increase {:.2f}".format(neg_trend["Trend"].mean(), neg_trend["Model"].mean()))
    print("\n")
    pos_trend = result.sort_values(by="Model").tail(50).reset_index(drop=True)
    neg_trend = result.sort_values(by="Model").head(50).reset_index(drop=True)
    print("Number of times trend is positive: {} of 50".format(len(pos_trend[pos_trend["Trend"]>0.5])))
    print("True mean trend increase {:.2f}, model trend increase {:.2f}".format(pos_trend["Trend"].mean(), pos_trend["Model"].mean()))
    print("Number of times trend is negative: {} of 50".format(len(pos_trend[neg_trend["Trend"]<-0.5])))
    print("True mean trend increase {:.2f}, model trend increase {:.2f}".format(neg_trend["Trend"].mean(), neg_trend["Model"].mean()))


def demand_performance():
    day = False
    if day:
        day_df = pd.read_csv("../models/curve_model/demand_scores_test_period/day_demand_results.csv")
        print("Day demand MAE {:.0f}, MAPE {:.2f}%\n".format(day_df["AE"].mean(), day_df["APE"].mean()))
        day_mae = pd.DataFrame(columns=["Day {}".format(i) for i in range(1, 14)], index=day_df["Period"].unique())
        day_mape = pd.DataFrame(columns=["Day {}".format(i) for i in range(1, 14)], index=day_mae.index)
        for p in day_df["Period"].unique():
            sub = day_df[day_df["Period"] == p].reset_index(drop=True)
            for i in range(len(sub)):
                day_mae.loc[p, "Day {}".format(i +1)] = sub.loc[i, "AE"]
                day_mape.loc[p, "Day {}".format(i +1)] = sub.loc[i, "APE"]
        for col in day_mae.columns:
            print("MAE {}: {:.2f}".format(col, day_mae[col].mean()))
        print("\n")
        for col in day_mape.columns:
            print("MAPE {}: {:.2f}".format(col, day_mape[col].mean()))
    do_hour = False
    if do_hour:
        hour_df = pd.read_csv("../models/curve_model/demand_scores_test_period/hour_demand_results.csv")
        print("Hour demand MAE {:.0f}, MAPE {:.2f}%\n".format(hour_df["AE"].mean(), hour_df["APE"].mean()))
        day_mae = pd.DataFrame(columns=["Day {}".format(i) for i in range(1, 14)], index=hour_df["Period"].unique())
        day_mape = pd.DataFrame(columns=["Day {}".format(i) for i in range(1, 14)], index=hour_df.index)
        for p in hour_df["Period"].unique():
            sub = hour_df[hour_df["Period"] == p].reset_index(drop=True)[["Date", "AE", "APE"]]
            sub = sub.groupby(by="Date").mean().reset_index(drop=True)
            for i in range(len(sub)):
                day_mae.loc[p, "Day {}".format(i +1)] = sub.loc[i, "AE"]
                day_mape.loc[p, "Day {}".format(i +1)] = sub.loc[i, "APE"]
        for col in day_mae.columns:
            print("MAE {}: {:.2f}".format(col, day_mae[col].mean()))
        print("\n")
        for col in day_mape.columns:
            print("MAPE {}: {:.2f}".format(col, day_mape[col].mean()))
    curve = True
    if curve:
        hour_df = pd.read_csv("../models/curve_model/demand_scores_test_period/hour_demand_results.csv", usecols=["Period", "Date", "Hour", "Demand Forecast"])
        hour_df["Date"] = pd.to_datetime(hour_df["Date"], format="%Y-%m-%d")
        true_demand = get_auction_data("03.06.2019", "31.05.2020", "d", os.getcwd())
        hour_df = hour_df.merge(true_demand, on=["Date", "Hour"])
        errors = {}
        d_class_df = pd.DataFrame(columns=["Class", "MAE", "MAPE"])
        for col in hour_df.columns[4:]:
            ae = abs(hour_df[col]-hour_df["Demand Forecast"]).mean()
            ape = 100 * ae / abs(hour_df[col]).mean()
            errors[ae] = ape
            row = {"Class": col, "MAE": ae, "MAPE": ape}
            d_class_df = d_class_df.append(row, ignore_index=True)
        print(d_class_df)
        d_class_df.to_csv("../analysis/CurveModel/demand_per_class.csv", index=False)
        print("\nMAE tot: {:.0f}".format(sum(errors.keys())/len(errors.keys())))
        print("MAPE tot: {:.2f}%".format(sum(errors.values())/len(errors.values())))
        print("-----------------------------")
        forecast = hour_df["Demand Forecast"]
        errors = hour_df.iloc[:, 4:]
        for c in range(len(errors.columns)):
            errors.iloc[:, c] = abs(errors.iloc[:, c] - forecast)
        errors["Day"] = errors.index // 24 % 14 + 1
        grouped = errors.groupby(by="Day").mean()
        print(grouped.mean(axis=1))
        errors = 100 * errors / hour_df.iloc[:, 4:]
        errors["Day"] = errors.index // 24 % 14 + 1
        errors = errors.groupby(by="Day").mean()
        mean_mapes = errors.mean(axis=1).round(2)
        first_week_mape = mean_mapes.values[0:7]
        week1_mean = first_week_mape.mean()
        print("--------------------")
        print("Mean mape week 1: {:.2f} %".format(week1_mean))
        sec_week_mape = mean_mapes.values[7:len(mean_mapes)]
        week2_mean = sec_week_mape.mean()
        print("Mean mape week 2: {:.2f} %".format(week2_mean))
        print("Decrease of {:.2f} %".format(100*(week2_mean-week1_mean)/week1_mean))
        print("--------------------")



def supply_performance():
    make_result = False
    if make_result:
        all_periods = get_testing_periods()
        df = pd.DataFrame(columns=["Period", "Date", "Hour"])
        s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105,
                     165, 210]
        for s in s_classes:
            df["s {}".format(s)] = np.NAN
        df.to_csv("../analysis/CurveModel/supply_forecast.csv", index=False, float_format='%.2f')
        for j in range(len(all_periods)):
            print(j+1)
            p = all_periods[j]
            s_date = p[0]
            p_df = get_data(s_date, s_date+timedelta(days=13), ["Month", "Weekend"], os.getcwd(), "h")
            p_df["Period"] = np.NaN
            p_df = p_df[["Period", "Date", "Hour", "Month", "Weekend"]]
            for s in s_classes:
                p_df["s {}".format(s)] = np.NAN
            last_week_supply = get_auction_data(s_date - timedelta(days=7), s_date - timedelta(days=1), "s",
                                                os.getcwd())
            supply_mean_week = last_week_supply.groupby(by="Hour").mean().reset_index()
            for i in range(len(p_df)):
                mean_supply_curve = supply_mean_week.iloc[p_df.loc[i, "Hour"]]
                supply = get_supply_curve(p_df.loc[i, "Month"], p_df.loc[i, "Hour"], p_df.loc[i, "Weekend"],
                                          mean_supply_curve, safe=True)
                for v in range(len(supply)):
                    volume = supply.values[v]
                    p_df.iloc[i, v+5] = volume
            p_df = p_df.drop(columns=["Month", "Weekend"])
            p_df["Period"] = j + 1
            p_df = p_df.round(2)
            p_df.to_csv("../analysis/CurveModel/supply_forecast.csv", mode="a", header=False, index=False,
                        float_format='%.2f')
    df = pd.read_csv("../analysis/CurveModel/supply_forecast.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    mae = pd.DataFrame(columns=df.columns)
    mape = pd.DataFrame(columns=df.columns)
    supply = get_auction_data("03.06.2019", "31.05.2020", "s", os.getcwd())
    for period in df["Period"].unique():
        forecast = df[df["Period"] == period].reset_index(drop=True)
        d_1 = forecast.loc[0, "Date"]
        s_sub = supply[(supply["Date"] >= d_1) & (supply["Date"] <= d_1+timedelta(days=13))].reset_index(drop=True)
        for c in range(3, len(forecast.columns)):
            forecast[forecast.columns[c]] = abs(forecast[forecast.columns[c]] - s_sub[s_sub.columns[c-1]])
        mae = mae.append(forecast, ignore_index=True)
        for c in range(3, len(forecast.columns)):
            forecast[forecast.columns[c]] = 100 * forecast[forecast.columns[c]] / s_sub[s_sub.columns[c-1]]
        mape = mape.append(forecast, ignore_index=True)
    mae_mean = mae.drop(columns=["Period", "Date", "Hour"]).mean(axis=0)
    print("MAE supply {:.0f} MWh".format(mae_mean.mean()))
    mape_mean = mape.drop(columns=["Period", "Date", "Hour"]).mean(axis=0)
    print("MAPE supply {:.2f} %".format(mape_mean.mean()))
    s_class_df = pd.DataFrame(columns=["Class", "MAE", "MAPE"])
    s_class_df["Class"] = mape_mean.keys()
    s_class_df["MAE"] = mae_mean.values
    s_class_df["MAPE"] = mape_mean.values
    print(s_class_df)
    s_class_df.to_csv("../analysis/CurveModel/supply_per_class.csv", index=False)
    print("-----------------------------")
    mae_per_day = mae.copy().drop(columns=["Period", "Date", "Hour"])
    mae_per_day["Day"] = mae_per_day.index//24 % 14 + 1
    mae_per_day = mae_per_day.groupby(by="Day").mean()
    print(mae_per_day.mean(axis=1))
    print("-----------------------------")
    mape_per_day = mape.copy().drop(columns=["Period", "Date", "Hour"])
    mape_per_day["Day"] = mape_per_day.index//24 % 14 + 1
    mape_per_day = mape_per_day.groupby(by="Day").mean()
    mape_per_day = mape_per_day.mean(axis=1).round(2)
    print(mape_per_day)
    first_week_mape = mape_per_day.values[0:7]
    week1_mean = first_week_mape.mean()
    print("--------------------")
    print("Mean mape week 1: {:.2f} %".format(week1_mean))
    sec_week_mape = mape_per_day.values[7:len(mape_per_day)]
    week2_mean = sec_week_mape.mean()
    print("Mean mape week 2: {:.2f} %".format(week2_mean))
    print("Decrease of {:.2f} %".format(100 * (week2_mean - week1_mean) / week1_mean))
    print("--------------------")


def plot_perf_per_price_class():
    demand = pd.read_csv("../analysis/CurveModel/demand_per_class.csv")
    supply = pd.read_csv("../analysis/CurveModel/supply_per_class.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_perf(ax1, demand, "Demand", main_col, 1)
    plot_perf(ax2, supply, "Supply", sec_col, 1.05)
    plt.tight_layout(w_pad=3)
    plt.savefig("../analysis/CurveModel/perf_per_class.png")


def plot_perf(ax, data, title, color, ylim_c):
    ax.set_title("{} Curve Performance per Price Class".format(title), y=1.03, size=14)
    ax.set_ylabel("MAE [MWh]", size=11, labelpad=8)
    ax.set_xlabel("{} price class [€]".format(title), size=11)
    mae_mean = data.iloc[:, 1].mean()
    l1 = ax.plot(data.iloc[:, 0], data.iloc[:, 1], color=color)
    a = ax.twinx()
    a.set_ylabel("MAPE [%]", size=11, labelpad=8)
    mape_mean = data.iloc[:, 2].mean()
    l2 = a.plot(data.iloc[:, 0], data.iloc[:, 2], color=color, ls="dotted")
    labs = [d[2:] for d in data.iloc[:, 0].values.astype(str)]
    ax.set_xticklabels(labs, rotation=90)
    for x in [a, ax]:
        x.set_ylim(x.get_ylim()[0], x.get_ylim()[1]*ylim_c)
    labels = ["MAE ({:,})".format(int(mae_mean)), "MAPE ({:.1f}%)".format(mape_mean)]
    for line in a.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
               fancybox=True, shadow=True, handles=[l1[0], l2[0]], labels=labels, prop={"size": 11}).get_lines():
        line.set_linewidth(2)


def test_base_load_test_period():
    df = pd.read_csv("../../data/output/auction/base_load_test.csv")
    print("Mean Base load: {:.2f}".format(df["Water Value"].mean()))
    print("Mean Base load lower: {:.2f}".format(df["WL Lower"].mean()))
    print("Mean Base load upper: {:.2f}".format(df["WL Upper"].mean()))
    print("Mean system price {:.2f}".format(df["System Price"].mean()))


def check_spike_coverage():
    get_result = False
    if get_result:
        df = pd.read_csv("../results/test/CurveModel/forecast.csv")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        result = pd.DataFrame(columns=["Period", "Date", "Hour", "System Price", "Forecast", "Upper", "Lower", "Pos", "Neg"])
        periods = df["Period"].unique()
        data = get_data("01.03.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
        for p in periods:
            forecast = df[df["Period"]==p].reset_index(drop=True)
            dates = forecast["Date"].dt.date.unique()
            for d in dates:
                f = forecast[forecast["Date"].dt.date==d]
                w_end = d - timedelta(days=1)
                w_start = w_end - timedelta(days=59)
                window = data[(data["Date"].dt.date >= w_start) & (data["Date"].dt.date <= w_end)]
                assert len(window) == 1440
                w_mean, st_dev = (window["System Price"].mean(), window["System Price"].std())
                upper, lower = (w_mean+1.96*st_dev, w_mean-1.96*st_dev)
                pos_spikes = f[f["System Price"]>upper]
                neg_spikes = f[f["System Price"]<lower]
                pos_spikes["Pos"] = 1
                pos_spikes["Neg"] = 0
                neg_spikes["Neg"] = 1
                neg_spikes["Pos"] = 0
                result = result.append(pos_spikes, ignore_index=True)
                result = result.append(neg_spikes, ignore_index=True)
        result.to_csv("../analysis/CurveModel/spikes_forecast.csv", index=False)
    else:
        result = pd.read_csv("../analysis/CurveModel/spikes_forecast.csv")
    covered = result[(result["System Price"]<=result["Upper"]) & (result["System Price"]>=result["Lower"])]
    print("Spikes coverage {:.2f} %".format(100*len(covered)/len(result)))
    pos = result[result["Pos"]==1]
    covered = pos[(pos["System Price"]<=pos["Upper"]) & (pos["System Price"]>=pos["Lower"])]
    print("Pos spikes coverage {:.2f} %".format(100*len(covered)/len(pos)))
    neg = result[result["Neg"]==1]
    covered = neg[(neg["System Price"]<=neg["Upper"]) & (neg["System Price"]>=neg["Lower"])]
    print("Neg spikes coverage {:.2f} %".format(100*len(covered)/len(neg)))


def check_predicted_spikes():
    get_result = True
    if get_result:
        df = pd.read_csv("../results/test/CurveModel/forecast.csv")
        n_spikes = 9180
        cov_spike = 0.71
        non_spikes = len(df) - n_spikes
        cov_non_spike = 100 * (len(df) * 0.9455 - n_spikes * cov_spike) / non_spikes
        print("Coverage non-spikes = {:.2f} %".format(cov_non_spike))
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        result = pd.DataFrame(columns=["System Price", "Forecast", "T Pos", "T Neg", "P Pos", "P Neg"])
        periods = df["Period"].unique()
        data = get_data("01.03.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
        for p in periods:
            print(p)
            forecast = df[df["Period"]==p].reset_index(drop=True)
            dates = forecast["Date"].dt.date.unique()
            for d in dates:
                f = forecast[forecast["Date"].dt.date==d]
                w_end = d - timedelta(days=1)
                w_start = w_end - timedelta(days=59)
                window = data[(data["Date"].dt.date >= w_start) & (data["Date"].dt.date <= w_end)]
                assert len(window) == 1440
                w_mean, st_dev = (window["System Price"].mean(), window["System Price"].std())
                upper, lower = (w_mean+1.96*st_dev, w_mean-1.96*st_dev)
                f["T Pos"] = df.apply(lambda r: 1 if r["System Price"]>upper else 0, axis=1)
                f["T Neg"] = df.apply(lambda r: 1 if r["System Price"]<lower else 0, axis=1)
                f["P Pos"] = df.apply(lambda r: 1 if r["Forecast"]>upper else 0, axis=1)
                f["P Neg"] = df.apply(lambda r: 1 if r["Forecast"]>upper else 0, axis=1)
                f = f[["System Price", "Forecast", "T Pos", "T Neg", "P Pos", "P Neg"]]
                result = result.append(f, ignore_index=True)
        result.to_csv("../analysis/CurveModel/spikes_forecast_2.csv", index=False)
    else:
        result = pd.read_csv("../analysis/CurveModel/spikes_forecast_2.csv")
    pred_spikes = result[(result["P Pos"]==1)|(result["P Neg"]==1)]
    pred_pos = pred_spikes[pred_spikes["P Pos"]==1]
    correct_pos = pred_spikes[pred_spikes["T Pos"]==1]
    pred_neg = pred_spikes[pred_spikes["P Neg"]==1]
    correct_neg = pred_spikes[pred_spikes["T Neg"]==1]
    correct_spikes = len(correct_pos) + len(correct_neg)
    print("Predictd spikes turned out as spikes {:.2f} %".format(100*correct_spikes/len(pred_spikes)))
    print("Predictd pos spikes turned out as pos spikes {:.2f} %".format(100*len(correct_pos)/len(pred_pos)))
    print("Predictd neg spikes turned out as neg spikes {:.2f} %".format(100*len(correct_neg)/len(pred_neg)))

def check_trend_coverage():
    output=88.29
    forecast = pd.read_csv("../results/test/CurveModel/forecast.csv")
    assert False
    forecast["Date"] = pd.to_datetime(forecast["Date"], format="%Y-%m-%d")
    find_trend = False
    if find_trend:
        data = get_data("01.05.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
        result = pd.DataFrame(columns=["Period", "Trend", "Trend w1", "Trend w2", "Model", "Model w1", "Model w2"], index=forecast["Period"].unique())
        for period in forecast["Period"].unique():
            sub_df = forecast[forecast["Period"] == period].reset_index(drop=True)
            s_date = sub_df.loc[0, "Date"]
            true_df = data[(data["Date"] >= s_date - timedelta(days=7)) & (data["Date"] <= s_date+timedelta(days=13))].reset_index(drop=True)
            prev_lvl = true_df.head(168)["System Price"].median()
            result.loc[period, "Period"] = period
            result.loc[period, "Trend"] = true_df.tail(336)["System Price"].median() - prev_lvl
            result.loc[period, "Trend w1"] = true_df.loc[168:335, "System Price"].median() - prev_lvl
            result.loc[period, "Trend w2"] = true_df.tail(168)["System Price"].median() - prev_lvl
            result.loc[period, "Model"] = sub_df["Forecast"].median() - prev_lvl
            result.loc[period, "Model w1"] = sub_df.head(168)["Forecast"].median() - prev_lvl
            result.loc[period, "Model w2"] = sub_df.tail(168)["Forecast"].median() - prev_lvl
        pos_trend = result.sort_values(by="Trend").tail(50)
        neg_trend = result.sort_values(by="Trend").head(50)
        print(pos_trend["Period"].tolist())
        print(neg_trend["Period"].tolist())
    p_p = [10, 266, 269, 138, 158, 137, 268, 331, 124, 267, 329, 122, 330, 108, 123, 339, 44, 157, 30, 346, 156, 155,
           154, 153, 146, 151, 152, 150, 345, 344, 147, 43, 343, 340, 148, 149, 342, 341, 42, 31, 32, 41, 40, 33, 37,
           38, 36, 39, 35, 34]
    n_p = [211, 210, 212, 209, 248, 351, 213, 247, 208, 246, 214, 206, 245, 205, 207, 244, 252, 243, 250, 249, 253, 251,
           215, 1, 224, 2, 225, 204, 350, 67, 226, 221, 66, 223, 242, 222, 281, 254, 220, 68, 216, 69, 65, 185, 280,
           186, 184, 70, 283, 282]
    pos_trend = forecast[forecast["Period"].isin(p_p)].reset_index(drop=True)
    neg_trend = forecast[forecast["Period"].isin(n_p)].reset_index(drop=True)
    pos_trend["Pos"] = 1
    pos_trend["Neg"] = 0
    neg_trend["Pos"] = 0
    neg_trend["Neg"] = 1
    df = pos_trend.append(neg_trend, ignore_index=True)
    df = remove_true_spikes(df)
    covered = df[(df["System Price"]>=df["Lower"]) & (df["System Price"]<=df["Upper"])]
    print("Trend coverage {:.2f} %".format(100*len(covered)/len(df)))


def remove_true_spikes(df):
    data = get_data("01.03.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    result = pd.DataFrame(columns=df.columns)
    for p in df["Period"].unique():
        forecast = df[df["Period"] == p].reset_index(drop=True)
        dates = forecast["Date"].dt.date.unique()
        for d in dates:
            f = forecast[forecast["Date"].dt.date == d]
            w_end = d - timedelta(days=1)
            w_start = w_end - timedelta(days=59)
            window = data[(data["Date"].dt.date >= w_start) & (data["Date"].dt.date <= w_end)]
            assert len(window) == 1440
            w_mean, st_dev = (window["System Price"].mean(), window["System Price"].std())
            upper, lower = (w_mean + 1.96 * st_dev, w_mean - 1.96 * st_dev)
            no_spike = f[(f["System Price"]<=upper) & (f["System Price"]>=lower)]
            result = result.append(no_spike, ignore_index=True)
    return result


def check_supply_shifts():
    true_supply = get_auction_data("01.05.2019", "31.05.2020", "s", os.getcwd())
    calculate = False
    if calculate:
        df = pd.read_csv("../analysis/CurveModel/supply_forecast.csv")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        score = pd.DataFrame(columns=["Period", "Start", "MAE", "MAPE"])
        shifts = pd.DataFrame(columns=["Period", "Start", "MAE", "MAPE"])
        for p in df["Period"].unique():
            print(p)
            pred = df[df["Period"]==p].reset_index(drop=True)
            start_d = pred.loc[0, "Date"]
            true = true_supply[(true_supply["Date"]>=start_d) & (true_supply["Date"]<=start_d+timedelta(days=13))].reset_index(drop=True)
            period_true_means = true.iloc[:, 2:].mean(axis=0)
            one_week_before = start_d-timedelta(days=7)
            last_week = true_supply[(true_supply["Date"] >= one_week_before) & (true_supply["Date"]<start_d)].reset_index(drop=True)
            assert len(last_week)==168 and len(true)==336 and len(pred)==336
            last_week_mean = last_week.iloc[:, 2:].mean(axis=0)
            change = abs(period_true_means-last_week_mean)
            change_mean = change.values.mean()
            change_mape = 100*change/last_week_mean
            mape_mean = change_mape.values.mean()
            row = {"Period": p, "Start": start_d, "MAE": change_mean, "MAPE": mape_mean}
            shifts = shifts.append(row, ignore_index=True)
            pred = pred.drop(columns=["Period", "Date", "Hour"])
            true = true.drop(columns=["Date", "Hour"])
            mae = abs(pred-true)
            mape = 100*mae/true
            mae_mean = mae.mean(axis=0).mean()
            mape_mean = mape.mean(axis=0).mean()
            row = {"Period": p, "Start": start_d, "MAE": mae_mean, "MAPE": mape_mean}
            score = score.append(row, ignore_index=True)
        shifts.to_csv("../analysis/CurveModel/supply_shifts_per_period.csv", index=False, float_format='%.2f')
        score.to_csv("../analysis/CurveModel/supply_score_per_period.csv", index=False, float_format='%.2f')
    else:
        shifts = pd.read_csv("../analysis/CurveModel/supply_shifts_per_period.csv")
        shifts["Start"] = pd.to_datetime(shifts["Start"], format="%Y-%m-%d")
        score = pd.read_csv("../analysis/CurveModel/supply_score_per_period.csv")
        score["Start"] = pd.to_datetime(score["Start"], format="%Y-%m-%d")
    cols = ["Wind Prod", "Total Hydro", "Total Hydro Dev", "Prec Norway", "Prec Norway 7", "Coal", "Gas", "EEX"]
    data = get_data("01.05.2019", "31.05.2020", cols, os.getcwd(), "d")
    base_load_marg_test = pd.read_csv("../../data/output/auction/base_load_test.csv")
    cols = base_load_marg_test.columns
    base_load_marg = pd.read_csv("../../data/output/auction/water_values.csv", usecols=cols)
    base_load_marg = base_load_marg.append(base_load_marg_test, ignore_index=True).tail(397)[["Date", "Water Value"]]
    base_load_marg = base_load_marg.rename(columns={"Water Value": "BL MC"})
    base_load_marg["Date"] = pd.to_datetime(base_load_marg["Date"], format="%Y-%m-%d")
    data = pd.merge(data, base_load_marg)
    #most = shifts.sort_values(by="MAPE").tail(60)
    periods = [209, 36, 322, 67, 142, 179]
    most_change = shifts[shifts["Period"].isin(periods)]
    cols = ["Period", "Start"] + data.columns[1:].tolist()
    analysis = pd.DataFrame(columns=cols)
    prev_weeks = pd.DataFrame(columns=cols)
    periods_df = pd.DataFrame(columns=cols)
    plot_most_change = True
    if plot_most_change:
        plot_worst_supply(most_change, true_supply, data)
    for p in periods:
        s_date = most_change[most_change["Period"]==p].head(1)["Start"].values[0]
        prev_week = data[(data["Date"]>s_date-np.timedelta64(8, 'D')) & (data["Date"]<s_date)].iloc[:, 1:].mean(axis=0)
        period = data[(data["Date"]>=s_date) & (data["Date"]<s_date+np.timedelta64(13, 'D'))].iloc[:, 1:].mean(axis=0)
        diff = period - prev_week
        diff["Period"], diff["Start"] = (p, s_date)
        prev_week["Period"], prev_week["Start"] = (p, s_date)
        period["Period"], period["Start"] = (p, s_date)
        prev_weeks = prev_weeks.append(prev_week, ignore_index=True)
        periods_df = periods_df.append(period, ignore_index=True)
        analysis = analysis.append(diff, ignore_index=True)
    prev_weeks = prev_weeks.round(1)
    prev_weeks = prev_weeks.sort_values(by="Start").reset_index(drop=True)
    prev_weeks.to_csv("../analysis/CurveModel/supply_volatility.csv", index=False)
    periods_df = periods_df.round(1)
    periods_df = periods_df.sort_values(by="Start").reset_index(drop=True)
    periods_df.to_csv("../analysis/CurveModel/supply_volatility.csv", mode="a", header=False, index=False)
    analysis = analysis.round(1)
    analysis = analysis.sort_values(by="Start").reset_index(drop=True)
    analysis.to_csv("../analysis/CurveModel/supply_volatility.csv", mode="a", header=False, index=False)


def plot_worst_supply(most_change, true_supply, data):
    s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105,
                 165, 210]
    most_change = most_change.reset_index(drop=True)
    cols = ["Period"]
    cols.extend([c for c in true_supply.columns])
    df_last_week = pd.DataFrame(columns=cols)
    df_period = pd.DataFrame(columns=df_last_week.columns)
    periods = most_change["Period"].unique()
    period_blmc = {}
    for i in range(len(periods)):
        period = periods[i]
        s_date = most_change.loc[i, "Start"]
        prev_data = data[(data["Date"]>=s_date-timedelta(days=7)) & (data["Date"]<s_date)]
        prev_blmc = prev_data["BL MC"].mean()
        prev_week = true_supply[(true_supply["Date"]>=s_date-timedelta(days=7)) & (true_supply["Date"]<s_date)]
        assert len(prev_week) == 168
        prev_week["Period"] = period
        df_last_week = df_last_week.append(prev_week, ignore_index=True)
        per_data = data[(data["Date"]>=s_date) & (data["Date"]<=s_date+timedelta(days=13))]
        per_blmc = per_data["BL MC"].mean()
        period_blmc[period] = [prev_blmc, per_blmc]
        period_df = true_supply[(true_supply["Date"]>=s_date) & (true_supply["Date"]<=s_date+timedelta(days=13))]
        period_df["Period"] = period
        df_period = df_period.append(period_df, ignore_index=True)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    s_col = plt.get_cmap("tab10")(1)
    for i in range(len(periods)):
        period = periods[i]
        prev_blmc, per_blmc = (period_blmc[period][0], period_blmc[period][1])
        mape = most_change[most_change["Period"]==period].head(1)["MAPE"].values[0]
        last_week = df_last_week[df_last_week["Period"]==period]
        last_week_mean = last_week.iloc[:, 3:].mean(axis=0).values
        last_week_mean = last_week_mean / 1000
        this_period = df_period[df_period["Period"]==period].reset_index(drop=True)
        s_date = this_period.loc[0, "Date"]
        e_date = s_date + timedelta(days=13)
        s_date, e_date = (dt.strftime(s_date, "%d  %b"), dt.strftime(e_date, "%d  %b"))
        period_mean = this_period.iloc[:, 3:].mean(axis=0).values
        period_mean = period_mean / 1000
        x_max = max(period_mean[-3], last_week_mean[-3])
        x_min = min(period_mean[0], last_week_mean[0])
        ax = axs[i//3][i%3]
        if i in [0, 3]:
            ax.set_ylabel("Price [€]", size=11)
        if i in [3, 4, 5]:
            ax.set_xlabel("Volume [GWh]", size=11)
        prev_line = LineString(np.column_stack((last_week_mean, s_classes)))
        prev_bline = LineString([(min(last_week_mean), prev_blmc), (max(last_week_mean), prev_blmc)])
        prev_x = prev_bline.intersection(prev_line).x
        per_line = LineString(np.column_stack((period_mean, s_classes)))
        per_bline = LineString([(min(period_mean), per_blmc), (max(period_mean), per_blmc)])
        per_x = per_bline.intersection(per_line).x
        print("Period {} hor shift: {:.1f} GWh (from {:.1f} to {:.1f})".format(period, per_x-prev_x, prev_x, per_x))
        ax.plot(last_week_mean, s_classes, color=s_col, ls="dotted", lw=2, label="Last week mean")
        ax.plot(period_mean, s_classes, color=s_col, lw=2, label="Period mean")
        ax.arrow(prev_x, prev_blmc, per_x-prev_x, per_blmc-prev_blmc, head_width=2, head_length=2, width=0.5,
                 length_includes_head=True, color="black", zorder=100)
        for line in ax.legend(loc='upper left', ncol=1,
                             fancybox=True, shadow=True, prop={"size": 11}).get_lines():
            line.set_linewidth(2)
        ax.set_ylim(-10, 80)
        ax.set_xlim(x_min, x_max)
        ax.set_title("{})  {} - {} ($\Delta={:.1f}\%$)".format(i+1, s_date, e_date, mape), size=13)
    fig.suptitle("Examples of Supply Volatility", size=14, y=0.99)
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/supply_volatility.png")


def r_wind():
    x = [-5.2, 3, 3.7, 3.2, 5.4, -5.1]
    y = [-3.6, 2.7, 1.7, 3.4, 6.5, -4.6]
    r_coeff = round(stats.pearsonr(x, y)[0], 2)
    print("Correlation hor and wind {}".format(r_coeff))
    x = [4.2, -2.3, 1.7, -0.6, -5.1, 0.3]
    y = [-2.8, 0.5, -1.3, 0.3, 4.4, -3]
    r_coeff = round(stats.pearsonr(x, y)[0], 2)
    print("Correlation vert and hydro dev {}".format(r_coeff))
    y = [-52.6, 111.8, -15.7, 62.7, 132.7, -72.1]
    r_coeff = round(stats.pearsonr(x, y)[0], 2)
    print("Correlation vert and prec {}".format(r_coeff))
    y = [3.2, -1.1, -2.4, -0.2, -0.8, -2.9]
    r_coeff = round(stats.pearsonr(x, y)[0], 2)
    print("Correlation vert and coal {}".format(r_coeff))
    y = [8.3, -6.9, -0.1, -8.7, 5.2, 2]
    r_coeff = round(stats.pearsonr(x, y)[0], 2)
    print("Correlation vert and EEX {}".format(r_coeff))


if __name__ == '__main__':
    # plot_all_monthly_errors_all_models()
    # path_names_ = ["CurveModel_mean_supply", "CurveModel_351_1"]
    # compare_results_two_models(path_names_)
    # find_best_and_worst_periods_curve_model()
    # demand_and_supply_errors_curve_model()
    # t_test()
    # monthly_error_double()
    # error_distributions()
    # prob_dist()
    # example_periods()
    # example_curves()
    combined_period_curves()
    # t_test_2()
    # val_vs_test()
    # check_data_test_val()
    # assess_spike_detection()
    # performance_without_spikes()
    # trend_detection()
    # demand_performance()
    # supply_performance()
    # plot_perf_per_price_class()
    # test_base_load_test_period()
    # check_spike_coverage()
    # check_predicted_spikes()
    # check_trend_coverage()
    # check_supply_shifts()
    # r_wind()

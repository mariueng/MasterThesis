# script for generating validation periods
from data import data_handler
import pandas as pd
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def get_seasons():
    winter = ("07.01.2019", "31.03.2019")
    spring = ("01.04.2019", "30.06.2019")
    summer = ("01.07.2019", "29.09.2019")
    fall = ("30.09.2019", "29.12.2019")
    return [winter, spring, summer, fall]


def get_four_periods_median_method(write_summary):
    periods = []
    seasons = get_seasons()
    for s in seasons:
        period_table = pd.DataFrame(columns=["Start date", "End date", "Mean", "Variance", "Mean Dev", "Var Dev",
                                             "Norm Mean Dev", "Norm Var Dev", "Score"])
        data = data_handler.get_data(s[0], s[1], ["System Price"], os.getcwd(), "h")
        first_date_in_season = data.iloc[0]["Date"].date()
        last_date_in_season = data.iloc[-1]["Date"].date()
        number_of_periods = math.floor((len(data) / 24) / 7)
        if write_summary:
            print("First date in season: {}".format(first_date_in_season))
            print("Last date in season: {}".format(last_date_in_season))
            print("Number of periods in season: {}\n".format(number_of_periods))
        for i in range(number_of_periods):
            first_date = first_date_in_season + dt.timedelta(days=7 * i)
            last_date = np.datetime64(first_date + dt.timedelta(days=13))
            first_date = np.datetime64(first_date)
            period = (data['Date'] >= first_date) & (data['Date'] <= last_date)
            period_data = data[period]
            period_mean = period_data["System Price"].mean()
            period_variance = period_data.var()["System Price"]
            row = {"Start date": first_date, "End date": last_date, "Mean": period_mean, "Variance": period_variance}
            period_table = period_table.append(row, ignore_index=True)

        season_mean = period_table["Mean"].mean()
        season_variance = period_table["Variance"].mean()


        for index, period in period_table.iterrows():
            period_mean_dev = abs(period["Mean"] - season_mean)
            period_var_dev = abs(period["Variance"] - season_variance)
            period_table.at[index, "Mean Dev"] = period_mean_dev
            period_table.at[index, "Var Dev"] = period_var_dev

        mean_dev_mean = period_table["Mean Dev"].mean()
        mean_dev_var = period_table["Var Dev"].mean()

        for index, period in period_table.iterrows():
            norm_dev_mean = abs(period["Mean Dev"] / mean_dev_mean)
            norm_dev_var = abs(period["Var Dev"] / mean_dev_var)
            score = norm_dev_var + norm_dev_mean
            period_table.at[index, "Norm Mean Dev"] = norm_dev_mean
            period_table.at[index, "Norm Var Dev"] = norm_dev_var
            period_table.at[index, "Score"] = score

        period_table = period_table.sort_values(by=["Score"], ignore_index=True)
        start_date_optimal_period = period_table.at[0, "Start date"].date()
        end_date_optimal_period = period_table.at[0, "End date"].date()
        periods.append((start_date_optimal_period, end_date_optimal_period))

        if write_summary:
            print("Season mean: {}, season variance: {}".format(round(season_mean, 2), round(season_variance, 2)))
            print(period_table[["Mean", "Variance", "Norm Mean Dev", "Norm Var Dev", "Score"]])
            print("Optimal start: {}, optimal end: {}".format(start_date_optimal_period, end_date_optimal_period))
            print("\n")

    if write_summary:
        print("The following four periods are suggested as validation test periods")
        for p in periods:
            print(p)
        print("\n")
    return periods


# method for plotting the periods during the year
def plot_price_year_and_periods(periods, write_dates):
    print("--Plotting validation periods for 2019--\n")
    price_color = "steelblue"
    per_color = "firebrick"
    label_pad = 12
    title_pad = 20
    data = data_handler.get_data("01.01.2019", "31.12.2019", ["System Price"], "h")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plt.plot(data["Date"], data["System Price"], label="SYS", color=price_color)
    period = 1
    for p in periods:
        mask = (data['Date'] >= np.datetime64(p[0])) & (data['Date'] <= np.datetime64(p[1]))
        period_data = data[mask]
        plt.plot(period_data["Date"], period_data["System Price"], color=per_color, label="Per. {}".format(period))
        period += 1
    for line in plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    for s in get_seasons():
        if get_seasons().index(s) == 0:
            start_date = dt.datetime.strptime(s[0], '%d.%m.%Y')
            plt.axvline(start_date, linewidth=1, color='black', linestyle="--")
        end_date = dt.datetime.strptime(s[1], '%d.%m.%Y')
        plt.axvline(end_date, linewidth=1, color='black', linestyle="--")
    if write_dates:
        plt.ylim(0, max(data["System Price"])*1.12)
        for i in range(len(periods)):
            p = periods[i]
            start_string = p[0].strftime("%d %b")
            end_string = p[1].strftime("%d %b")
            date_string = start_string + " - " + end_string
            plt.text(i*0.22+0.12, 0.92, date_string, transform = ax.transAxes, color=per_color)
    plt.ylabel("System price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.title("Validation Periods", pad=title_pad)
    plt.tight_layout()
    out_path = "../../data/output/plots/random/val_periods_median_method.png"
    plt.savefig(out_path)
    plt.close()


def get_all_2019_periods():
    periods = []
    first_date = dt.datetime(2019, 1, 1).date()
    while first_date != dt.datetime(2019, 12, 19).date():
        last_date = first_date + dt.timedelta(days=13)
        periods.append((first_date, last_date))
        first_date = first_date + dt.timedelta(days=1)
    return periods


def get_validation_periods():
    periods = []
    first_date = dt.datetime(2018, 6, 4).date()
    while first_date != dt.datetime(2019, 5, 21).date():
        last_date = first_date + dt.timedelta(days=13)
        periods.append((first_date, last_date))
        first_date = first_date + dt.timedelta(days=1)
    return periods


def get_testing_periods():
    periods = []
    first_date = dt.datetime(2019, 6, 3).date()
    while first_date != dt.datetime(2020, 5, 19).date():
        last_date = first_date + dt.timedelta(days=13)
        periods.append((first_date, last_date))
        first_date = first_date + dt.timedelta(days=1)
    return periods


def get_one_period():
    periods = []
    first_date = dt.datetime(2019, 1, 28).date()
    last_date = first_date + dt.timedelta(days=13)
    periods.append((first_date, last_date))
    return periods


def get_random_periods(number):
    periods = []
    random.seed(1)
    all_val_dates = pd.date_range(dt.datetime(2018, 6, 4), dt.datetime(2019, 5, 19), freq="d").tolist()
    chosen_dates = random.sample(all_val_dates, number)
    chosen_dates.sort()
    for d in chosen_dates:
        periods.append((d.date(), d.date()+dt.timedelta(days=13)))
    return periods


if __name__ == '__main__':
    # periods_ = get_four_periods_median_method(write_summary=False)
    #print(periods_)
    # plot_price_year_and_periods(periods_, write_dates=True)
    #periods_2019 = get_all_2019_periods()
    #print(periods_2019)
    # periods_ = get_one_period()
    # periods_ = get_testing_periods()
    periods_ = get_random_periods(10)
    print(periods_)

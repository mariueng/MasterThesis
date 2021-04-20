import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from data.data_handler import get_data
import calendar
pd.options.mode.chained_assignment = None  # default='warn'

first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)
half_fig = (6.5, 7)


def decompose_daily_to_hourly_demand(day_demand_forecast, start_date, end_date):
    prev_workdir = os.getcwd()
    print(prev_workdir)
    os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
    data = get_data(start_date, end_date, ["Month", "Weekday"], os.getcwd(), "d")
    day_demand_forecast = day_demand_forecast.merge(data, on="Date")
    result = get_data(start_date, end_date, [], os.getcwd(), "h")
    result["Demand Forecast"] = np.NAN
    profiles = pd.read_csv(r"hourly_demand_profiles.csv")
    for i in range(len(day_demand_forecast)):
        daily_demand = day_demand_forecast.loc[i, "Demand Forecast"]
        month = day_demand_forecast.loc[i, "Month"]
        weekday = day_demand_forecast.loc[i, "Weekday"]
        profile = profiles[(profiles["Month"] == month) & (profiles["Weekday"] == weekday)].reset_index(drop=True)
        for j in range(24):
            result.loc[i * 24 + j, "Demand Forecast"] = daily_demand * profile[str(j)][0]
    os.chdir(prev_workdir)
    return result


def explore_daily_demand():
    demand_col = "Total Vol"
    hourly_demand = get_data("01.01.2014", "31.12.2019", [demand_col, "Weekday", "Month"], os.getcwd(), "h")
    normal = True
    weekday = True
    month = True
    if normal:
        panel = get_hourly_panel(hourly_demand, demand_col)
        plot_hourly_demand(panel, "Hourly Demand 2014 to 2019")
    if weekday:
        for i in range(7):
            day = list(calendar.day_name)[i]
            df = hourly_demand[hourly_demand["Weekday"] == i + 1].reset_index(drop=True)
            panel = get_hourly_panel(df, demand_col)
            plot_hourly_demand(panel, "Hourly Demand 2014 to 2019, {}".format(day))
            plt.bar(panel["Hour"], panel["Portion"])
    if month:
        for i in range(1, 13):
            cur_month = calendar.month_name[i]
            df = hourly_demand[hourly_demand["Month"] == i].reset_index(drop=True)
            panel = get_hourly_panel(df, demand_col)
            plot_hourly_demand(panel, "Hourly Demand 2014 to 2019, {}".format(cur_month))
            plt.bar(panel["Hour"], panel["Portion"])


def plot_hourly_demand(df, title):
    plt.subplots(figsize=half_fig)
    plt.bar(df["Hour"], df["Portion"], color=first_color)
    plt.title(title, pad=title_pad)
    plt.xlabel("Hour of day", labelpad=label_pad)
    plt.ylabel("Mean portion of daily demand", labelpad=label_pad)
    plt.savefig("eda/{}.png".format(title.replace(" ", "_").replace(",", "").lower()))
    plt.close()


def get_hourly_panel(hour_df, demand_col):
    df = pd.DataFrame(columns=["Hour", "Portion"])
    df["Hour"] = range(24)
    df["Portion"] = np.NZERO
    first_date = hour_df.loc[0, "Date"]
    last_date = hour_df.loc[len(hour_df) - 1, "Date"]
    all_dates = hour_df["Date"].unique()
    for i in range(len(all_dates)):
        j = i + 1
        date = all_dates[i]
        hourly_demand = hour_df[hour_df["Date"] == date].reset_index(drop=True)
        sum_of_hourly_demand = hourly_demand[demand_col].sum()
        hourly_demand["Demand portion"] = hourly_demand[demand_col] / sum_of_hourly_demand
        df["Portion"] = (df["Portion"] * i + hourly_demand["Demand portion"]) / j
    return df


def save_hourly_demand_profiles():
    columns = ["Month", "Weekday"] + [str(i) for i in range(24)]
    df = pd.DataFrame(columns=columns)
    demand_col = "Total Vol"
    hourly_demand = get_data("01.01.2014", "31.12.2019", [demand_col, "Weekday", "Month"], os.getcwd(), "h")
    for month in range(1, 13):
        month_df = hourly_demand[hourly_demand["Month"] == month]
        for weekday in range(1, 8):
            row = {"Month": month, "Weekday": weekday}
            month_weekday_df = month_df[month_df["Weekday"] == weekday].reset_index(drop=True)
            panel = get_hourly_panel(month_weekday_df, demand_col)
            for hour in range(24):
                row[str(hour)] = round(panel.loc[hour, "Portion"], 6)
            df = df.append(row, ignore_index=True)
    df.to_csv("hourly_demand_profiles.csv", index=False, float_format="%g")


def evaluate_demand_profiles():
    demand_col = "Total Vol"
    data = get_data("01.01.2019", "31.12.2019", [demand_col, "Month", "Weekday"], os.getcwd(), "h")
    profiles = pd.read_csv(r"hourly_demand_profiles.csv")
    dates = pd.date_range(data.loc[0, "Date"], data.loc[len(data)-1, "Date"], freq="d")
    ae = []
    ape = []
    for i in range(len(dates)):
        date = dates[i]
        hourly_demand = data[data["Date"] == date].reset_index(drop=True)
        sum_of_hourly_demand = hourly_demand[demand_col].sum()
        month = hourly_demand.loc[0, "Month"]
        weekday = hourly_demand.loc[0, "Weekday"]
        profile = profiles[(profiles["Month"] == month) & (profiles["Weekday"] == weekday)].drop(columns=["Month", "Weekday"]).transpose()
        hourly_demand["Profile"] = profile.values
        hourly_demand["Demand Forecast"] = sum_of_hourly_demand * hourly_demand["Profile"]
        hourly_demand["AE"] = abs(hourly_demand[demand_col] - hourly_demand["Demand Forecast"])
        hourly_demand["APE"] = 100 * hourly_demand["AE"] / hourly_demand[demand_col]
        ae.append(hourly_demand["AE"].mean())
        ape.append(hourly_demand["APE"].mean())
    print("MAE hourly demand decomposition: {:.1f}".format(sum(ae) / len(ae)))
    print("MAPE hourly demand decomposition: {:.2f}".format(sum(ape) / len(ape)))



if __name__ == '__main__':
    # save_hourly_demand_profiles()
    evaluate_demand_profiles()

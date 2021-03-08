import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
import requests
import os
from data.data_handler import get_data
import pandas as pd
import warnings
import numpy as np

label_pad = 12
title_pad = 20

def plot_norm_weekday():
    start_date = dt(2019, 1, 1)
    end_date = dt(2019, 1, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday"], os.getcwd(), "h")
    # training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    grouped_df = training_data.groupby(by="Weekday").mean()
    normalized_df = (grouped_df - grouped_df.mean()) / grouped_df.std()
    plt.subplots(figsize=(6.5, 7))
    true_color = "steelblue"
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.bar(days, normalized_df["System Price"], color=true_color)
    plt.title("Mean Normalized Price Per Weekday 2019", pad=title_pad)
    plt.ylabel("Norm. Price", labelpad=label_pad)
    plt.xlabel("Day of week", labelpad=label_pad)
    y_min = min(normalized_df["System Price"]) * 1.1
    y_max = max(normalized_df["System Price"]) * 1.1
    plt.ylim(y_min, y_max)
    for i, v in enumerate(normalized_df["System Price"].tolist()):
        sys = round(grouped_df["System Price"].tolist()[i], 1)
        if v < 0:
            pad = -0.1
        else:
            pad = 0.05
        plt.text(i, v + pad, sys, color="steelblue", fontweight='bold', ha='center')
    plt.tight_layout()
    path = "output/plots/eda/price_per_week_day_2019_2.png"
    plt.savefig(path)


def plot_norm_month():
    warnings.filterwarnings("ignore")
    start_date = dt(2014, 1, 1)
    end_date = dt(2019, 12, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Month"], os.getcwd(), "d")
    years = [i for i in range(start_date.year, end_date.year + 1)]
    result_df = pd.DataFrame(columns=["Month"])
    result_df["Month"] = range(1, 13)
    for year in years:
        df_year = training_data[training_data["Date"].dt.year == year]
        df_year["System Price"] = (df_year["System Price"] - df_year["System Price"].mean()) / df_year[
            "System Price"].std()
        df_year = df_year.groupby(by="Month").mean()
        df_year = df_year.rename(columns={"System Price": "Price {}".format(year)})
        result_df = result_df.merge(df_year, on="Month", how="outer")
    col = [c for c in result_df.columns if c != "Month"]
    result_df['mean'] = result_df[col].mean(axis=1)
    plt.subplots(figsize=(6.5, 7))
    bar_color = "steelblue"
    plt.bar(result_df["Month"], result_df["mean"], color=bar_color)
    plt.xlabel("Month", labelpad=label_pad)
    plt.ylabel("Norm. mean price", labelpad=label_pad)
    plt.title("Normalized Mean Price Per Month 2014-2019", pad=title_pad)
    plt.tight_layout()
    path = "output/plots/eda/price_per_month_2014_2019.png"
    plt.savefig(path)


def plot_daily_vs_hourly_prices():
    start_date = dt(2019, 6, 1)
    end_date = dt(2019, 6, 30)
    df_h = get_data(start_date, end_date, ["System Price"], os.getcwd(), "h")
    df_h = df_h.rename(columns={'System Price': "Hourly Price"})
    df_h["Hour"] = pd.to_datetime(df_h['Hour'], format="%H").dt.time
    df_h["DateTime"] = df_h.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    df_d = get_data(start_date, end_date, ["System Price"], os.getcwd(), "d")
    df_d = df_d.rename(columns={'System Price': "Daily Price"})
    df_d["DateTime"] = df_d["Date"] + timedelta(hours=12)
    f, ax = plt.subplots(figsize=(13, 7))
    hour_col = "steelblue"
    day_col = "firebrick"
    plt.plot(df_h["DateTime"], df_h["Hourly Price"], color=hour_col, label="Hourly price", linewidth=1.5)
    plt.plot(df_d["DateTime"], df_d["Daily Price"], color=day_col, label="Daily price", linewidth=2.5)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    plt.title("Daily and Hourly Electricity Price", pad=title_pad)
    plt.tight_layout()
    date_string = dt.strftime(start_date, "%Y_%m-%d") + "_" + dt.strftime(end_date, "%Y_%m-%d")
    path = "output/plots/eda/hourly_and_daily_prices_{}.png".format(date_string)
    plt.savefig(path)


def plot_norm_week():
    start_date = dt(2014, 1, 1)
    end_date = dt(2019, 12, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Week"], os.getcwd(), "d")
    years = [i for i in range(start_date.year, end_date.year + 1)]
    result_df = pd.DataFrame(columns=["Week"])
    result_df["Week"] = range(1, 53)
    for year in years:
        df_year = training_data[training_data["Date"].dt.year == year]
        df_year = df_year[df_year["Week"] <= 52]  # exclude 53
        df_year["System Price"] = (df_year["System Price"] - df_year["System Price"].mean()) / df_year[
            "System Price"].std()
        df_year = df_year.groupby(by="Week").mean()
        df_year = df_year.rename(columns={"System Price": "Price {}".format(year)})
        result_df = result_df.merge(df_year, on="Week", how="outer")
    col = [c for c in result_df.columns if c != "Week"]
    result_df['mean'] = result_df[col].mean(axis=1)
    plt.subplots(figsize=(6.5, 7))
    bar_color = "steelblue"
    plt.bar(result_df["Week"], result_df["mean"], color=bar_color)
    plt.xlabel("Week", labelpad=label_pad)
    plt.ylabel("Norm. mean price", labelpad=label_pad)
    plt.title("Normalized Mean Price Per Week 2014-2019", pad=title_pad)
    plt.tight_layout()
    path = "output/plots/eda/price_per_week_2014_2019.png"
    plt.savefig(path)


def plot_temperatures():
    t_columns = ["T Nor", "T Hamar", "T Krsand", "T Troms", "T Namsos", "T Bergen"]
    df = get_data("01.01.2019", "31.12.2019", t_columns, os.getcwd(), "d")
    plt.subplots(figsize=(13, 7))
    for col in t_columns:
        if "Nor" in col:
            width = 4
        else:
            width = 1
        plt.plot(df["Date"], df[col], label=col[1:], linewidth=width)
    for line in plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Temperature Norway 2019", pad=title_pad)
    plt.ylabel("Celsius", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ymax = max(df[t_columns].max())*1.1
    ymin = min(df[t_columns].min())*0.9
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    path = "output/plots/eda/temperature_norway_2019.png"
    plt.savefig(path)

if __name__ == '__main__':
    print("Running eda..")
    # plot_norm_weekday()
    # plot_norm_month()
    # plot_daily_vs_hourly_prices()
    # plot_norm_week()
    plot_temperatures()


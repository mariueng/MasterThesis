import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
import requests
from pathlib import Path
import os
from data.data_handler import get_data
import pandas as pd
import warnings
import matplotlib.ticker as ticker
from src.system.scores import get_all_point_metrics
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from shapely.geometry import LineString

label_pad = 12
title_pad = 20
full_fig = (13, 7)
half_fig = (6.5, 7)
first_color = "steelblue"
sec_color = "firebrick"
third_color = "darkorange"


def plot_norm_weekday():
    start_date = dt(2019, 1, 1)
    end_date = dt(2019, 1, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday"], os.getcwd(), "h")
    # training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    grouped_df = training_data.groupby(by="Weekday").mean()
    normalized_df = (grouped_df - grouped_df.mean()) / grouped_df.std()
    plt.subplots(figsize=half_fig)
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
    plt.subplots(figsize=half_fig)
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
    f, ax = plt.subplots(figsize=full_fig)
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
    plt.subplots(figsize=half_fig)
    bar_color = "steelblue"
    plt.bar(result_df["Week"], result_df["mean"], color=bar_color)
    plt.xlabel("Week", labelpad=label_pad)
    plt.ylabel("Norm. mean price", labelpad=label_pad)
    plt.title("Normalized Mean Price Per Week 2014-2019", pad=title_pad)
    plt.tight_layout()
    path = "output/plots/eda/price_per_week_2014_2019.png"
    plt.savefig(path)


def plot_temperatures():
    t_columns = ["Norway", "Hamar", "Krsand", "Troms", "Namsos", "Bergen"]
    t_columns = ["Temp {}".format(i) for i in t_columns]
    df = get_data("01.01.2019", "31.12.2019", t_columns, os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    for col in t_columns:
        if "Norway" in col:
            width = 4
        else:
            width = 1
        plt.plot(df["Date"], df[col], label=col[5:], linewidth=width)
    for line in plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Temperature Norway 2019", pad=title_pad)
    plt.ylabel("Celsius", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ymax = max(df[t_columns].max()) * 1.1
    ymin = min(df[t_columns].min()) * 0.9
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    path = "output/plots/eda/temperature_norway_2019.png"
    plt.savefig(path)


def plot_precipitation():
    t_columns = ["Norway", "Hamar", "Kvin", "Troms", "Oppdal", "Bergen"]
    t_columns = ["Prec {}".format(i) for i in t_columns]
    df = get_data("01.01.2019", "31.12.2019", t_columns, os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    for col in t_columns:
        plt.plot(df["Date"], df[col], label=col[5:], linewidth=1)
    for line in plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Precipitation Norway 2019", pad=title_pad)
    plt.ylabel("Mm", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    ymax = max(df[t_columns].max()) * 1.1
    ymin = min(df[t_columns].min()) * 0.9
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    path = "output/plots/eda/precipitation_norway_2019.png"
    plt.savefig(path)


def plot_col_per_year(col_name, title, ylabel):
    data = get_data("01.01.2014", "31.12.2019", [col_name], os.getcwd(), "d")
    fig, ax = plt.subplots(figsize=full_fig)
    horizon = [y for y in range(2014, 2020)]
    for year in horizon:
        start = dt(year, 1, 1)
        end = dt(year, 12, 31)
        mask = (data['Date'] >= start) & (data['Date'] <= end)
        sub_df = data.loc[mask].reset_index()
        plt.plot(sub_df[col_name].values, label=str(year))
    ticks = []
    for x in sub_df["Date"].dt.strftime('%b'):
        if x not in ticks:
            ticks.append(x)
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs=[30 * i + ((i + 1) % 2) + 15 for i in range(12)], nbins=12))
    ax.set_xticklabels(ticks)
    for line in plt.legend(loc='upper center', ncol=len(horizon), bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title(title, pad=title_pad)
    plt.ylabel(ylabel, labelpad=label_pad)
    plt.xlabel("Month", labelpad=label_pad)
    plt.tight_layout()
    path = "output/plots/eda/" + title.lower().replace(" ", "_") + ".png"
    print(path)
    plt.savefig(path)
    plt.close()


def plot_all_variables_per_year():
    ylabel = "Price [€]"
    title = "Daily System Price per Year"
    col_name = "System Price"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Celsius"
    title = "Avg. Daily Temp. Norway per Year"
    col_name = "Temp Norway"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Reservoir GWh"
    title = "Daily Accumulated Hydro Dev. per Year"
    col_name = "Total Hydro Dev"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Volume MWh"
    title = "Daily Volume per Year"
    col_name = "Total Vol"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Supply MWh"
    title = "Daily Supply per Year"
    col_name = "Supply"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Demand MWh"
    title = "Daily Demand per Year"
    col_name = "Demand"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Reservoir GWh"
    title = "Acc. Hydro Level per Year"
    col_name = "Total Hydro"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Production MWh"
    title = "Wind Production Denmark per Year"
    col_name = "Wind DK"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Mm"
    title = "Precipitation Seven Days Ahead"
    col_name = "Prec Forecast"
    plot_col_per_year(col_name, title, ylabel)


def plot_correlation(dep_variable, expl_variable):
    df = get_data("01.01.2014", "31.12.2019", [dep_variable, expl_variable], os.getcwd(), "d")
    df = df[[expl_variable, dep_variable]]
    plt.subplots(figsize=full_fig)
    r_coeff = round(stats.pearsonr(df[dep_variable], df[expl_variable])[0] ** 2, 3)
    sns.regplot(x=expl_variable, y=dep_variable, data=df, scatter_kws={"color": first_color}, label="Days in 2014-2019",
                line_kws={"color": sec_color, "label": "Regression (R$^2$ = {})".format(r_coeff)})
    plt.xlabel("{} [{}]".format(expl_variable, get_suffix(expl_variable)), labelpad=label_pad)
    plt.ylabel("{} [{}]".format(dep_variable, get_suffix(dep_variable)), labelpad=label_pad)
    plt.title("{} vs. {}".format(get_word_col_name(expl_variable), get_word_col_name(dep_variable)), pad=title_pad,
              fontsize=14)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    path = "output/plots/eda/reg_" + dep_variable.replace(" ", "_") + "_" + expl_variable.replace(" ", "_") + ".png"
    print("Saved to {}".format(path))
    plt.savefig(path)
    plt.close()


def get_suffix(column):
    if "Price" in column or "Coal" in column or "Oil" in column or "Gas" in column or "Low Carbon" in column:
        return "€"
    elif "Hydro" in column:
        return "GWh"
    elif "Temp Norway" in column:
        return "$℃$"
    elif "Prec" in column:
        return "Mm"
    else:
        return "MWh"


def plot_correlation_norm_price_per_year(dep_variable, expl_variable):
    df = get_data("01.01.2014", "31.12.2019", [dep_variable, expl_variable], os.getcwd(), "d")
    df["Norm Price"] = np.NAN
    for year in range(2014, 2020):
        df_year = df[df["Date"].dt.year.isin([year])]
        df_year["Norm Price"] = (df_year["System Price"] - df_year["System Price"].mean())
        df.loc[df_year.index, "Norm Price"] = df_year["Norm Price"]
    dep_variable = "Norm Price"
    df_2019 = df[df["Date"].dt.year.isin([2019])]
    df = df[df["Date"].dt.year.isin(range(2014, 2019))]
    plt.subplots(figsize=full_fig)
    r_coeff = round(stats.pearsonr(df[dep_variable], df[expl_variable])[0] ** 2, 3)
    sns.regplot(x=expl_variable, y=dep_variable, data=df, scatter_kws={"color": first_color}, label="Days in 2014-2018",
                line_kws={"color": sec_color, "label": "Regression 2014-2018 (R$^2$ = {})".format(r_coeff)})
    sns.regplot(x=expl_variable, y=dep_variable, data=df_2019, scatter_kws={"color": sec_color}, label="Days in 2019",
                fit_reg=False)
    plt.xlabel("{} [{}]".format(expl_variable, get_suffix(expl_variable)), labelpad=label_pad)
    plt.ylabel("{} [{}]".format(dep_variable, get_suffix(dep_variable)), labelpad=label_pad)
    plt.title("{} vs. {}".format(get_word_col_name(expl_variable), get_word_col_name(dep_variable)), pad=title_pad,
              fontsize=14)
    for line in plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    path = "output/plots/eda/reg_" + dep_variable.replace(" ", "_") + "_" + expl_variable.replace(" ",
                                                                                                  "_") + "_2019_marked.png"
    print("Saved to {}".format(path))
    plt.savefig(path)
    plt.close()


def get_word_col_name(col_name):
    replace_dict = {"Temp Norway": "Temperature Norway", "Total Vol": "Volume", "Total Hydro": "Acc. Hydro Level",
                    "Total Hydro Dev": "Acc. Hydro Level Deviation", "Wind DK": "Wind Denmark", "Prec Forecast":
                        "Precipitation Forecast"}
    if col_name not in replace_dict.keys():
        return col_name
    else:
        return replace_dict[col_name]


def check_lagged_correlation(dep_variable, pred_variable):
    df = get_data("01.01.2014", "31.12.2019", [dep_variable, pred_variable], os.getcwd(), "d")
    df = df[[pred_variable, dep_variable]]
    lags = [i for i in range(-11, 8)]
    for i in lags:
        df[str(i)] = df[pred_variable].shift(i)
        df_i = df[[str(i), dep_variable]]
        df_i = df_i.dropna()
        r_coeff = round(stats.pearsonr(df_i[dep_variable], df_i[str(i)])[0], 4)
        print("Shifting {} {} days gives pearson coeff: {}".format(pred_variable, i, r_coeff))


def lin_model_test():
    cols = ["System Price", "Total Hydro Dev"]
    train = get_data("01.01.2014", "31.12.2018", cols, os.getcwd(), "d")
    test = get_data("01.01.2019", "31.12.2019", cols, os.getcwd(), "d")
    y = train[cols[0]]
    x = train[cols[1]]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    test_mean = test[cols[0]].mean()
    mean_error = test_mean - model.params["const"]
    forecast = []
    for i in range(len(test)):
        hydro_dev = test.loc[i, cols[1]]
        x = model.get_prediction(exog=(1, hydro_dev)).predicted_mean[0]
        forecast.append(x)
    forecast_df = pd.DataFrame({"System Price": test[cols[0]], "Forecast": forecast})
    mae_orig = get_all_point_metrics(forecast_df)["mae"]
    ad_forecast = [i + mean_error for i in forecast]
    ad_forecast_df = pd.DataFrame({"System Price": test[cols[0]], "Forecast": ad_forecast})
    mae_ad = get_all_point_metrics(ad_forecast_df)["mae"]
    plt.subplots(figsize=full_fig)
    plt.plot(test["Date"], forecast, label="Forecast (mae={:.2f})".format(mae_orig), color=sec_color)
    plt.plot(test["Date"], ad_forecast, label="Adj. Forecast (mae={:.2f})".format(mae_ad), color=third_color)
    plt.plot(test["Date"], test["System Price"], label="True", color=first_color)
    plt.title("Linear model trained on Total Hydro Dev", pad=title_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.ylabel("Daily price [€]", labelpad=label_pad)
    for line in plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("output/plots/eda/lin_model_total_hydro_dev.png")
    plt.close()


def lin_model_test_norm_per_year():
    df = get_data("01.01.2014", "31.12.2019", ["System Price", "Total Hydro Dev"], os.getcwd(), "d")
    df["Norm Price"] = np.NAN
    for year in range(2014, 2020):
        df_year = df[df["Date"].dt.year.isin([year])]
        df_year["Norm Price"] = (df_year["System Price"] - df_year["System Price"].mean())
        df.loc[df_year.index, "Norm Price"] = df_year["Norm Price"]
    test = df[df["Date"].dt.year.isin([2019])].reset_index()
    train = df[df["Date"].dt.year.isin(range(2014, 2019))].reset_index()
    y = train["Norm Price"]
    x = train["Total Hydro Dev"]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    forecast = []
    for i in range(len(test)):
        hydro_dev = test.loc[i, "Total Hydro Dev"]
        x = model.get_prediction(exog=(1, hydro_dev)).predicted_mean[0]
        forecast.append(x)
    forecast_df = pd.DataFrame({"System Price": test["Norm Price"], "Forecast": forecast})
    mae = get_all_point_metrics(forecast_df)["mae"]
    plt.subplots(figsize=full_fig)
    plt.plot(test["Date"], forecast, label="Forecast (mae={:.2f})".format(mae), color=sec_color)
    plt.plot(test["Date"], test["Norm Price"], label="True", color=first_color)
    plt.title("Linear model trained on Total Hydro Dev and Norm. Price", pad=title_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.ylabel("Daily norm. price [€]", labelpad=label_pad)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("output/plots/eda/lin_model_total_hydro_dev_norm_price.png")
    plt.close()


def plot_autocorrelation():
    from statsmodels.graphics.tsaplots import plot_acf
    data = get_data("01.01.2014", "31.12.2019", ["System Price"], os.getcwd(), "d")["System Price"]
    fig, ax = plt.subplots(figsize=full_fig)
    plot_acf(data, lags=21, ax=ax, label="c", vlines_kwargs={"colors": "black", "label": "h"})
    plt.title("Autocorrelation System Price (2014-2019)", pad=title_pad)
    plt.ylabel("Pearson's coefficient", labelpad=label_pad)
    plt.xlabel("Day lag", labelpad=label_pad)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["95% confidence boundary", "Correlation"]
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True, handles=handles, labels=labels).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("output/plots/eda/autocorr_price.png")
    plt.close()


def explore_eikon_data():
    columns = ["Oil", "Gas", "Coal", "Low Carbon", "System Price"]
    df = get_data("01.01.2014", "31.12.2019", columns, os.getcwd(), "d")
    df = df[["Date"] + columns]
    plt.subplots(figsize=full_fig)
    for col in df.columns:
        if col != "Date":
            plt.plot(df["Date"], df[col], label=col)
    for line in plt.legend(loc='upper center', ncol=len(columns), bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Fossil Fuel Commodity Prices 2014-2019", pad=title_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.tight_layout()
    plt.savefig("output/plots/eda/fossil_prices.png")
    plt.close()


def auction_data():
    raw_folder = "input/auction/raw"
    vol_price_df = get_data("01.07.2014", "31.12.2020", ["Total Vol", "System Price"], os.getcwd(), "h")
    net_flow_df = pd.DataFrame(
        columns=["Date", "Hour", "Acs demand", "Acs supply", "Net flow", "Total Volume", "Int Volume"])
    vol_price_df["Full vol"] = np.nan
    vol_price_df["Full price"] = np.nan
    vol_price_df["Disc vol"] = np.nan
    vol_price_df["Disc price"] = np.nan
    for file in sorted(Path(raw_folder).iterdir()):  # list all raw xls files
        date_string = str(file).split("\\")[3].split("_")[-1][:-4]
        date = dt.strptime(date_string, "%Y-%m-%d").date()
        print(date)
        vol_price_df_date = vol_price_df[vol_price_df["Date"].dt.date == date]
        if not os.path.exists('output/plots/all_bids/{}'.format(date)):
            os.makedirs('output/plots/all_bids/{}'.format(date))
        df_result = pd.DataFrame()
        lower_lim = -10
        upper_lim = 210
        df_result["Price"] = range(lower_lim, upper_lim + 1)
        df_raw = pd.read_excel(file)
        df_raw.columns = [str(i) for i in df_raw.columns]
        all_columns = [i for i in df_raw.columns]
        all_hour_columns = [i for i in all_columns if "Bid curve chart data (Reference time)" not in i]
        for i in range(len(all_hour_columns)):
            cols = [i * 2, i * 2 + 1]
            df_h = df_raw[df_raw.columns[cols]]
            hour = int(df_h.columns[1].split(" ")[1][0:2])
            print("Hour {}".format(hour))
            row_index = vol_price_df.loc[(vol_price_df["Date"].dt.date == date) & (vol_price_df["Hour"] == hour)].index[
                0]
            exc_volume = vol_price_df_date[vol_price_df_date["Hour"] == hour]["Total Vol"].tolist()[0]
            exc_price = vol_price_df_date[vol_price_df_date["Hour"] == hour]["System Price"].tolist()[0]
            idx_vol_demand = \
                df_h[df_h[df_h.columns[0]] == "Bid curve chart data (Volume for accepted blocks buy)"].index[0]
            idx_vol_supply = \
                df_h[df_h[df_h.columns[0]] == "Bid curve chart data (Volume for accepted blocks sell)"].index[0]
            idx_net_flow = df_h[df_h[df_h.columns[0]] == "Bid curve chart data (Volume for net flows)"].index[0]
            acs_demand = df_h.loc[idx_vol_demand, df_h.columns[1]]
            acs_supply = df_h.loc[idx_vol_supply, df_h.columns[1]]
            net_flows = df_h.loc[idx_net_flow, df_h.columns[1]]
            add_flow_to_demand = net_flows < 0
            add_flow_to_supply = net_flows > 0
            idx_bid_start = df_h[df_h[df_h.columns[0]] == "Buy curve"].index[0] + 1
            df_h = df_h.iloc[idx_bid_start:].reset_index(drop=True)
            df_h = df_h.rename(columns={df_h.columns[0]: 'Category', df_h.columns[1]: 'Value'})
            df_h = df_h.dropna(how='all')
            index_sell = df_h[df_h["Category"] == "Sell curve"].index[0]
            df_buy = df_h[0:index_sell]
            df_d = pd.DataFrame(columns=["Price", "Volume"])
            df_d["Price"] = df_buy[df_buy["Category"] == "Price value"]["Value"].tolist()
            df_d["Volume"] = df_buy[df_buy["Category"] == "Volume value"]["Value"].tolist()
            df_d = df_d[lower_lim < df_d["Price"]]
            df_d = df_d[upper_lim > df_d["Price"]].reset_index(drop=True)
            df_d["Volume"] = df_d["Volume"] + acs_demand
            if add_flow_to_demand:
                df_d["Volume"] = df_d["Volume"] + abs(net_flows)
            demand_line_full = LineString(np.column_stack((df_d["Volume"], df_d["Price"])))
            df_d = get_discrete_bid_df(df_d, lower_lim, upper_lim)
            demand_line_discrete = LineString(np.column_stack((df_d["Volume"], df_d["Price"])))
            plt.subplots(figsize=full_fig)
            plt.plot(df_d["Volume"], df_d["Price"], label="Demand", zorder=1)
            df_sell = df_h[index_sell + 1:]
            df_s = pd.DataFrame(columns=["Price", "Volume"])
            df_s["Price"] = df_sell[df_sell["Category"] == "Price value"]["Value"].tolist()
            df_s["Volume"] = df_sell[df_sell["Category"] == "Volume value"]["Value"].tolist()
            df_s = df_s[lower_lim < df_s["Price"]]
            df_s = df_s[upper_lim > df_s["Price"]].reset_index(drop=True)
            df_s["Volume"] = df_s["Volume"] + acs_supply
            if add_flow_to_supply:
                df_s["Volume"] = df_s["Volume"] + abs(net_flows)
            supply_line_full = LineString(np.column_stack((df_s["Volume"], df_s["Price"])))
            df_s = get_discrete_bid_df(df_s, lower_lim, upper_lim)
            demand_col_name = "Demand h {}".format(hour)
            if demand_col_name in df_result.columns:
                demand_col_name = demand_col_name + "_2"
            supply_col_name = "Supply h {}".format(hour)
            if supply_col_name in df_result.columns:
                supply_col_name = demand_col_name + "_2"
            df_result[demand_col_name] = df_d["Volume"]
            df_result[supply_col_name] = df_s["Volume"]
            supply_line_discrete = LineString(np.column_stack((df_s["Volume"], df_s["Price"])))
            plt.plot(df_s["Volume"], df_s["Price"], label="Supply", zorder=2)
            full_intersect = supply_line_full.intersection(demand_line_full)
            full_vol = full_intersect.x
            full_price = full_intersect.y
            vol_price_df.loc[row_index, "Full vol"] = full_vol
            vol_price_df.loc[row_index, "Full price"] = full_price
            discrete_intersect = supply_line_discrete.intersection(demand_line_discrete)
            if type(discrete_intersect) == LineString:
                disc_price = upper_lim
                disc_vol = max(df_d["Volume"])
                print("In date {} hour {} we found no intersection".format(date, hour))
            else:
                disc_vol = discrete_intersect.x
                disc_price = discrete_intersect.y
            vol_price_df.loc[row_index, "Disc vol"] = disc_vol
            vol_price_df.loc[row_index, "Disc price"] = disc_price
            plt.scatter(disc_vol, disc_price, color="red", zorder=3, s=140,
                        label="Disc. intersect ({:.2f}, {:.2f})".format(disc_vol, disc_price))
            plt.scatter(exc_volume, exc_price, color="green", zorder=4,
                        label="True intersect ({:.2f}, {:.2f})".format(exc_volume, exc_price))
            plt.legend()
            plt.ylim(lower_lim, max(100, disc_price * 1.1))
            plt.ylabel("Price")
            plt.xlabel("Volume")
            plt.title("Discrete Bid Curves {} - {}".format(date, hour))
            plt.tight_layout()
            save_curve_path = 'output/plots/all_bids/{}/{}.png'.format(date, hour)
            if os.path.exists(save_curve_path):
                plt.savefig('output/plots/all_bids/{}/{}_2'.format(date, hour))
            else:
                plt.savefig('output/plots/all_bids/{}/{}'.format(date, hour))
            plt.close()
            flow_dict = {"Date": date, "Hour": hour, "Acs demand": acs_demand, "Acs supply": acs_supply,
                         "Net flow": net_flows, "Total Volume": exc_volume, "Int Volume": disc_vol}
            net_flow_df = net_flow_df.append(flow_dict, ignore_index=True)
        cols = [i for i in df_result.columns if "Demand" in i or "Supply" in i]
        df_result[cols] = df_result[cols].astype(float).round(2)
        save_disc_path = "input/auction/csv_disc/{}.csv".format(date)
        df_result.to_csv(save_disc_path, index=False, float_format='%.2f')
    vol_price_df = vol_price_df.dropna()
    vol_price_df.to_csv("output/auction/vol_price_auction.csv", index=False, float_format='%.2f')
    net_flow_df.to_csv("output/auction/volume_analyses.csv", index=False, float_format='%.2f')


def get_discrete_bid_df(df, lower, upper):
    price_range = range(lower, upper + 1)
    result = pd.DataFrame(columns=df.columns)
    result["Price"] = price_range
    for i in range(len(result)):
        closest = df.iloc[(df["Price"] - price_range[i]).abs().argsort()[:1]].reset_index(drop=True).loc[0, "Volume"]
        result.loc[i, "Volume"] = closest
    return result





def rename_folders_from_raw():
    raw_folder = "input/auction/raw_s"
    # dates = [(dt(2015, 1, 1) + timedelta(days=x)).date() for x in range(365)]
    all_files = sorted(Path(raw_folder).iterdir())
    for file in all_files:  # list all raw xls files
        date_string = str(file).split("_")[-3][:-3]
        date = dt.strptime(date_string, "%d-%m-%Y").date()
        print(date)
        # dates.remove(date)
        first_path = "\\".join(str(file).split("\\")[0:3])
        mcp_name = "_".join(str(file).split("\\")[3].split("_")[0:3])
        save_path = first_path + "\\" + mcp_name + "_" + str(date) + ".xls"
        os.rename(file, save_path)
    # print(dates)




def min_max_price():
    price = get_data("01.01.2014", "31.12.2020", ["System Price"], os.getcwd(), "h")["System Price"].tolist()
    max_price = max(price)
    min_price = min(price)
    print("Max {}, min {}".format(max_price, min_price))


def eda_disc_auction_data(overview, make_analysis_csv):
    if overview:
        all_dates = [i.strftime("%Y-%m-%d") + ".csv" for i in
                     pd.date_range(dt(2014, 7, 1), dt(2021, 1, 1) - timedelta(days=1), freq='d')]
        all_files = sorted(Path("input/auction/csv_disc").iterdir())
        all_end_files = [str(i).split("\\")[-1] for i in all_files]
        print("--------------------------------\n")
        print("There are {} csv-files in folder ranging {} dates".format(len(all_end_files), len(all_dates)))
        for d in all_dates:
            if d not in all_end_files:
                print("Missing csv file for {}".format(d))
        number_of_auctions = len(all_dates) * 24 + 1  # adding one as 2014 has +1 hour but no -1 hour
        print("Number of auctions during period: {}".format(number_of_auctions))
        prices_vol = get_data("01.07.2014", "31.12.2020", ["System Price", "Total Vol"], os.getcwd(), "h")
        max_p = prices_vol[prices_vol["System Price"] == prices_vol["System Price"].max()].iloc[0]
        print("Max price: {} h{}:\t {} (volume {})".format(max_p["Date"].date(), max_p["Hour"], max_p["System Price"],
                                                           max_p["Total Vol"]))
        min_p = prices_vol[prices_vol["System Price"] == prices_vol["System Price"].min()].iloc[0]
        print("Min price: {} h{}:\t {} (volume {})".format(min_p["Date"].date(), min_p["Hour"], min_p["System Price"],
                                                           min_p["Total Vol"]))
        max_v = prices_vol[prices_vol["Total Vol"] == prices_vol["Total Vol"].max()].iloc[0]
        print("Max vol: {} h{}:\t {} (price {})".format(max_v["Date"].date(), max_v["Hour"], max_v["Total Vol"],
                                                        max_v["System Price"]))
        min_v = prices_vol[prices_vol["Total Vol"] == prices_vol["Total Vol"].min()].iloc[0]
        print("Min vol: {} h{}:\t {} (price {})".format(min_v["Date"].date(), min_v["Hour"], min_v["Total Vol"],
                                                        min_v["System Price"]))
        print("--------------------------------\n")
    if make_analysis_csv:
        years = range(2014, 2021)
        all_files = sorted(Path("input/auction/csv_disc").iterdir())
        true_df = get_data("01.07.2014", "31.12.2020", ["System Price", "Total Vol"], os.getcwd(), "h")
        df = pd.DataFrame(columns=["Date", "Hour", "True price", "True vol", "Disc price", "Disc vol"])
        for f in all_files:
            date = dt.strptime(str(f).split("\\")[-1][:-4], "%Y-%m-%d").date()
            print(date)
            data = pd.read_csv(f)
            for i in range(int((len(data.columns) - 1) / 2)):
                demand = data[["Price"]+[data.columns[1+i*2]]]
                hour = demand.columns[1].split(" ")[-1]
                if "_" not in hour:
                    row = true_df[(true_df["Date"].dt.date==date) & (true_df["Hour"]==int(hour))].reset_index(drop=True)
                    demand_line = LineString(np.column_stack((demand["Price"], demand[demand.columns[1]])))
                    supply = data[["Price"]+[data.columns[1+i*2+1]]]
                    supply_line = LineString(np.column_stack((supply["Price"], supply[supply.columns[1]])))
                    intersect = supply_line.intersection(demand_line)
                    new = {"Date": date, "Hour": hour, "True price": row.loc[0, "System Price"], "True vol": row.loc[0, "Total Vol"],
                           "Disc price": round(intersect.x, 4), "Disc vol": round(intersect.y, 2)}
                    df = df.append(new, ignore_index=True)
                    df.to_csv("output/auction/vol_price_disc_analysis.csv", index=False)
    df = pd.read_csv("output/auction/vol_price_disc_analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Mae price"] = abs(df["True price"] - df["Disc price"])
    df["Mape price"] = 100 * df["Mae price"] / df["True price"]
    df["Mae vol"] = abs(df["True vol"] - df["Disc vol"])
    df["Mape vol"] = 100 * df["Mae vol"] / df["True vol"]
    df_wrong_2020 = df[df["Date"] >= dt(2020, 6, 3)]
    df = df[df["Date"] < dt(2020, 6, 3)]
    print("MAE price discrete bids: {:.3f}".format(df["Mae price"].mean()))
    print("MAPE price discrete bids: {:.3f}%".format(df["Mape price"].mean()))
    print("MAE volume discrete bids: {:.3f}".format(df["Mae vol"].mean()))
    print("MAPE volume discrete bids: {:.3f}\n".format(df["Mape vol"].mean()))
    df_grouped = df.groupby(by=df["Date"].dt.year).mean()[["Mae price", "Mape price", "Mae vol", "Mape vol"]]
    # print(df_grouped)
    print("MAE price discrete bids wrong 2020: {:.3f}".format(df_wrong_2020["Mae price"].mean()))
    print("MAE volume discrete bids wrong 2020: {:.3f}\n".format(df_wrong_2020["Mae vol"].mean()))


def investigate_unit_price_information_loss():
    df_2014 = pd.read_csv("output/vol_price_auction_2014.csv")
    df_2019 = pd.read_csv("output/vol_price_auction_2019.csv")
    df_2020 = pd.read_csv("output/vol_price_auction_2020.csv")
    for year, df in {2014: df_2014, 2019: df_2019, 2020: df_2020}.items():
        print("\n{} ----------------------".format(year))
        df["full_vol_dev"] = 100 * (df["Total Vol"] - df["Full vol"]) / df["Total Vol"]
        print("Full: Mean error in volume {:.3f}%".format(df["full_vol_dev"].mean()))
        print("Full: Abs mean error in volume {:.3f}%".format(abs(df["full_vol_dev"]).mean()))
        df["disc_vol_dev"] = 100 * (df["Total Vol"] - df["Disc vol"]) / df["Total Vol"]
        print("Disc: Mean error in volume {:.3f}%".format(df["disc_vol_dev"].mean()))
        print("Disc: Abs mean error in volume {:.3f}%".format(abs(df["disc_vol_dev"]).mean()))
        df["full_price_dev"] = 100 * (df["System Price"] - df["Full price"]) / df["System Price"]
        print("Full: Mean error in price {:.3f}%".format(df["full_price_dev"].mean()))
        print("Full: Abs mean error in price {:.3f}%".format(abs(df["full_price_dev"]).mean()))
        df["disc_price_dev"] = 100 * (df["System Price"] - df["Disc price"]) / df["System Price"]
        print("Disc: Mean error in price {:.3f}%".format(df["disc_price_dev"].mean()))
        print("Disc: Abs mean error in price {:.3f}%".format(abs(df["disc_price_dev"]).mean()))




if __name__ == '__main__':
    print("Running eda..")
    # plot_norm_weekday()
    # plot_norm_month()
    # plot_daily_vs_hourly_prices()
    # plot_norm_week()
    # plot_temperatures()
    # plot_precipitation()
    # plot_all_variables_per_year()
    # plot_correlation("System Price", "Coal")
    # plot_correlation_norm_price_per_year("System Price", "Total Hydro Dev")
    # check_lagged_correlation("System Price", "Total Hydro Dev")
    # lin_model_test()
    # lin_model_test_norm_per_year()
    # plot_autocorrelation()
    # explore_eikon_data()
    # auction_data()
    # min_max_price()
    # rename_folders_from_raw()
    # auction_data()
    # merge_same_dates_in_one_csv()
    # plot_mean_curves()
    eda_disc_auction_data(overview=False, make_analysis_csv=False)
    # investigate_unit_price_information_loss()

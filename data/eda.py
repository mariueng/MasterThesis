import os
import warnings
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from data.data_handler import get_data
from data.data_handler import get_auction_data
from scipy import stats
from src.system.scores import get_all_point_metrics
from data.bid_curves import get_original_bid_methods
import random

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


def plot_random_auctions(n):
    random.seed(1)
    all_hours = [i for i in pd.date_range(dt(2014, 7, 1), dt(2020, 6, 3), freq='h')]
    chosen_auctions = random.sample(all_hours, n)
    for auction in chosen_auctions:
        print(auction)
        date_string = dt.strftime(auction, "%Y-%m-%d")
        hour = str(auction.hour)
        true_demand, true_supply = get_true_volumes(date_string, hour)
        date_string_2 = dt.strftime(auction, "%d.%m.%Y")
        demand_day = get_auction_data(date_string_2, date_string_2, ["d"], os.getcwd())
        est_demand = demand_day[demand_day["Hour"] == int(hour)].T[2:]
        est_demand["Price"] = [int(i[2:]) for i in est_demand.index]
        est_demand = est_demand.rename(columns={est_demand.columns[0]: "Est. demand"})
        supply_day = get_auction_data(date_string_2, date_string_2, ["s"], os.getcwd())
        est_supply = supply_day[supply_day["Hour"] == int(hour)].T[2:]
        est_supply["Price"] = [int(i[2:]) for i in est_supply.index]
        col_1 = plt.get_cmap("tab10")(0)
        col_2 = plt.get_cmap("tab10")(1)
        est_supply = est_supply.rename(columns={est_supply.columns[0]: "Est. supply"})
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(full_fig[0]+8, full_fig[1]))
        fig.suptitle("Supply and Demand from Price Classes {}, h {}".format(date_string, hour))
        ax1.plot(true_demand["True demand"], true_demand["Price"], linestyle="dotted", color=col_1,
                 label="True demand", linewidth=2)
        ax1.plot(est_demand["Est. demand"], est_demand["Price"], color=col_1, label="Est. demand")
        ax2.plot(true_supply["True supply"], true_supply["Price"], linestyle="dotted", color=col_2,
                 label="True supply", linewidth=2)
        ax2.plot(est_supply["Est. supply"], est_supply["Price"], color=col_2, label="Est. supply")
        ax3.plot(est_demand["Est. demand"], est_demand["Price"], color=col_1, label="Est. demand")
        ax3.plot(est_supply["Est. supply"], est_supply["Price"], color=col_2, label="Est. supply")
        for ax in [ax1, ax2, ax3]:
            ax.set_ylabel("Price [€]", labelpad=label_pad)
            ax.set_xlabel("Volume [MWh]", labelpad=label_pad)
            for line in ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                                  shadow=True).get_lines():
                line.set_linewidth(2)
        plt.tight_layout()
        plt.savefig("output/auction/price_classes/random_auctions/{}_{}.png".format(date_string, hour))
        plt.close()


def get_true_volumes(date_string, hour):  # Helping method
    columns = [int(hour) * 2, int(hour) * 2 + 1]
    path_to_orig = "input/auction/raw/mcp_data_report_{}.xls".format(date_string)
    df = pd.read_excel(path_to_orig, usecols=columns)
    df = df.rename(columns={df.columns[0]: 'Category', df.columns[1]: 'Value'})
    df = df.dropna(how='all')
    get_initial_info_raw, get_buy_and_sell_dfs, get_auction_df = get_original_bid_methods()
    idx_demand, idx_supply, idx_flow, acs_demand, acs_supply, flow, idx_bid = get_initial_info_raw(df)
    add_flow_demand = flow < 0
    add_flow_supply = flow > 0
    lower_lim = -10
    upper_lim = 210
    df_buy, df_sell = get_buy_and_sell_dfs(df, idx_bid)
    df_buy = get_auction_df(df_buy, add_flow_demand, lower_lim, upper_lim, acs_demand, flow)
    df_sell = get_auction_df(df_sell, add_flow_supply, lower_lim, upper_lim, acs_supply, flow)
    df_buy = df_buy.rename(columns={"Volume": "True demand"})
    df_sell = df_sell.rename(columns={"Volume": "True supply"})
    return df_buy, df_sell


def eda_auction_data():
    hourly_errors = pd.read_csv("input/auction/time_series_errors.csv")
    mae_demand = hourly_errors["MAE d"].mean()
    mae_supply = hourly_errors["MAE s"].mean()
    max_mae_dem = hourly_errors["MAE d"].max()
    max_mae_sup = hourly_errors["MAE s"].max()
    print("For all hours in dataset:\tMAE demand = {:.2f}, MAE supply = {:.2f}".format(mae_demand, mae_supply))
    print("For all hours in dataset:\tMax MAE dem = {:.2f}, Max MAE sup = {:.2f}".format(max_mae_dem, max_mae_sup))


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
    # plot_random_auctions(10)
    eda_auction_data()


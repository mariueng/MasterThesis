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
random.seed(1)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import acf
from statsmodels.tsa.seasonal import seasonal_decompose


label_pad = 12
title_pad = 20
full_fig = (13, 7)
half_fig = (6.5, 7)
first_color = "steelblue"
sec_color = "firebrick"
third_color = "darkorange"
fourth_color = "mediumseagreen"
fifth_color = "silver"
sixth_color = "palevioletred"
seventh_color = "teal"
seven_colors = [first_color, sec_color, third_color, fourth_color, fifth_color, sixth_color, seventh_color]


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
    df = get_data("01.01.2019", "31.12.2019", ["Prec Norway", "Prec Norway 7"], os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["Prec Norway"], label="Acc. precipitation Norway", color=first_color)
    plt.plot(df["Date"], df["Prec Norway 7"], label="Acc. precipitation 7 days ahead Norway", color=sec_color)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Precipitation Norway 2019", pad=title_pad)
    plt.ylabel("Mm", labelpad=label_pad)
    plt.xlabel("Date", labelpad=label_pad)
    # ymax = df["Prec Norway 7"].max() * 1.03
    ymax = 1400
    plt.ylim(0, ymax)
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
    title = "Wind Production Sweden and Denmark per Year"
    col_name = "Wind Prod"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Mm"
    title = "Precipitation Norway"
    col_name = "Prec Norway"
    plot_col_per_year(col_name, title, ylabel)
    ylabel = "Mm"
    title = "Precipitation Norway Seven Days Ahead"
    col_name = "Prec Norway 7"
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
    data = get_data("01.01.2014", "31.12.2019", ["System Price"], os.getcwd(), "d")["System Price"]
    fig, ax = plt.subplots(figsize=full_fig)
    plot_acf(data, lags=21, ax=ax, label="c", vlines_kwargs={"colors": "black", "label": "h"})
    plt.title("Autocorrelation System Price (2014-2019)", pad=title_pad)
    plt.ylabel("Pearson's coefficient", labelpad=label_pad, size=11)
    plt.xlabel("Day lag", labelpad=label_pad, size=11)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["95% confidence boundary", "Correlation"]
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True, handles=handles, labels=labels).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.show()
    assert False
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


def plot_european_prices():
    data = get_data("01.07.2014", "02.06.2019", ["System Price", "APX", "OMEL", "EEX"], os.getcwd(), "d")
    # data = get_data("01.01.2019", "31.12.2019", ["System Price", "APX", "OMEL", "EEX", "Week", "Weekday"], os.getcwd(), "d")
    data = data.rename(columns={"System Price": "Nord Pool"})
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data = data.drop(columns=["Date"])
    data = data.groupby(by=["Year", "Month"]).mean()
    #grouped = data[["APX", "OMEL", "EEX", "Week"]].groupby(by="Week").mean()
    #for market in ["APX", "OMEL", "EEX"]:
        #m_df = grouped[market]
        #for week in grouped.index:
            #mean = m_df.loc[week]
            #data_mask = data[data["Week"] == week]
            #data.loc[data_mask.index, market] = mean
    #for market in ["APX", "OMEL", "EEX"]:
        #data[market] = data.apply(lambda row: row[market] if row["Weekday"] == 4 else np.NaN, axis=1)
    plt.subplots(figsize=(13, 5))
    colors = [sec_color, third_color, fourth_color, first_color]
    markets = ["EEX", "APX", "OMEL", "Nord Pool"]
    #data = data[["Date", "Nord Pool", "EEX", "APX", "OMEL"]]
    #data = data.interpolate()
    for i in range(len(markets)):
        col = markets[i]
        avg_price = data[col].mean()
        plt.plot(range(len(data)), data[col], label="{} (€{:.1f})".format(col, avg_price), color=colors[i])
    plt.ylim(0, 80)
    for line in plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                          shadow=True, prop={"size":11}).get_lines():
        line.set_linewidth(2)
    plt.title("European and Nord Pool Weekly Mean System Prices - Training Data", pad=title_pad, size=14)
    plt.xlabel("Year", labelpad=label_pad, size=11)
    plt.ylabel("Price [€]", labelpad=label_pad, size=11)
    plt.xticks([i + 6 for i in [0, 12, 24, 36, 48]], ["2015", "2016", "2017", "2018", "2019"])
    plt.tight_layout()
    plt.savefig("output/plots/eda/european_market_prices_2.png")
    plt.close()


def plot_nordpool_prices():
    from matplotlib.dates import DateFormatter
    data = get_data("01.07.2014", "31.12.2019", ["System Price"], os.getcwd(), "d")
    h_data = get_data("01.07.2014", "31.12.2019", ["System Price"], os.getcwd(), "h")
    d_range_2014 = pd.date_range(dt(2014,1,1), dt(2014,6,30), freq="d")
    h_range_2014 = pd.date_range(dt(2014,1,1), dt(2014,6,30), freq="h")
    d_pd = pd.DataFrame(columns=data.columns)
    d_pd["Date"] = d_range_2014
    h_pd = pd.DataFrame(columns=h_data.columns)
    h_pd["Date"] = [d.date() for d in h_range_2014]
    h_pd["Hour"] = [d.hour for d in h_range_2014]
    for df in [data, h_data]:
        test = df[df["Date"]>dt(2019,6,2)]
        df.loc[test.index, "System Price"] = np.NAN
    data = d_pd.append(data, ignore_index=True)
    h_data = h_pd.append(h_data, ignore_index=True)
    h_data["Hour"] = pd.to_datetime(h_data['Hour'], format="%H").dt.time
    h_data["DateTime"] = h_data.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(13, 9))
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    years = [i for i in range(2014, 2020)]
    for i in range(len(years)):
        ax = axs[i]
        h_df = h_data[h_data["DateTime"].dt.year == years[i]]
        df = data[data["Date"].dt.year == years[i]].reset_index(drop=True)
        ax.plot(h_df["DateTime"], h_df["System Price"], color=first_color, lw=0.5, label="Hour")
        ax.plot(df["Date"], df["System Price"], color=sec_color, lw=2, label="Day")
        ax.set_xlim(df.loc[0, "Date"], df.loc[len(df)-1, "Date"])
        a = ax.twinx()
        a.set_ylabel("{}".format(years[i]), size=12, labelpad=8)
        a.set_yticks([])
        if i == 0:
            for line in ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), fancybox=True,
                                   shadow=True, prop={"size": 11}).get_lines():
                line.set_linewidth(2)
            a.spines['top'].set_visible(False)
        if i != 5:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:
            date_form = DateFormatter("%b")
            ax.xaxis.set_major_formatter(date_form)
        ax.set_ylim(5, 70)
        y_low, y_high = (df["System Price"].min(), df["System Price"].max())
        #span = y_high-y_low
        y_pos = 30
        # ax.text(x=df.loc[183, "Date"], y=y_pos, s="{}".format(years[i]), size=14)
        #a.text(y=y_pos, s="{}".format(years[i]), size=13, rotation=90)
    fig.text(x=0.01, y=0.48, s="Price [€]", size=12, rotation=90)
    fig.suptitle("Nord Pool System Price - Training Data", size=14, x=0.52)
    plt.xlabel("Time", labelpad=label_pad, size=11)
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.subplots_adjust(left=0.05)
    plt.savefig("output/plots/eda/nordpool_all_train.png")
    plt.close()


def plot_spring_2019():
    data = get_data("01.03.2019", "30.06.2019", ["System Price", "Total Hydro Dev", "Wind Prod", "Prec Norway"], os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    plt.plot(data["Date"], data["System Price"])
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(data["Date"], data["Total Hydro Dev"])
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(data["Date"], data["Prec Norway"])
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(data["Date"], data["Wind Prod"])
    plt.show()
    plt.close()


def check_demand_stationarity():
    # H0: Suggests the time series has a unit root, meaning it is non-stationary.
    # H1: Null-hyp is rejected. Suggests the time series does not have a unit root, meaning it is stationary.
    # p-value <= 0.05: Reject the null hypothesis
    from statsmodels.tsa.stattools import adfuller
    demand_col = "Curve Demand"
    data = get_data("01.07.2014", "02.06.2020", [demand_col], os.getcwd(), "d")
    x = data[demand_col].values
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def explore_holyday():
    df = get_data("20.12.2019", "3.01.2020", ["Curve Demand", "System Price"], os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["System Price"])
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["Curve Demand"])
    plt.show()
    plt.close()


def explore_snowfall():
    df = get_data("01.01.2014", "31.12.2019", ["Temp Norway"], os.getcwd(), "d")
    ninja = pd.read_csv("input/ninja/snow_temp_norway_2014_2019.csv")
    ninja["Date"] = pd.to_datetime(ninja["Date"], format="%Y-%m-%d")
    ninja = ninja[["Date", "temperature", "snow_mass"]]
    df = df.merge(ninja, on="Date")
    test_temperature = True
    if test_temperature:
        for year in range(2014, 2020):
            sub = df[df["Date"].dt.year == year]
            plt.subplots(figsize=full_fig)
            plt.plot(sub["Date"], sub["Temp Norway"], label="old")
            plt.plot(sub["Date"], sub["temperature"], label="new")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
    test_snowmass = True
    if test_snowmass:
        for year in range(2014, 2020):
            sub = df[df["Date"].dt.year == year]
            plt.subplots(figsize=full_fig)
            plt.plot(sub["Date"], sub["snow_mass"], label="snowmass")
            plt.title("Snowmass {}".format(year))
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()


def explore_coal_and_wv():
    d = pd.read_csv("output/auction/water_values.csv", usecols=["Date", "Water Value"])
    d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m-%d")
    df = get_data("07.07.2014", "02.06.2019", ["Week", "Coal"], os.getcwd(), "d")
    df = df.merge(d, on="Date")
    r_coeff = round(stats.pearsonr(df["Coal"], df["Water Value"])[0], 3)
    print("R coal  and wv : {}".format(r_coeff))
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["Water Value"], label="water value")
    plt.plot(df["Date"], df["Coal"], label="coal price")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    for i in range(len(df) // 7):
        df.loc[i*7:i*7+7, "Week"] = i + 1
    df = df.groupby(by="Week").mean().reset_index()
    df["Coal Diff 1"] = df["Coal"] - df["Coal"].shift(1)
    df["WV Diff 1"] = df["Water Value"] - df["Water Value"].shift(1)
    df["Coal Diff 2"] = df["Coal Diff 1"].shift(1)
    df = df.dropna()
    r_coeff = round(stats.pearsonr(df["Coal Diff 1"], df["WV Diff 1"])[0], 3)
    print("R coal diff 1 and wv diff 1: {}".format(r_coeff))
    plt.subplots(figsize=full_fig)
    plt.plot(df["Week"], df["WV Diff 1"], label="wv diff 1")
    plt.plot(df["Week"], df["Coal Diff 1"], label="coal diff 1")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    r_coeff = round(stats.pearsonr(df["Coal"], df["Water Value"])[0], 3)
    print("R coal  and wv week: {}".format(r_coeff))
    plt.subplots(figsize=full_fig)
    plt.plot(df["Week"], df["Water Value"], label="water value")
    plt.plot(df["Week"], df["Coal"], label="coal price")
    plt.xlabel("Week")
    plt.legend()
    plt.tight_layout()
    plt.show()


def explore_hydro_and_coal_multiple_regression():
    warnings.filterwarnings("ignore")
    from sklearn.svm import SVR
    d = pd.read_csv("output/auction/water_values.csv", usecols=["Date", "Water Value"])
    d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m-%d")
    df = get_data("01.07.2014", "02.06.2019", ["Weekday", "Season", "Coal", "Total Hydro Dev"], os.getcwd(), "d")
    df = df.merge(d, on="Date")
    all_dates = [i.date() for i in pd.date_range(dt(2014, 8, 1), dt(2019, 5, 19), freq="d")]
    chosen_dates = random.sample(all_dates, 50)
    run_svr = True
    if run_svr:
        w = 30
        maes = []
        for date in chosen_dates:
            first_d = date - timedelta(days=w)
            end_d = date + timedelta(days=13)
            data = df[(df["Date"].dt.date >= first_d) & (df["Date"].dt.date <= end_d)]
            data["Neg Dev"] = data.apply(lambda row: 1 if row["Total Hydro Dev"] < 0 else 0, axis=1)
            for col in ["Coal", "Total Hydro Dev"]:
                data["Log {}".format(col)] = np.log(abs(data[col]))
            data["Log Total Hydro Dev"] = data.apply(lambda row: - row["Log Total Hydro Dev"] if row["Neg Dev"] == 1 else
                                                   row["Log Total Hydro Dev"], axis=1)
            data = data[["Date", "Log Coal", "Log Total Hydro Dev", "Water Value"]]
            training = data.head(w)
            testing = data.tail(14)
            model = SVR(kernel='poly')
            x_rows = ["Log Coal", "Log Total Hydro Dev"]
            model.fit(training[x_rows].values, training[["Water Value"]].values.ravel())
            y_pred = model.predict(testing[x_rows].values)
            maes.append(abs(testing["Water Value"].values - y_pred).mean())
        print("MAE svr = {}".format(sum(maes) / len(maes)))
    run_naive = True
    if run_naive:
        maes = []
        for date in chosen_dates:
            first_d = date - timedelta(days=7)
            end_d = date + timedelta(days=13)
            data = df[(df["Date"].dt.date >= first_d) & (df["Date"].dt.date <= end_d)]
            training = data.head(7)
            testing = data.tail(14).copy()
            for index in testing.index:
                weekday = testing.loc[index, "Weekday"]
                train_wv = training[training["Weekday"] == weekday].head(1)["Water Value"].values[0]
                testing.loc[index, "Pred"] = train_wv
            maes.append(abs(testing["Water Value"] - testing["Pred"]).mean())
        print("MAE naive = {}".format(sum(maes) / len(maes)))
    run_coal_naive = True
    if run_coal_naive:
        maes = []
        for date in chosen_dates:
            first_d = date - timedelta(days=14)
            end_d = date + timedelta(days=13)
            data = df[(df["Date"].dt.date >= first_d) & (df["Date"].dt.date <= end_d)]
            train = data.head(14)
            test = data.tail(14).copy()
            first_week_coal = train.head(7)["Coal"].mean()
            sec_week_coal = train.tail(7)["Coal"].mean()
            w_diff = sec_week_coal - first_week_coal
            last_week_wv = train.tail(7)["Water Value"]
            test["Pred"] = [i + w_diff for i in last_week_wv] + [i + w_diff * 2 for i in last_week_wv]
            maes.append(abs(test["Water Value"] - test["Pred"]).mean())
        print("MAE coal naive = {}".format(sum(maes) / len(maes)))


def price_distribution():
    d_df = get_data("01.07.2014", "02.06.2019", ["System Price"], os.getcwd(), "d")
    print("Min d {:.2f}, max d {:.2f}".format(d_df["System Price"].min(), d_df["System Price"].max()))
    h_df = get_data("01.07.2014", "02.06.2019", ["System Price"], os.getcwd(), "h")
    print("Min h {:.2f}, max h {:.2f}".format(h_df["System Price"].min(), h_df["System Price"].max()))
    plot_dist = True
    if plot_dist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        plot_ax_dist(ax1, d_df["System Price"], "Daily prices [€]")
        plot_ax_dist(ax2, h_df["System Price"], "Hourly prices [€]")
        fig.suptitle("Nord Pool Daily and Hourly Price Distribution - Training Data", size=14)
        plt.tight_layout()
        plt.savefig("output/plots/eda/price_distribution.png")
    quantiles = [0.01, 0.05, 0.1, 0.16, 0.5, 0.84, 0.9, 0.95, 0.99]
    print("Hourly standard dev: {}".format(h_df["System Price"].std()))
    print("Daily standard dev: {}".format(d_df["System Price"].std()))
    df = pd.DataFrame(columns=["Res"] + quantiles)
    for res, value in {"Day": d_df, "Hour": h_df}.items():
        row = {"Res": res}
        for q in quantiles:
            row[q] = value["System Price"].quantile(q)
        df = df.append(row, ignore_index=True)
    df = df.round(2)
    print(df)
    df.to_csv("output/tables/price_quantiles.csv", index=False, float_format="%g")


def plot_ax_dist(ax, data, x_title):
        ax.hist(data, bins=100, density=100, color=first_color)
        ax.set_xlabel(x_title, labelpad=label_pad, size=11)
        ax.set_ylabel("Proportion [%]", labelpad=label_pad, size=11)
        mean = data.mean()
        st_dev = 4 * data.std()
        ax.axis(xmax=mean + st_dev)


def price_seasonality():
    plot = True
    if plot:
        df = get_data("01.07.2014", "02.06.2020", ["System Price", "Month", "Weekday"], os.getcwd(), "h")
        w_df = df[["Weekday", "System Price"]].groupby(by="Weekday").mean()
        w_df["Day"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        h_df = df[["Hour", "System Price"]].groupby(by="Hour").mean().reset_index()
        h_df["Norm"] = h_df["System Price"] - h_df["System Price"].mean()
        w_df["Norm"] = w_df["System Price"] - w_df["System Price"].mean()
        m_df = df[["Month", "System Price"]].groupby(by="Month").mean()
        m_df["Month name"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m_df["Norm"] = m_df["System Price"] - m_df["System Price"].mean()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
        plot_ax_season(ax1, h_df["Norm"], h_df["Hour"].astype(str), "Hour of day", True)
        plot_ax_season(ax2, w_df["Norm"], w_df["Day"], "Day of week", False)
        plot_ax_season(ax3, m_df["Norm"], m_df["Month name"], "Month of year", False)
        fig.suptitle("Seasonality of Nord Pool System Price", size=15)
        plt.tight_layout()
        plt.savefig("output/plots/eda/price_seasonality.png")
    assert False
    df = get_data("01.07.2014", "02.06.2020", ["System Price", "Month", "Holiday", "Weekday"], os.getcwd(), "d")
    sun_df = df[df["Weekday"] == 7]
    reg = df[df["Weekday"] != 7]
    sun_factor = sun_df["System Price"].mean() / reg["System Price"].mean()
    print("Mean Sunday factor: {:.2f}".format(sun_factor))
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df_h = df[df["Holiday"] == 1]
    df_reg = df[df["Holiday"] == 0]
    print("Regular mean {:.2f}, hol mean {:.2f}".format(df_reg["System Price"].mean(), df_h["System Price"].mean()))
    print("Number of holidays: {}".format(len(df_h)))
    df_reg = df_reg.groupby(by=["Year", "Month"]).mean()
    df_h = df_h.groupby(by=["Year", "Month"]).mean()
    factors = []
    for i in range(len(df_h)):
        row = df_h.iloc[i]
        name = row.name
        h_price = row["System Price"]
        reg_price = df_reg[df_reg.index == name]["System Price"].values[0]
        factors.append(h_price/reg_price)
    print("Mean Holiday factor: {:.2f}".format(sum(factors) / len(factors)))


def plot_ax_season(ax, data, x_ticks, x_label, y_label):
    ax.bar(x_ticks, data, color=first_color)
    ax.set_xlabel(xlabel=x_label, labelpad=label_pad, size=12)
    if y_label:
        ax.set_ylabel("Deviation from mean [€]", labelpad=label_pad, size=12)
    ax.set_xticklabels(x_ticks)

def plot_autocorr_2():
    d_df = get_data("01.07.2014", "02.06.2020", ["System Price"], os.getcwd(), "d")
    h_df = get_data("01.07.2014", "02.06.2020", ["System Price"], os.getcwd(), "h")
    fig, (ax1, ax2) = plt.subplots(2, figsize=full_fig)
    plot_acf(d_df["System Price"], lags=21, ax=ax1,  color=sec_color, vlines_kwargs={"color": sec_color}, title="Day")
    #values = acf(d_df["System Price"])
    #print("One day lag: {:.2f}".format(values[1]))
    #values = acf(h_df["System Price"])
    #print("One hour lag: {:.2f}".format(values[1]))
    fig.suptitle("Daily and Hourly System Price Autocorrelation", size=14)
    ax1.set_ylabel("Pearson's coefficient", labelpad=label_pad, size=11)
    ax1.set_xlabel("Day lag", labelpad=label_pad, size=11)
    plot_acf(h_df["System Price"], lags=36, ax=ax2,  color=sec_color, vlines_kwargs={"color": sec_color}, title="Hour")
    ax2.set_ylabel("Pearson's coefficient", labelpad=label_pad, size=11)
    ax2.set_xlabel("Hour lag", labelpad=label_pad, size=11)
    plt.tight_layout(h_pad=1)
    plt.savefig("output/plots/eda/autocorr_price_final.png")
    plt.close()

def measure_volatility():
    import math
    annual_volatility = True
    if annual_volatility:
        df = get_data("01.01.2015", "31.12.2018", ["System Price", "Weekday", "Holiday"], os.getcwd(), "d")
        df["System Price"] = df.apply(
                lambda row: row["System Price"] * 1.12 if row["Holiday"] == 1 and row["Weekday"] != 7 else
                row["System Price"], axis=1)
        annual_volatilities = []
        for year in df["Date"].dt.year.unique():
            sub_df = df[df["Date"].dt.year == year]
            sub_df = get_decomposition(sub_df)
            sub_df["Diff"] = sub_df["Trend"].shift(1)
            sub_df["Dev"] = 100 * (sub_df["Trend"] - sub_df["Trend"].shift(1)) / sub_df["Trend"]
            daily_stdev = sub_df["Dev"].std()
            print("{}: {}".format(year, math.sqrt(len(sub_df)-1) * daily_stdev))
            annual_volatilities.append(math.sqrt(len(sub_df)-1) * daily_stdev)
        print("Mean annual volatility: {:.2f}%".format(sum(annual_volatilities)/len(annual_volatilities)))
    assert False
    daily_volatility = True
    if daily_volatility:
        df = get_data("01.01.2015", "31.12.2019", ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
        all_dates = pd.date_range(dt(2015, 1, 1), dt(2019, 12, 31), freq="d")
        volatilities = []
        for day in all_dates:
            sub_df = df[df["Date"] == day]
            sub_df = sub_df.rename(columns={"System Price": "Trend"})
            sub_df["Diff"] = sub_df["Trend"].shift(1)
            sub_df["Dev"] = 100 * (sub_df["Trend"] - sub_df["Trend"].shift(1)) / sub_df["Trend"]
            hourly_std = sub_df["Dev"].std()
            volatilities.append(math.sqrt(len(sub_df)-1) * hourly_std)
        print("Mean daily volatility: {:.2f}%".format(sum(volatilities)/len(volatilities)))

def get_decomposition(df):
    df = df.reset_index(drop=True)
    decomp = seasonal_decompose(df["System Price"], model='multiplicative', period=7)
    df["Factor"] = decomp.seasonal
    df["Trend"] = decomp.trend
    df["Adj Trend"] = df["System Price"] / df["Factor"]
    df.loc[0, "Trend"] = sum(df["Adj Trend"].head(3) * np.asarray([0.2, 0.3, 0.5]))
    df.loc[len(df)-1, "Trend"] = sum(df["Adj Trend"].tail(3) * np.asarray([0.2, 0.3, 0.5]))
    df["Trend"] = df["Trend"].interpolate(method='linear', limit_area="inside")
    df = df.drop(columns=["Adj Trend", "Weekday", "Holiday", "System Price", "Factor"])
    return df


def simple_curves():
    supply = [1.09**(i) for i in range(1, 30)]
    demand_1 = [1.09**(i*0.8) for i in range(30, 1, -2)]
    last_val = demand_1[-1]
    for j in range(1, 6):
        demand_1.append(last_val-0.15*j)
    plt.subplots(figsize=(6.5, 6))
    d_col = plt.get_cmap("tab10")(0)
    s_col = plt.get_cmap("tab10")(1)
    plt.plot(range(len(supply)), supply, color=s_col, lw=2, label="Supply")
    plt.plot(range(2, len(demand_1)+2), demand_1, color=d_col, lw=2, label="Demand 1")
    plt.plot(range(14, len(demand_1)+14), [i + 6 for i in demand_1], color=d_col, lw=2, label="Demand 2")
    plt.ylim(0, 15)
    plt.hlines(2.62, 0, 10, ls="dotted", color="grey")
    plt.hlines(8.12, 0, 23.5, ls="dotted", color="grey")
    plt.vlines(10, 0, 2.6, ls="dotted", color="grey")
    plt.vlines(23.5, 0, 8.12, ls="dotted", color="grey")
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.xlabel("Quantity", labelpad=label_pad+4)
    plt.ylabel("Price", labelpad=label_pad+4)
    plt.gca().yaxis.set_label_coords(-0.025, 0.87)
    plt.gca().xaxis.set_label_coords(0.9, -0.025)
    plt.text(9.5, -0.5, "$Q_1$")
    plt.text(23, -0.5, "$Q_2$")
    plt.text(-2, 2.5, "$P_1$")
    plt.text(-2, 8.12, "$P_2$")
    plt.text(29, 12, "$S$", size=12)
    plt.text(4, 7, "$D_1$", size=12)
    plt.text(16, 13, "$D_2$", size=12)
    plt.xlim(0, 35)
    plt.arrow(0, 0, 35, 0, width=0.015, color="k", clip_on=False, head_width=0.5, head_length=0.5)
    plt.arrow(0, 0, 0, 14, width=0.015, color="k", clip_on=False, head_width=0.8, head_length=0.4)
    plt.gca().set(frame_on=False)
    plt.tight_layout()
    path = "output/plots/eda/simple_curve.png"
    plt.savefig(path)


def plot_articles():
    from collections import Counter
    data = pd.read_csv("input/articles/phase_2.csv", sep=";", usecols=["Type of method/model", "Horizon", "Output"])
    short = get_count_df(data[data["Horizon"].str.contains("1")].reset_index(drop="True"))
    medium = get_count_df(data[data["Horizon"].str.contains("2")].reset_index(drop="True"))
    long = get_count_df(data[data["Horizon"].str.contains("3")].reset_index(drop="True"))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=full_fig)
    axs = [ax1, ax2, ax3]
    dfs = [short, medium, long]
    labels = ["Short", "Mid", "Long"]
    for i in range(len(axs)):
        df, ax = (dfs[i], axs[i])
        s_prob = len(df[df["Output"] == 2]) / len(df)
        df = df.groupby(['Model', 'Output'])['Model'].count().unstack("Output").fillna(0)
        df.plot(kind='bar', stacked=True, ax=ax, color=[first_color, sec_color])
        ax.set_xticklabels(["Fund.", "M. A", "R-F", "Stat.", "CI"], rotation=0, size=11)
        ax.legend(["Point ({:.0f}%)".format(100 * (1-s_prob)), "Prob. ({:.0f}%)".format(100*s_prob)], prop={"size":11})
        if i == 0:
            ax.set_ylabel("Number of studies", labelpad=label_pad, size=11)
        ax.set_xlabel("Model approach", labelpad=label_pad, size=11)
        ax.set_title("{}-term".format(labels[i]), size=13, y=1.01)
    fig.suptitle("Outcome of Scopus Literature Review - November 2020", y=0.99, size=14)
    plt.tight_layout()
    plt.savefig("output/plots/eda/articles.png")


def get_count_df(df):
    new = pd.DataFrame(columns=["Model", "Output"])
    new = new.fillna(np.PZERO)
    for i in range(len(df)):
        methods = [int(m.replace(" ", "")) for m in df.loc[i, "Type of method/model"].split(",")]
        output = [int(o.replace(" ", "")) for o in df.loc[i, "Output"].split(",")]
        for m in methods:
            for o in output:
                row = {"Model": m, "Output": o}
                new = new.append(row, ignore_index=True)
    return new


def normal_test():
    d = get_data("01.07.2014", "12.06.2019", ["System Price"], os.getcwd(), "d")
    print("mean dat {:.3f}".format(d["System Price"].mean()))
    print(stats.normaltest(d["System Price"])[1])
    d = get_data("01.07.2014", "02.06.2019", ["System Price"], os.getcwd(), "h")
    print("mean dat {:.3f}".format(d["System Price"].mean()))
    print(stats.normaltest(d["System Price"])[1])

if __name__ == '__main__':
    print("Running eda..")
    # plot_norm_weekday()
    # plot_norm_month()
    # plot_daily_vs_hourly_prices()
    # plot_norm_week()
    # plot_temperatures()
    # plot_precipitation()
    # plot_all_variables_per_year()
    # plot_correlation("System Price", "Prec Norway 7")
    # plot_correlation_norm_price_per_year("System Price", "Total Hydro Dev")
    # check_lagged_correlation("System Price", "Total Hydro Dev")
    # lin_model_test()
    # lin_model_test_norm_per_year()
    # plot_autocorrelation()
    # explore_eikon_data()
    # plot_random_auctions(10)
    # eda_auction_data()
    # plot_european_prices()
    # plot_nordpool_prices()
    # plot_spring_2019()
    # check_demand_stationarity()
    # explore_holyday()
    # explore_snowfall()
    # explore_coal_and_wv()
    # explore_hydro_and_coal_multiple_regression()
    # price_distribution()
    # price_seasonality()
    plot_autocorr_2()
    # measure_volatility()
    # simple_curves()
    # plot_articles()
    # normal_test()


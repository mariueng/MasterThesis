from src.system.generate_periods import get_random_periods
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import numpy as np
from src.models.curve_model.daily_demand_naive import predict_daily_demand_naive
from src.models.curve_model.hourly_demand import decompose_daily_to_hourly_demand
import random
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


def estimate_hourly_demand_errors():
    periods = get_all_2019_periods()
    columns = ["Day {}".format(i) for i in range(1, 15)]
    df = pd.DataFrame(columns=columns)
    for i in range(len(periods)):
        p_df = pd.DataFrame(columns=df.columns)
        period = periods[i]
        start = period[0]
        print("Running from {}".format(start))
        end = period[1]
        trans_table = pd.read_csv("demand_temp_transitions.csv")
        day_demand_forecast = predict_daily_demand_naive(start, end, trans_table)
        hourly_demand_forecast = decompose_daily_to_hourly_demand(day_demand_forecast, start, end, plot=False)
        for j in range(14):
            start_idx = j*24
            day_j = hourly_demand_forecast.iloc[start_idx:start_idx+24, :].reset_index(drop=True)
            p_df["Day {}".format(j+1)] = day_j["Curve Demand"] - day_j["Demand Forecast"]
        df = df.append(p_df, ignore_index=True)
    df.to_csv("demand_errors.csv", index=False, float_format="%g")
    print(df)
    print("-----------------------\n")
    for col in df.columns:
        print("{} mean {:.2f} MWh".format(col, abs(df[col]).mean()))


def get_upper_and_lower_bound(alpha, demand, n, error_series, supply_line, price_errors, point_forecast):
    max_vol = max(supply_line.coords.xy[0])
    min_vol = min(supply_line.coords.xy[0])
    upper_index = int(n * alpha)
    lower_index = int(n * (1-alpha))
    min_upper = point_forecast + price_errors["Positive 95"].values[0]
    max_lower = point_forecast - price_errors["Negative 95"].values[0]
    chosen_errors = np.random.choice(error_series, n)
    chosen_errors.sort()
    upper_demand = demand + chosen_errors[upper_index]
    lower_demand = demand + chosen_errors[lower_index]
    if upper_demand > max_vol:
        upper_demand = max_vol
    elif upper_demand < min_vol:
        upper_demand = min_vol
    if lower_demand < min_vol:
        lower_demand = min_vol
    elif lower_demand > max_vol:
        lower_demand = max_vol
    plot_demand_dist(supply_line, demand)
    assert False
    u_d_line = LineString([(upper_demand, -10), (upper_demand, 210)])
    l_d_line = LineString([(lower_demand, -10), (lower_demand, 210)])
    upper_inter = supply_line.intersection(u_d_line)
    if type(upper_inter) is MultiPoint or type(upper_inter) is LineString:
        print(supply_line.coords.xy[0])
        print(supply_line.coords.xy[1])
        print(u_d_line)
        print("Max volume {}".format(max_vol))
        print("Point demand: {}".format(demand))
        print("Upper demand: {}".format(upper_demand))
    lower_inter = supply_line.intersection(l_d_line)
    if type(lower_inter) is MultiPoint or type(lower_inter) is LineString:
        print(supply_line.coords.xy[0])
        print(supply_line.coords.xy[1])
        print(l_d_line)
        print("Min volume {}".format(min_vol))
        print("Point demand: {}".format(demand))
        print("Lower demand: {}".format(lower_demand))
    upper_bound = max(min_upper, upper_inter.y)
    upper_bound = min(200, upper_bound)
    #upper_bound = min(upper_bound, point_forecast * 1.5)
    lower_bound = min(max_lower, lower_inter.y)
    lower_bound = max(-1, lower_bound)
    #upper_bound = min(upper_bound, point_forecast * 0.5)
    return upper_bound, lower_bound


def plot_demand_dist(supply_line, demand):
    import scipy.stats as stats
    np.random.seed(seed=10)
    a, b = demand-5000, demand+5000
    mu, sigma = demand, 3200
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    values = dist.rvs(20)
    values.sort()
    lower_demand = values[2]
    upper_demand = values[17]
    values = np.delete(values, 17)
    values = np.delete(values, 2)
    plt.subplots(figsize=(13, 7))
    plt.plot(supply_line.coords.xy[0], supply_line.coords.xy[1], label="Supply", color=plt.get_cmap("tab10")(1))
    y_min, y_max = plt.ylim()
    price = supply_line.intersection(LineString([(demand, y_min), (demand, y_max)])).y
    l_price = supply_line.intersection(LineString([(lower_demand, y_min), (lower_demand, y_max)])).y
    u_price = supply_line.intersection(LineString([(upper_demand, y_min), (upper_demand, y_max)])).y
    plt.vlines(demand, y_min, y_max, label="Demand forecast (€{:.1f})".format(price), linewidth=3, zorder=5)
    for i in range(len(values)):
        lab = "_nolegend" if i != 0 else "Simulated demand"
        plt.vlines(values[i], y_min, y_max, label=lab, linestyles="dotted", color="grey")
    plt.vlines(lower_demand, y_min, y_max, label="Lower demand (€{:.1f})".format(l_price), linewidth=3, linestyle="dotted")
    plt.vlines(upper_demand, y_min, y_max, label="Upper demand (€{:.1f})".format(u_price), linewidth=3, linestyle="dotted")
    for line in plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                           shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylim(y_min, y_max)
    plt.title("Demand Simulation for Fundamental Uncertainty Estimation", pad=20)
    plt.xlabel("Volume [MWh]", labelpad=12)
    plt.ylabel("Price [€]", labelpad=12)
    plt.tight_layout()
    plt.savefig("eda/sim_demand.png")
    assert False

def estimate_daily_price_errors():
    from src.models.curve_model.curve_model import CurveModel
    periods = get_random_periods(2)
    periods = get_all_2019_periods()
    columns = ["Day {}".format(i) for i in range(1, 15)]
    df = pd.DataFrame(columns=columns)
    model = CurveModel()
    for i in range(len(periods)):
        p_df = pd.DataFrame(columns=df.columns)
        period = periods[i]
        print("Running from {}".format(period[0]))
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        for j in range(14):
            day_j_forecast = result_df.iloc[j*24:j*24+24, :]
            errors = day_j_forecast["System Price"] - day_j_forecast["Forecast"]
            p_df["Day {}".format(1+j)] = errors.reset_index(drop=True)
        df = df.append(p_df, ignore_index=True)
    df = df.round(2)
    df.to_csv("price_errors.csv", index=False, float_format="%g")
    print("-----------------------\n")
    for col in df.columns:
        print("{} mean {:.2f} Euro".format(col, abs(df[col]).mean()))


def estimate_daily_price_errors_2():
    df = pd.read_csv("../../results/validation/CurveModel/forecast.csv", usecols=["Period", "System Price", "Forecast"])
    columns = ["Day {}".format(i) for i in range(1, 15)]
    result = pd.DataFrame(columns=columns)
    for period in df["Period"].unique():
        r = pd.DataFrame(columns=result.columns)
        sub = df[df["Period"]==period].reset_index(drop=True)
        for i in range(14):
            day_df = sub.loc[i*24:i*24+23, :].reset_index(drop=True)
            r["Day {}".format(i+1)] = day_df["System Price"] - day_df["Forecast"]
        result = result.append(r, ignore_index=True)
    result = result.transform(np.sort)
    result = result.iloc[160:len(result)-91]
    table = pd.DataFrame(columns=["Day", "Mean", "Negative 95", "Positive 95"])
    for col in result.columns:
        row = {"Day": col, "Negative 95": 1.96*result[[col]][result[col] < 0][col].std(),
        "Positive 95": 1.96*result[[col]][result[col] > 0][col].std(), "Mean": abs(result[col]).mean()}
        table = table.append(row, ignore_index=True)
    table = table.round(2)
    table.to_csv("price_table.csv", index=False, float_format="%g")
    result = result.round(2)
    result.to_csv("price_errors.csv", index=False, float_format="%g")

if __name__ == '__main__':
    print("Running methods")
    # estimate_hourly_demand_errors()
    # estimate_daily_price_errors()
    # estimate_daily_price_errors_2()

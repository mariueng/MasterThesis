from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from data.data_handler import get_data
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from src.system.generate_periods import get_all_2019_periods
from src.system.generate_periods import get_random_periods
import warnings
random.seed(1)
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)


def get_decomp_new(hist):
    hist = hist.reset_index(drop=True)
    demand_column = "Curve Demand"
    decomp = seasonal_decompose(hist[demand_column], model='multiplicative', period=7)
    hist["Factor"] = decomp.seasonal
    hist["Trend"] = decomp.trend
    hist["Adj Trend"] = hist["Curve Demand"] / hist["Factor"]
    hist.loc[0, "Trend"] = sum(hist["Adj Trend"].head(3) * np.asarray([0.2, 0.3, 0.5]))
    hist.loc[len(hist)-1, "Trend"] = sum(hist["Adj Trend"].tail(3) * np.asarray([0.2, 0.3, 0.5]))
    hist["Trend"] = hist["Trend"].interpolate(method='quadratic', limit_area="inside")
    hist = hist.drop(columns=["Adj Trend"])
    return hist


def make_transition_table():
    history = get_data("01.07.2014", "31.12.2019", ["Curve Demand", "Temp Norway", "Weekend", "Holiday"], os.getcwd(), "d")
    df = pd.DataFrame(columns=["Date", "From demand", "From temp", "Diff", "Weekend", "Factor"])
    h_factor = get_hol_factor()
    for i in range(len(history)-7):
        current = history.iloc[i, :]
        next_week = history.iloc[i+7, :]
        date = current["Date"]
        cur_demand = current["Curve Demand"] * h_factor if current["Holiday"] == 1 else current["Curve Demand"]
        next_demand = next_week["Curve Demand"] * h_factor if next_week["Holiday"] == 1 else next_week["Curve Demand"]
        from_temp = current["Temp Norway"]
        to_temp = next_week["Temp Norway"]
        difference = to_temp - from_temp
        f = next_demand / cur_demand
        row = {"Date": date, "From demand": cur_demand, "From temp":from_temp, "Diff": difference,
               "Weekend": current["Weekend"], "Factor": f}
        df = df.append(row, ignore_index=True)
    df.to_csv("demand_temp_transitions.csv", index=False, float_format="%g")


def predict_daily_demand_naive(start, end, trans_table, plot, true_demand):
    hist_data = get_data(start-td(days=7), start-td(days=1), ["Curve Demand", "Weekday", "Temp Norway", "Holiday"], os.getcwd(), "d")
    hist_data["Curve Demand"] = hist_data.apply(
            lambda row: row["Curve Demand"] * get_hol_factor() if row["Holiday"] == 1 else row["Curve Demand"], axis=1)
    temp_forecast = get_data(start, end, ["Weekday", "Temp Norway"], os.getcwd(), "d")
    result_df = get_data(start, end, ["Weekday", "Holiday"], os.getcwd(), "d")
    for i in range(len(result_df)):
        weekend = 0 if result_df.loc[i, "Weekday"] < 6 else 1
        if i < 7:
            from_row = hist_data[hist_data["Weekday"] == result_df.loc[i, "Weekday"]].reset_index(drop=True)
            from_temp = from_row.loc[0, "Temp Norway"]
            from_demand = from_row.loc[0, "Curve Demand"]
        else:
            from_row = temp_forecast[temp_forecast["Weekday"] == result_df.loc[i, "Weekday"]].reset_index(drop=True)
            from_temp = from_row.loc[0, "Temp Norway"]
            from_row = result_df[result_df["Weekday"] == result_df.loc[i, "Weekday"]].reset_index(drop=True)
            from_demand = from_row.loc[0, "Demand Forecast"]
        diff = temp_forecast.loc[i, "Temp Norway"] - from_temp
        sub_table = trans_table[trans_table["Weekend"] == weekend].reset_index(drop=True)
        df_from = pd.Series(data={"From demand": from_demand, "From temp": from_temp, "Diff": diff}).to_frame().transpose()
        normalised = df_from.append(sub_table[["From demand", "From temp", "Diff"]], ignore_index=True)
        normalized_df = (normalised-normalised.min())/(normalised.max()-normalised.min())
        current_state = normalized_df.iloc[0:1, :]
        sub_states = normalized_df.iloc[1:, :].reset_index(drop=True)
        for col in current_state.columns:
            sub_states[col] = abs(sub_states[col] - current_state.loc[0, col])
        sub_states["Score"] = sub_states.sum(axis=1)
        numbers = 3
        sub_states = sub_states.sort_values(by="Score").head(numbers)
        smallest_df = sub_table.loc[sub_states.index]
        mean_factor = smallest_df["Factor"].mean()
        forecast = from_demand * mean_factor
        if result_df.loc[i, "Holiday"] == 1:
            forecast = forecast / get_hol_factor()
        result_df.loc[i, "Demand Forecast"] = forecast
    result_df = result_df.merge(true_demand, on="Date")
    if plot:
        plt.subplots(figsize=full_fig)
        plt.plot(result_df["Date"], result_df["Demand Forecast"], color=first_color, label="Demand forecast")
        mape = 100 * (abs((result_df["Curve Demand"]-result_df["Demand Forecast"]) / result_df["Curve Demand"])).mean()
        plt.plot(result_df["Date"], result_df["Curve Demand"], label="True demand", color=sec_color)
        for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                               fancybox=True, shadow=True).get_lines():
            line.set_linewidth(2)
        plt.title("Daily Demand Forecast from {} to {} (MAPE = {:.2f})".format(start.date(), end.date(), mape), pad=title_pad)
        plt.xlabel("Date", labelpad=label_pad)
        plt.ylabel("Volume [MWh]", labelpad=label_pad)
        plt.tight_layout()
        plt.savefig("daily_demand_plots/{}_to_{}.png".format(start.date(), end.date()))
        plt.close()
    result_df = result_df[["Date", "Demand Forecast", "Curve Demand"]]
    return result_df


def evaluate_naive_demand():
    periods = get_random_periods(10)
    periods = get_all_2019_periods()
    periods = [(dt(2019, 5, 6).date(), dt(2019, 5, 19).date())]
    df = pd.DataFrame(columns=["Period", "Date", "Demand Forecast", "Curve Demand"])
    result = pd.DataFrame(columns=["Period", "Start", "End", "MAE", "MAPE"])
    trans_table = pd.read_csv("demand_temp_transitions.csv")
    for i in range(len(periods)):
        p = periods[i]
        start = p[0]
        print("Start {}".format(start))
        end = p[1]
        p_df = pd.DataFrame(columns=df.columns)
        forecast_df = predict_daily_demand_naive(start, end, trans_table)
        p_df["Date"] = forecast_df["Date"]
        p_df["Demand Forecast"] = forecast_df["Demand Forecast"]
        p_df["Curve Demand"] = get_data(start, end, ["Curve Demand"], os.getcwd(), "d")["Curve Demand"]
        p_df["Period"] = int(i+1)
        df = df.append(p_df[["Period", "Date", "Demand Forecast", "Curve Demand"]], ignore_index=True)
        ae = (abs(p_df["Curve Demand"] - p_df["Demand Forecast"])).mean()
        ape = 100 * ae / p_df["Curve Demand"].mean()
        result = result.append({"Period": int(i+1), "Start": start, "End": end, "MAE": ae, "MAPE": ape}, ignore_index=True)
    print(result)
    print("-----------------------")
    print("MAE {:.2f}, MAPE {:.2f}".format(result["MAE"].mean(), result["MAPE"].mean()))
    #result.to_csv("naive_demand.csv", index=False)
    print(result.sort_values(by='MAPE', ascending=False).head(3))


def get_general_temp_coeff():
    df = get_data("01.01.2019", "31.12.2019", ["Weekday", "Curve Demand", "Temp Norway"], os.getcwd(), "d")
    df = get_decomp_new(df)
    lin_reg = LinearRegression()
    lin_reg.fit(df[["Temp Norway"]].values, df[["Trend"]].values)
    pred_line = lin_reg.predict(df[["Temp Norway"]].values)
    print(lin_reg.coef_[0])
    print(lin_reg.intercept_)
    assert False
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["Trend"])
    plt.plot(df["Date"], pred_line)
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(df["Date"], df["Temp Norway"])
    plt.show()


def explore_holiday_factor():
    df = get_data("01.07.2014", "31.12.2019", ["Curve Demand", "Weekday", "Holiday"], os.getcwd(), "d")
    holidays = df[(df["Holiday"] == 1) | (df["Weekday"] == 7)]
    no_holidays = df[(df["Holiday"] == 0) & (df["Weekday"] != 7)]
    mean_curve_holiday = holidays["Curve Demand"].mean()
    mean_regular = no_holidays["Curve Demand"].mean()
    print(mean_regular)
    print(mean_curve_holiday)
    factor_for_multiplication = mean_regular / mean_curve_holiday
    print(factor_for_multiplication)


def get_hol_factor():
    return 1.1095


if __name__ == '__main__':
    # explore_holiday_factor()
    evaluate_naive_demand()
    # make_transition_table()
    # get_general_temp_coeff()

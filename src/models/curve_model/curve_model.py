from datetime import datetime as dt
from datetime import timedelta
from matplotlib.lines import Line2D
import pandas as pd
import os
import numpy as np
import random
from src.system.scores import calculate_interval_score
from src.system.scores import calculate_coverage_error
from src.system.scores import get_all_point_metrics
from shapely.geometry import LineString
from src.system.generate_periods import get_random_periods
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from data.data_handler import get_auction_data
from src.models.curve_model.daily_demand_naive import predict_daily_demand_naive
from src.models.curve_model.supply_curve import get_supply_curve
from src.models.curve_model.supply_curve import get_supply_curve_water_values
from src.models.curve_model.hourly_demand import decompose_daily_to_hourly_demand
from src.models.curve_model.simulate_errors import get_upper_and_lower_bound
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import shutil

class CurveModel:
    def __init__(self):
        prev_workdir = os.getcwd()
        os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
        self.name = "Curve Model"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))
        dem_paths = ["daily_demand_plots", "hourly_demand_plots"]
        price_paths = ["price_curves", "prices"]
        self.plot_demand = plot_demand
        if self.plot_demand:
            for p in dem_paths:
                if os.path.exists(p):
                    shutil.rmtree(p)  # delete old
                os.makedirs(p)  # create new
        self.plot_price = plot_price
        if self.plot_price:
            for p in price_paths:
                if os.path.exists(p):
                    shutil.rmtree(p)  # delete old
                os.makedirs(p)  # create new
        self.day_demand_score = pd.DataFrame(
            columns=["Period", "Date", "Curve Demand", "Demand Forecast", "Error", "AE", "APE"])
        self.day_demand_score.to_csv("demand_scores/day_demand_results.csv", index=False)
        self.hour_demand_score = pd.DataFrame(
            columns=["Period", "Date", "Hour", "Curve Demand", "Demand Forecast", "Error", "AE", "APE"])
        self.hour_demand_score.to_csv("demand_scores/hour_demand_results.csv", index=False)
        self.period = 1
        os.chdir(prev_workdir)

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df, demand, supply = self.get_forecast(forecast_df)
        return forecast_df, demand, supply

    def get_forecast(self, forecast_df):
        demand_df, supply_df = (forecast_df[["Date", "Hour"]].copy(), forecast_df[["Date", "Hour"]].copy())
        prev_workdir = os.getcwd()
        os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
        start_date = forecast_df.at[0, "Date"]
        end_date = start_date + timedelta(days=13)
        true_price = get_data(start_date, end_date, ["System Price"], os.getcwd(), "h")
        #true_supply = get_auction_data(start_date, end_date, "s", os.getcwd())
        help_data = get_data(start_date - timedelta(days=7), end_date, ["Season", "Total Hydro Dev", "Coal"],
                             os.getcwd(), "d")
        #last_week_coal = help_data.head(7)["Coal"].mean()
        data, h_demand_forecast, supply_mean_week, d_errors, p_table = self.get_data_demand_and_supply(start_date, end_date)
        demand_df["Demand Forecast"] = h_demand_forecast["Demand Forecast"]
        for col in supply_mean_week.columns[1:]:
            supply_df[col] = np.NAN
        #wv_model = pickle.load(open("wv_model.pickle", 'rb'))
        if plot_price:
            if not os.path.exists("price_curves/curves_{}_{}".format(start_date.date(), end_date.date())):
                os.makedirs("price_curves/curves_{}_{}".format(start_date.date(), end_date.date()))
                curve_data = get_auction_data(start_date, end_date, ["s", "d"], os.getcwd())  # only used for plotting
            fig, ax, lines = self.create_background_curve_plot(supply_mean_week, h_demand_forecast)
        for i in range(len(forecast_df)):
            true_p = true_price.loc[i, "System Price"]
            demand = h_demand_forecast.loc[i, "Demand Forecast"]
            hour = forecast_df.loc[i, "Hour"]
            month = data.loc[i, "Month"]
            weekend = data.loc[i, "Weekend"]
            mean_supply_curve = supply_mean_week.iloc[hour]
            #last_week_row = help_data[help_data["Date"] == forecast_df.loc[i, "Date"] - timedelta(days=7)]
            #help_row = help_data[help_data["Date"] == forecast_df.loc[i, "Date"]]
            safe_supply = get_supply_curve(month, hour, weekend, mean_supply_curve, safe=True)  # VERSION 0
            for j in range(len(safe_supply)):
               supply_df.iloc[i, j+2] = safe_supply[j]
            # safe_supply = get_supply_curve_water_values(month, hour, weekend, mean_supply_curve, help_row, last_week_row, wv_model, last_week_coal, safe=True) # VERSION 1
            # safe_supply = true_supply.iloc[i][2:]
            volumes = safe_supply.values
            max_volume, min_volume = (max(volumes), min(volumes))
            supply_line = LineString(np.column_stack((volumes, s_classes)))
            demand = max_volume if demand > max_volume else demand
            demand = min_volume if demand < min_volume else demand
            demand_line = LineString([(demand, -10), (demand, 210)])
            point_forecast = supply_line.intersection(demand_line).y
            forecast_df.loc[i, "Forecast"] = point_forecast
            if plot_price:
                curve_hour = curve_data.iloc[i]
                plot_curves(curve_hour, demand_line, supply_line, fig, ax, lines, point_forecast, true_p,
                            start_date, end_date)
            # upper, lower = (point_forecast * 1.2, point_forecast * 0.8)
            upper, lower = get_prob_forecast(0.95, demand, 1000, month, hour, weekend, mean_supply_curve, i,
                                             point_forecast, d_errors, p_table)
            forecast_df.loc[i, "Upper"] = upper
            forecast_df.loc[i, "Lower"] = lower
        os.chdir(prev_workdir)
        mean_forecast = forecast_df["Forecast"].median()
        for i in range(14):
            margin_up, margin_low = (p_table.loc[i, "Positive 95"], p_table.loc[i, "Positive 95"])
            #margin_up, margin_low = (p_table.loc[i, "Mean"], p_table.loc[i, "Mean"])
            min_upper = mean_forecast + margin_up
            max_lower = mean_forecast - margin_low
            forecast_df.loc[i*24:i*24+24, "Upper"][forecast_df["Upper"] < min_upper] = min_upper
            forecast_df.loc[i*24:i*24+24, "Lower"][forecast_df["Lower"] > max_lower] = max_lower
        return forecast_df, demand_df, supply_df

    @staticmethod
    def create_background_curve_plot(supply_mean_week, h_demand_forecast):
        min_x = min(supply_mean_week["s -10"])
        max_x = max(h_demand_forecast["Demand Forecast"]) * 1.1
        fig, ax = plt.subplots(figsize=full_fig)
        ax.axis(xmin=min_x, xmax=max_x, ymin=-1, ymax=100)
        ax.set_xlabel("Volume [MWh]", labelpad=label_pad)
        ax.set_ylabel("Price [€]", labelpad=label_pad)
        lines = [Line2D([0], [0], color=d_col, linestyle="solid", figure=fig),
                 Line2D([0], [0], color=s_col, linestyle="solid", figure=fig),
                 Line2D([0], [0], color=d_col, linestyle="dotted", figure=fig),
                 Line2D([0], [0], color=s_col, linestyle="dotted", figure=fig)]
        labels = ['Forecast demand', 'Forecast supply', 'True demand', "True supply"]
        for line in ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                              shadow=True, labels=labels, handles=lines).get_lines():
            line.set_linewidth(2)
        lines[0].set_ydata([-10, 210])
        lines[2].set_ydata(d_classes)
        for line in [lines[1], lines[3]]:
            line.set_ydata(s_classes)
        for line in lines:
            ax.add_line(line)
        ax.set_title("Title", pad=title_pad)
        plt.tight_layout()
        return fig, ax, lines

    def get_data_demand_and_supply(self, start_date, end_date):
        data = get_data(start_date, end_date, ["Month", "Weekend"], os.getcwd(), "h")
        trans_table = pd.read_csv("demand_temp_transitions.csv")
        day_demand = get_data(start_date, end_date, ["Curve Demand"], os.getcwd(), "d")
        day_forecast = predict_daily_demand_naive(start_date, end_date, trans_table, plot_demand, day_demand)
        self.save_demand_score(day_forecast, "day")
        hour_demand = get_data(start_date, end_date, ["Curve Demand"], os.getcwd(), "h")
        hour_forecast = decompose_daily_to_hourly_demand(day_forecast, start_date, end_date, plot_demand, hour_demand)
        self.save_demand_score(hour_forecast, "hour")
        last_week_supply = get_auction_data(start_date - timedelta(days=7), start_date - timedelta(days=1), "s",
                                            os.getcwd())
        supply_mean_week = last_week_supply.groupby(by="Hour").mean().reset_index()
        # trans_table_water_values = pd.read_csv("water_value_hydro_dev_profiles_2.csv")
        d_errors = pd.read_csv("demand_errors.csv")
        p_errors = pd.read_csv("price_table.csv")
        return data, hour_forecast, supply_mean_week, d_errors, p_errors

    def save_demand_score(self, df, res):
        df["Error"] = df["Curve Demand"] - df["Demand Forecast"]
        df["AE"] = abs(df["Error"])
        df["APE"] = 100 * df["AE"] / df["Curve Demand"]
        df["Period"] = self.period
        self.period += 1 if res == "hour" else 0
        df = df[self.day_demand_score.columns]
        df = df.round(2)
        df.to_csv("demand_scores/{}_demand_results.csv".format(res), mode='a', header=False, index=False)



def plot(ax, style, x, y):
    return ax.plot(x, y, style, animated=True)[0]


d_classes = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105, 165,
             210]
d_col = plt.get_cmap("tab10")(0)
s_col = plt.get_cmap("tab10")(1)
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)
plot_demand = False
plot_price = False


def get_prob_forecast(alpha, demand, n, month, hour, weekend, mean_supply_curve, i, point_forecast, d_errors, p_errors):
    demand_errors = d_errors["Day {}".format((i // 24) + 1)].values
    price_errors = p_errors[p_errors["Day"] == "Day {}".format((i // 24) + 1)]
    prob_supply = get_supply_curve(month, hour, weekend, mean_supply_curve, safe=False)
    prob_supply_line = LineString(np.column_stack((prob_supply.values, s_classes)))
    upper, lower = get_upper_and_lower_bound(alpha, demand, n, demand_errors, prob_supply_line, price_errors,
                                             point_forecast)
    return upper, lower


def plot_curves(curve_hour, d_line, s_line, fig, ax, lines, est_p, true_p, s_date, e_date):
    changed_y_axis = False
    if max(est_p, true_p) > 100:
        ax.axis(ymax=max(est_p, true_p) * 1.1)
        changed_y_axis = True
    date = curve_hour["Date"].date()
    hour = curve_hour["Hour"]
    demand_volumes = curve_hour.iloc[2:14].values
    supply_volumes = curve_hour.iloc[14:len(curve_hour)].values
    lines[0].set_xdata([i for i in d_line.coords.xy[0]])
    lines[1].set_xdata([i for i in s_line.coords.xy[0]])
    lines[2].set_xdata(demand_volumes)
    lines[3].set_xdata(supply_volumes)
    ax.set_title("Curve Forecast {}-{:02d}\t($p = {:.2f}, \^p = {:.2f}$)".format(date, hour, true_p, est_p),
                 pad=title_pad)
    path_save = "price_curves/curves_{}_{}/{}_{}.png".format(s_date.date(), e_date.date(), date, hour)
    plt.savefig(path_save)
    if changed_y_axis:
        ax.axis(ymax=100)


def run_model(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        start_date = time_df.at[0, "Date"].date()
        end_date = start_date + timedelta(days=13)
        print("Forecasting from {} to {}".format(start_date, end_date))
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0] - timedelta(days=7), period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_df["Hour Time"] = pd.to_datetime(result_df['Hour'], format="%H").dt.time
        result_df["DateTime"] = result_df.apply(lambda r: dt.combine(r['Date'], r['Hour Time']), 1)
        plt.subplots(figsize=full_fig)
        plt.plot(result_df["DateTime"], result_df["System Price"], label="True", color=first_color)
        plt.plot(result_df["DateTime"], result_df["Forecast"], label="Forecast", color=sec_color)
        plt.legend()
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.title("Curve Model {} to {}".format(start_date, end_date))
        plt.tight_layout()
        plt.savefig("prices/{}_{}.png".format(start_date, end_date))
        result_df = result_df.dropna()
        result_list.append(result_df)
    result = pd.concat(result_list).reset_index(drop=True)
    print(result[["Date", "System Price", "Forecast"]])
    print("-----------------\n")
    pm = get_all_point_metrics(result)
    print("Mape {:.2f}, smape {:.2f}, mae {:.2f}, rmse {:.2f}".format(pm["mape"], pm["smape"], pm["mae"], pm["rmse"]))
    print("IS Score: {:.2f}".format(calculate_interval_score(result)))
    err = calculate_coverage_error(result)
    print("Cov Error Score: {:.2f}, mean coverage {:.2f}%".format(err[0], err[1]))


def explore_supply_outlier():
    if os.path.exists("supply_yesterday_vs_week.csv"):
        result = pd.read_csv("supply_yesterday_vs_week.csv")
    else:
        periods = get_all_2019_periods()
        result = pd.DataFrame(columns=["Period", "Yesterday MAE", "Week MAE"])
        for i in range(len(periods)):
            p = periods[i]
            start_date = p[0]
            end_date = start_date + timedelta(days=13)
            prev_day = start_date - timedelta(days=1)
            yesterday_supply = get_auction_data(prev_day, prev_day, "s", os.getcwd())
            last_week_supply = get_auction_data(prev_day - timedelta(days=6), prev_day, "s", os.getcwd())
            supply_mean = last_week_supply.groupby(by="Hour").mean().reset_index()
            true_supply = get_auction_data(start_date, end_date, "s", os.getcwd())
            yesterday_aes = []
            weekly_aes = []
            print(i)
            for j in range(len(true_supply)):
                true = true_supply.iloc[j]
                hour = true["Hour"]
                yesterday = yesterday_supply[yesterday_supply["Hour"] == hour].iloc[0]
                week = supply_mean[supply_mean["Hour"] == hour].iloc[0]
                yest_aes = [abs(yesterday[i] - true[i]) for i in range(2, len(true))]
                yesterday_aes.append(round(sum(yest_aes) / len(yest_aes), 2))
                week_aes = [abs(week[i - 1] - true[i]) for i in range(2, len(true))]
                weekly_aes.append(round(sum(week_aes) / len(week_aes), 2))
            row = {"Period": i + 1, "Yesterday MAE": sum(yesterday_aes) / len(yesterday_aes),
                   "Week MAE": sum(weekly_aes) /
                               len(weekly_aes)}
            result = result.append(row, ignore_index=True)
        result["Weekly is best"] = np.where(result["Week MAE"] < result["Yesterday MAE"], True, False)
        result.to_csv("supply_yesterday_vs_week.csv", index=False, float_format="%g")
    week_is_best = result[result["Weekly is best"] == True]
    print("Week is best in {:.2f}% of the cases".format(100 * len(week_is_best) / len(result)))


def explore_demand_period():
    start = dt(2019, 6, 12)
    end = dt(2019, 6, 25)
    hist = get_data(start - timedelta(days=180), start - timedelta(days=1), ["Curve Demand"], os.getcwd(), "d")
    hist_and_period = get_data(start - timedelta(days=180), end, ["Curve Demand"], os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    plt.plot(hist_and_period["Date"], hist_and_period["Curve Demand"], color=sec_color)
    plt.plot(hist["Date"], hist["Curve Demand"], color=first_color)
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    dec_hist = get_decomp(hist)
    dec_hist_per = get_decomp(hist_and_period)
    plt.plot(dec_hist_per["Date"], dec_hist_per["Trend"], color=sec_color)
    plt.plot(dec_hist["Date"], dec_hist["Trend"], color=first_color)
    plt.show()
    plt.close()


def get_decomp(df):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(df["Curve Demand"], model='multiplicative', period=7)
    df["Factor"] = decomp.seasonal
    df["Trend"] = decomp.trend
    df["Trend"] = df["Trend"].interpolate()
    df['Trend'] = df.apply(
        lambda row: row["Curve Demand"] / row['Factor'] if np.isnan(row['Trend']) else row['Trend'], axis=1)
    return df


def explore_temp_and_price():
    from statsmodels.tsa.seasonal import seasonal_decompose
    start = dt(2019, 1, 25)
    end = dt(2019, 2, 7)
    window = 180
    prev_day = start - timedelta(days=1)
    columns = ["System Price", "Curve Demand", "Temp Norway"]
    hist = get_data(start - timedelta(days=window), prev_day, columns, os.getcwd(), "d")
    hist_hor = get_data(start - timedelta(days=window), end, columns, os.getcwd(), "d")
    for col in columns:
        plt.subplots(figsize=full_fig)
        hist_hor["Trend"] = seasonal_decompose(hist_hor[col], model='a', period=7, extrapolate_trend='freq').trend
        hist_hor["Shift"] = hist_hor["Trend"] - hist_hor["Trend"].shift(1)
        plt.plot(hist_hor["Date"], hist_hor["Shift"], label="Horizon {}".format(col))
        hist["Trend"] = seasonal_decompose(hist[col], model='a', period=7, extrapolate_trend='freq').trend
        hist["Shift"] = hist["Trend"] - hist["Trend"].shift(1)
        plt.plot(hist["Date"], hist["Shift"], label="Hist {}".format(col))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == '__main__':
    # explore_supply_outlier()
    # explore_demand_period()
    # explore_temp_and_price()
    model_ = CurveModel()
    periods_ = get_random_periods(1)
    periods_ = [(dt(2019, 1, 1), dt(2019, 1, 14))]
    run_model(model_, periods_)

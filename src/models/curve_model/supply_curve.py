import pandas as pd
import numpy as np
import os
from data.data_handler import get_data
from data.data_handler import get_auction_data
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from scipy import stats
import math
from datetime import datetime as dt
from datetime import timedelta

s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105, 165, 210]


def get_supply_curve(month, hour, weekend, weekly_mean, safe):
    prev_workdir = os.getcwd()
    os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
    index = int(weekend * 288 + (month-1) * 24 + hour)
    profiles = pd.read_csv(r"supply_week_coefficients.csv")
    profile = profiles.iloc[index, :]
    assert month == profile["Month"]
    assert hour == profile["Hour"]
    assert weekend == profile["Weekend"]
    profile = profile[3:]
    weekly_mean = weekly_mean[1:]
    df = pd.DataFrame()
    result = profile * weekly_mean
    if not safe:
        upper, lower = (1.15, 0.9)
    else:
        upper, lower = (1.05, 0.9)
    df["Result"] = result
    wv_df = get_water_values(df.copy())
    wv_df["Prices"] = wv_df.apply(lambda row: "s {}".format(int(row["Prices"])), axis=1)
    wv_df.index = wv_df["Prices"]
    wv_df = wv_df[["Volume"]]
    df = df.merge(wv_df, left_index=True, right_index=True, how="left")
    df = df.reset_index()
    df.loc[0, "Volume"] = df.loc[0, "Result"] * lower
    df.loc[len(df)-1, "Volume"] = df.loc[len(df)-1, "Result"] * upper
    df["Volume"] = df["Volume"].interpolate()
    os.chdir(prev_workdir)
    return df["Volume"]


def get_supply_curve_water_values(month, hour, weekend, weekly_mean, cur_row, last_week_row, wv_model, last_coal, safe):
    index = weekend * 288 + (month - 1) * 24 + hour
    profiles = pd.read_csv(r"supply_week_coefficients.csv")
    profile = profiles.iloc[index, :]
    assert month == profile["Month"]
    assert hour == profile["Hour"]
    assert weekend == profile["Weekend"]
    profile = profile[3:]
    weekly_mean = weekly_mean[1:]
    df = pd.DataFrame()
    result = profile * weekly_mean
    if not safe:
        return result
    else:
        df["Result"] = result
        wv_df = get_water_values(df.copy())
        wv_df["Prices"] = wv_df.apply(lambda row: "s {}".format(int(row["Prices"])), axis=1)
        wv_df.index = wv_df["Prices"]
        wv_df = wv_df[["Volume"]]
        est_wv = estimate_wv(wv_df.copy())
        # expected_wv = est_wv + get_expected_water_value(hydro_prec_row, month, wv_df)
        exp_wv_diff = get_expected_water_value_2(cur_row, last_week_row, wv_model, last_coal)
        lower_p, upper_p = get_neighbour_prices(wv_df)
        lower_class, upper_class = ("s {}".format(lower_p), "s {}".format(upper_p))
        wv_df["P"] = wv_df.index.str[2:].astype(int) + exp_wv_diff
        wv_df.loc[lower_class, "Volume"] = df.loc[lower_class, "Result"]
        wv_df.loc[upper_class, "Volume"] = df.loc[upper_class, "Result"]
        wv_df.loc[lower_class, "P"] = lower_p
        wv_df.loc[upper_class, "P"] = upper_p
        wv_df["Class"] = wv_df.index.str[2:].astype(int)
        wv_df = wv_df.sort_values(by="Class")
        wv_df = wv_df.reset_index()
        wv_df.loc[0, "P"] = wv_df["P"].min()
        wv_df.loc[len(wv_df)-1, "P"] = wv_df["P"].max()
        s_line = LineString(np.column_stack((wv_df["Volume"], wv_df["P"])))
        wv_df["New volume"] = np.NAN
        min_vol = wv_df.loc[0, "Volume"]
        max_vol = wv_df.loc[len(wv_df)-1, "Volume"]
        for i in range(len(wv_df)):
            h_line = LineString([(min_vol, wv_df.loc[i, "Class"]), (max_vol, wv_df.loc[i, "Class"])])
            inter = h_line.intersection(s_line)
            if type(inter) == MultiPoint or type(inter) == LineString:
                print(s_line.coords.xy[0])
                print(s_line.coords.xy[1])
                print(h_line.coords.xy[0])
                print(h_line.coords.xy[1])
                print("Max {}, min {}".format(max_vol, min_vol))
                print(wv_df)
            wv_df.loc[i, "New volume"] = h_line.intersection(s_line).x
        wv_df.index = wv_df["Prices"]
        wv_df = wv_df[["New volume"]]
        wv_df = wv_df.rename(columns={"New volume": "Volume"})
        df = df.merge(wv_df, left_index=True, right_index=True, how="left")
        df = df.reset_index()
        df.loc[0, "Volume"] = df.loc[0, "Result"] * 0.95
        df.index = s_classes
        df["Volume"] = df["Volume"].interpolate(limit_area="inside", method="index")
        df = df.reset_index(drop=True)
        df.loc[len(df)-1, "Volume"] = df.loc[len(df)-1, "Result"] * 1.1
        df["Volume"] = df["Volume"].interpolate(limit_area="inside", method="linear")
        return df["Volume"]


def get_neighbour_prices(wv_df):
    for i in range(len(s_classes)):
        c = s_classes[i]
        if "s {}".format(c) in wv_df.index:
            lower_p = s_classes[i-1]
            break
    for i in range(len(s_classes)-1, 0, -1):
        c = s_classes[i]
        if "s {}".format(c) in wv_df.index:
            upper_p = s_classes[i+1]
            break
    return lower_p, upper_p



def get_expected_water_value(hydro_prec, month, df):
    next_month = month + 1 if month != 12 else 1
    last_month = month - 1 if month != 1 else 12
    df = df[df["Month"].isin([last_month, month, next_month])].reset_index(drop=True)
    normalised = df[["Total Hydro Dev", "Prec Norway 7"]]
    normalised = normalised.append(hydro_prec[["Total Hydro Dev", "Prec Norway 7"]], ignore_index=True)
    normalized_df = (normalised - normalised.min()) / (normalised.max() - normalised.min())
    cur_row = normalized_df.iloc[-1:, :].reset_index(drop=True)
    print(cur_row)
    normalized_df = normalized_df.iloc[0:-1, :].reset_index(drop=True)
    for col in normalized_df.columns:
        normalized_df[col] = abs(normalized_df[col] - cur_row.loc[0, col])
    normalized_df["Score"] = normalized_df.sum(axis=1)
    normalized_df = normalized_df.sort_values(by="Score").head(20)
    similar = df.loc[normalized_df.index]
    print(similar)
    print("Mean wv {:.2f}".format(similar["Water Value"].mean()))
    assert False


def get_expected_water_value_2(this_row, last_week_row, wv_model, last_coal):
    last_hydro = last_week_row.head(1)["Total Hydro Dev"].values[0]
    cur_hydro = this_row.head(1)["Total Hydro Dev"].values[0]
    hydro_diff = cur_hydro - last_hydro
    cur_coal = this_row.head(1)["Coal"].values[0]
    coal_diff = cur_coal - last_coal
    season = this_row.head(1)["Season"].values[0]
    x = np.array([season, hydro_diff, coal_diff]).reshape(1, -1)
    y = wv_model.predict(x)[0]
    # print("Last coal: {}, coal diff {}, y: {}".format(last_coal, coal_diff, y))
    return y




def estimate_wv(df):
    df = df.reset_index()
    df["Prices"] = df.apply(lambda row: int(row["Prices"][2:]), axis=1)
    supply_line = LineString(np.column_stack((df["Volume"], df["Prices"])))
    mid_volume = 1 / 2 * (df.loc[0, "Volume"] + df.loc[len(df) - 1, "Volume"])
    mid_line = LineString([(mid_volume, -10), (mid_volume, 210)])
    wv = supply_line.intersection(mid_line).y
    return wv


def get_water_values(df):
    df = df.rename(columns={"Result": "Volume"})
    df["Prices"] = s_classes
    df = df.reset_index(drop=True)
    df["P Diff"] = df["Prices"] - df["Prices"].shift(1)
    df["Diff"] = df["Volume"] - df["Volume"].shift(1)
    df.loc[0:4, "Diff"] = np.NAN
    df.loc[len(df)-3:len(df)-1, "Diff"] = np.NAN
    df["Derivative"] = df["Diff"] / df["P Diff"]
    #print(df)
    df["Derivative"] = df.apply(lambda row: np.NAN if row["Derivative"] <= 230 else row["Derivative"], axis=1)
    keep_idx = []
    for i in range(len(df)):
        if not np.isnan(df.loc[i, "Derivative"]):
            keep_idx.append(i)
    #print(df)
    keep_idx = check_if_one_class_misses(keep_idx)
    df = df.loc[keep_idx]
    keep_idx = get_longest_index_flow(df)
    df = df.loc[keep_idx].reset_index(drop=True)
    return df


def check_if_one_class_misses(keep_idx):
    full_sequence = range(keep_idx[0], keep_idx[-1]+1)
    number_of_missing = len([a for a in full_sequence if a not in keep_idx])
    if number_of_missing == 1:
        return [a for a in full_sequence]
    else:
        return keep_idx


def get_longest_index_flow(df):
    df = df.reset_index()
    options = []
    cur_option = []
    for i in range(len(df)-1):
        if len(cur_option) == 0:
            cur_option.append(df.loc[i, "index"])
        if df.loc[i+1, "index"] == df.loc[i, "index"] + 1:
            cur_option.append(df.loc[i+1, "index"])
        else:
            options.append(cur_option)
            cur_option = []
        if i == len(df) - 2:
            options.append(cur_option)
    longest_flow = []
    for flow in options:
        if len(flow) >= len(longest_flow):
            longest_flow = flow
    return longest_flow


def make_weekday_supply_profile():
    days = get_data("07.07.2014", "29.12.2019", ["Weekend", "Week", "Month"], os.getcwd(), "h")
    bids = get_auction_data("07.07.2014", "29.12.2019", "s", os.getcwd())
    df = days.merge(bids, on=["Date", "Hour"])
    res_weekday = pd.DataFrame(columns=["Month", "Hour"] + ["s {}".format(i) for i in s_classes])
    res_weekend = pd.DataFrame(columns=res_weekday.columns)
    start_week = 52*2014+28
    df["Week no"] = df["Date"].dt.year*52 + df["Week"] - start_week
    last_week = int(df.loc[len(df) - 1, "Week no"])
    for week in range(last_week):
        print("Week {}".format(week))
        data = df[df["Week no"] == week]
        cur_month = int(data["Month"].median())
        for hour in range(24):
            h_data = data[data["Hour"] == hour].reset_index(drop=True)
            week_mean = h_data.iloc[:, 5:-1].mean(axis=0)
            res_weekday = res_weekday.append(get_factors(h_data, 0, cur_month, week_mean, hour), ignore_index=True)
            res_weekend = res_weekend.append(get_factors(h_data, 1, cur_month, week_mean, hour), ignore_index=True)
    res_weekday = res_weekday.groupby(by=["Month", "Hour"]).mean().reset_index()
    res_weekday["Weekend"] = 0
    res_weekend = res_weekend.groupby(by=["Month", "Hour"]).mean().reset_index()
    res_weekend["Weekend"] = 1
    df = res_weekday.append(res_weekend, ignore_index=True)
    df = df[["Month", "Hour", "Weekend"] + ["s {}".format(i) for i in s_classes]]
    df.to_csv("supply_week_coefficients.csv", index=False, float_format="%g")


def get_factors(df, i, month, week_mean, hour):
    mean = df[df["Weekend"] == i].iloc[:, 5:-1].mean(axis=0)
    factors = mean / week_mean
    factors["Month"] = month
    factors["Hour"] = hour
    return factors


def make_water_value_profile():
    df = pd.read_csv("../../../data/output/auction/water_values.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df[["Date", "Water Value", "Total Hydro Dev", "Prec Norway 7"]]
    start = df.loc[0, "Date"].date()
    end = df.loc[len(df)-1, "Date"].date()
    data = get_data(start, end, ["Month"], os.getcwd(), "d")
    df = df.merge(data, on="Date")
    import matplotlib.pyplot as plt
    plt.subplots(figsize=(13, 7))
    plt.plot(df["Date"], df["Water Value"])
    #plt.show()
    plt.close()
    df = df[["Date", "Month", "Total Hydro Dev", "Prec Norway 7", "Water Value"]]
    df.to_csv("water_value_hydro_dev_profiles.csv", index=False, float_format="%g")


def make_water_value_profile_2():
    df = pd.read_csv("../../../data/output/auction/water_values.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df[["Date", "Water Value", "Total Hydro Dev"]]
    df = df.dropna().reset_index(drop=True)
    start = df.loc[0, "Date"].date()
    end = df.loc[len(df)-1, "Date"].date()
    data = get_data(start, end, ["Month"], os.getcwd(), "d")
    df = df.merge(data, on="Date")
    result = pd.DataFrame(columns=["Date", "Month", "Hydro From", "Hydro Diff", "WV from", "WV Diff"])
    for i in range(len(df)-7):
        cur_dev = df.loc[i, "Total Hydro Dev"]
        hydro_diff = df.loc[i+7, "Total Hydro Dev"] - cur_dev
        cur_wv = df.loc[i, "Water Value"]
        wv_diff = df.loc[i+7, "Water Value"] - cur_wv
        row = {"Month": df.loc[i+7, "Month"], "Hydro From": cur_dev, "Hydro Diff": hydro_diff, "WV from": cur_wv,
               "WV Diff": wv_diff, "Date": df.loc[i+7, "Date"].date()}
        result = result.append(row, ignore_index=True)
    r_coeff = round(stats.pearsonr(result["Hydro Diff"], result["WV Diff"])[0], 3)
    print("R coeff {}".format(r_coeff))
    result.to_csv("water_value_hydro_dev_profiles_2.csv", index=False, float_format="%g")


def decision_tree_model():
    import sklearn.tree as tree
    from sklearn.tree import export_text
    import pickle
    wv = pd.read_csv("../../../data/output/auction/water_values.csv")
    wv["Date"] = pd.to_datetime(wv["Date"], format="%Y-%m-%d")
    df = get_data("01.07.2014", "02.06.2019", ["Season", "Coal"], os.getcwd(), "d")
    df = df.merge(wv, on="Date")
    df = df[["Date", "Season", "Total Hydro Dev", "Water Value", "Coal"]]
    snow = pd.read_csv("../../../data/input/ninja/snow_temp_norway_2014_2019.csv", usecols=["Date", "snow_mass"])
    snow["Date"] = pd.to_datetime(snow["Date"], format="%Y-%m-%d")
    df = df.merge(snow, on="Date")
    for i in range(7, len(df)):
        df.loc[i, "Hydro Change"] = df.loc[i, "Total Hydro Dev"] - df.loc[i-7, "Total Hydro Dev"]
        df.loc[i, "WV From"] = df.loc[i-7, "Water Value"]
        df.loc[i, "WV Change"] = df.loc[i, "Water Value"] - df.loc[i, "WV From"]
        df.loc[i, "Coal Change"] = df.loc[i, "Coal"] - df.loc[i-7, "Coal"]
    df = df.dropna()
    y = df[["WV Change"]].values
    x_cols = ["Season", "Hydro Change", "Coal Change"]
    x = df[x_cols].values
    model = tree.DecisionTreeRegressor(random_state=1, max_depth=6)
    model.fit(x, y)
    tree_rules = export_text(model, feature_names=x_cols)
    print(tree_rules)
    print(model.feature_importances_)
    print("Depth {}, leaves {}, and R^2 score {}".format(model.get_depth(), model.get_n_leaves(), model.score(x, y)))
    #pickle.dump(model, open("wv_model.pickle", 'wb'))


if __name__ == '__main__':
    print("running method")
    # make_weekday_supply_profile()
    #make_water_value_profile()
    # make_water_value_profile_2()
    #get_supply_curve(1, 0, 0, None)
    #get_supply_curve(1, 0, 1, None)
    decision_tree_model()

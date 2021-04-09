# script connecting src and data folder.
import datetime
import pandas as pd
import numpy as np
import os


def get_data(from_date, to_date, column_list, work_dir, resolution):  # dates on form "dd.mm.yyyy", columns in list ["System Price", ...]
    path, first_date, last_date = get_dates_and_path(from_date, to_date, work_dir, resolution, auction=False)
    if resolution == "h":
        columns = ["Date", "Hour"] + column_list
    elif resolution == "d":
        columns = ["Date"] + column_list
    all_data = pd.read_csv(path, usecols=columns)
    all_data = all_data[columns]  # order same as input
    all_data["Date"] = pd.to_datetime(all_data["Date"], format='%Y-%m-%d')
    mask = (all_data['Date'] >= first_date) & (all_data['Date'] <= last_date)
    df = all_data.loc[mask]
    df = df.reset_index(drop=True)
    return df


def get_auction_data(from_date, to_date, curves, work_dir):
    path, first_date, last_date = get_dates_and_path(from_date, to_date, work_dir, None, auction=True)
    demand_classes = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
    supply_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66,
                      75, 105, 165, 210]
    all_dates = pd.read_csv(path, usecols=["Date", "Hour"])
    all_dates["Date"] = pd.to_datetime(all_dates["Date"], format='%Y-%m-%d')
    mask = (all_dates['Date'] >= first_date) & (all_dates['Date'] <= last_date)
    dates_df = all_dates.loc[mask]
    first_index = dates_df.index[0]
    length = len(dates_df)
    columns = ["Date", "Hour"]
    if "d" in curves:
        columns += ["d {}".format(price) for price in demand_classes]
    if "s" in curves:
        columns += ["s {}".format(price) for price in supply_classes]
    bids_df = pd.read_csv(path, usecols=columns, header=0, skiprows=range(1, first_index), nrows=length+1)
    bids_df = bids_df[columns]
    bids_df["Date"] = pd.to_datetime(bids_df["Date"], format='%Y-%m-%d')
    mask = (bids_df['Date'] >= first_date) & (bids_df['Date'] <= last_date)
    df = bids_df.loc[mask]
    df = df.reset_index(drop=True)
    return df


def get_dates_and_path(from_date, to_date, work_dir, resolution, auction):
    path_to_project = "/".join(work_dir.split("\\")[0:5])
    if auction:
        path_to_all_data = "/data/input/auction/time_series.csv"
    else:
        res = "hourly" if resolution == "h" else "daily"
        path_to_all_data = "/data/input/combined/all_data_{}.csv".format(res)
    path = path_to_project + path_to_all_data
    if isinstance(from_date, str):
        first_date = datetime.datetime.strptime(from_date, '%d.%m.%Y')
    else:
        first_date = datetime.datetime(from_date.year, from_date.month, from_date.day)
    if isinstance(to_date, str):
        last_date = datetime.datetime.strptime(to_date, '%d.%m.%Y')
    else:
        last_date = datetime.datetime(to_date.year, to_date.month, to_date.day)
    if auction:
        if last_date > datetime.datetime(2020, 6, 2):
            print("Last date must be before 2nd july 2020")
            assert False
    valid_dates = check_valid_dates(first_date, last_date)
    if valid_dates:
        return path, first_date, last_date


def check_valid_dates(from_date, to_date):  # helping method checking that dates are valid
    valid = True
    if isinstance(from_date, str) and isinstance(to_date, str):
        try:
            first_date = datetime.datetime.strptime(from_date, '%d.%m.%Y')
            last_date = datetime.datetime.strptime(to_date, '%d.%m.%Y')
            if last_date < first_date:
                print("First date must come before last date")
                valid = False
        except Exception as e:
            print(e)
    else:
        if from_date > to_date:
            print("First date must come before last date")
            valid = False
    return valid


def get_path_to_all_data(work_dir, resolution):
    path_to_project = "/".join(work_dir.split("\\")[0:5])
    if resolution == "h":
        path_to_all_data = "/data/input/combined/all_data_hourly.csv"
    elif resolution == "d":
        path_to_all_data = "/data/input/combined/all_data_daily.csv"
    else:
        print("Resolution must be d or h")
        assert False
    path = path_to_project + path_to_all_data
    return path


if __name__ == '__main__':
    date_string_from = "01.01.2019"
    date_string_do = "31.12.2019"
    col = ["System Price"]
    data = get_data(date_string_from, date_string_do, col, os.getcwd())
    print(data)

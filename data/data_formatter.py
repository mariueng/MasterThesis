from distutils.command.config import config
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import math
from data_handler import get_data
from src.preprocessing.arcsinh.arcsinh import to_arcsinh


def combine_hour_day_month_year_to_datetime_index(df):
    """
    This method reduces columns containing different time information such as hour, day, month and year into a single
    column and sets it as index of the dataframe. NB! Assumes the column/index 'Date' exists.
    TODO: implement check/methods for whether index is 'Date'
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    cols = []
    if 'Hour' in df.columns:
        cols.append('Hour')
        df['Date'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
    if 'Day' in df.columns:
        cols.append('Day')
        df['Date'] = df['Date'] + pd.to_timedelta(df['Day'], unit='d')
    if 'Month' in df.columns:
        cols.append('Month')
        df['Date'] = df['Date'] + pd.to_timedelta(df['Month'], unit="m")
    if 'Year' in df.columns:
        cols.append('Year')
    if not 'Date' == df.index.name:
        # Do this
        code = 2
    df.drop(cols, axis=1, inplace=True)
    df.set_index('Date', inplace=True)
    return df


# helping method reformatting hour column
def reformat_hour_column(df, date_sep):
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H').dt.time
    format = "%d{}%m{}%Y".format(date_sep, date_sep)
    df['Date'] = pd.to_datetime(df['Date'], format=format)
    return df


def rename_column_names(df, col_name_changes):
    for from_name, to_name in col_name_changes.items():
        df.rename({from_name: to_name}, axis=1, inplace=True)
    return df


def convert_folder_to_csv(paths, replace_commas, header_row, resolution, make_integer):
    for p in paths:
        p_list = str(p).split("\\")
        del p_list[-2]  # remove raw from the list
        p_list[-1] = p_list[-1][:-3] + "csv"
        out_path = "\\".join(p_list)
        df = pd.read_html(str(p), thousands=None, header=header_row, encoding="utf-8")[0]
        if replace_commas:
            # df = df.apply(lambda x: x.str.replace(',', '.'))
            for y in df.columns:
                if (df[y].dtype == object):
                    df[y] = df[y].apply(lambda x: str(x).replace(',', '.'))
        col_name_changes = {}
        if resolution == "h":
            col_name_changes = {'Unnamed: 0': 'Date', 'Hours': 'Hour'}
            int_columns_from = 2
        elif resolution == "d":
            col_name_changes = {'Unnamed: 0': 'Date'}
            int_columns_from = 1
        elif resolution == "w":
            col_name_changes = {'Unnamed: 0': 'Date'}
            int_columns_from = 2
        df = rename_column_names(df, col_name_changes)
        if make_integer:
            df = make_all_columns_integers_without_date(df, int_columns_from)
        df.to_csv(out_path, sep=",", index=False, encoding="utf-8")
        print("Saved " + out_path + " to file")


def get_all_csv_files_from_directory(path):
    result = []
    paths = sorted(Path(path).iterdir())  # list all files in directory
    for p in paths:
        if ".csv" in str(p):
            result.append(p)
    return result


def write_price_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution == "h":
        path = "..\\data\\input\\price_hourly"
        out_path = "..\\data\\input\\combined\\price_hourly.csv"
        columns = [0, 1, 2]  # date=0, hour=1, sys=2, (Tr.Heim=14)
    elif resolution == "d":
        path = "..\\data\\input\\price_daily"
        out_path = "..\\data\\input\\combined\\price_daily.csv"
        columns = [0, 1]  # date=0, sys=1, (Tr.Heim=13)
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=False)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, usecols=columns)
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
    col_name_changes = {'SYS': 'System Price'}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all.to_csv(out_path, index=False)


# helping method
def aggregate_markets_volume(df):
    market_to_columns = {
        'NO Buy Vol': ["NO1 Buy", "NO2 Buy", "NO3 Buy", "NO4 Buy", "NO5 Buy"],
        'NO Sell Vol': ["NO1 Sell", "NO2 Sell", "NO3 Sell", "NO4 Sell", "NO5 Sell"],
        'SE Buy Vol': ["SE1 Buy", "SE2 Buy", "SE3 Buy", "SE4 Buy"],
        'SE Sell Vol': ["SE1 Sell", "SE2 Sell", "SE3 Sell", "SE4 Sell"],
        'FI Buy Vol': ["FI Buy"],
        'FI Sell Vol': ["FI Sell"],
        'DK Buy Vol': ["DK1 Buy", "DK2 Buy"],
        'DK Sell Vol': ["DK1 Sell", "DK2 Sell"],
        "Nordic Buy Vol": ["NO Buy Vol", "SE Buy Vol", "FI Buy Vol", "DK Buy Vol"],
        "Nordic Sell Vol": ["NO Sell Vol", "SE Sell Vol", "FI Sell Vol", "DK Sell Vol"],
        'Baltic Buy Vol': ["EE Buy", "LV Buy", "LT Buy"],
        'Baltic Sell Vol': ["EE Sell", "LV Sell", "LT Sell"]
    }
    for market in market_to_columns.keys():
        df[market] = df[market_to_columns[market]].sum(axis=1)
    return df


def write_volume_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution == "h":
        path = "..\\data\\input\\volume_hourly"
        out_path = "..\\data\\input\\combined\\volume_hourly.csv"
    elif resolution == "d":
        path = "..\\data\\input\\volume_daily"
        out_path = "..\\data\\input\\combined\\volume_daily.csv"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=False)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
    col_name_changes = {'Turnover at system price': 'Total Vol'}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all = aggregate_markets_volume(df_all)
    incl_columns = ["Date", "Total Vol", "NO Buy Vol", "NO Sell Vol", "SE Buy Vol", "SE Sell Vol", "DK Buy Vol",
                    "DK Sell Vol", "FI Buy Vol", "FI Sell Vol", "Nordic Buy Vol", "Nordic Sell Vol", "Baltic Buy Vol",
                    "Baltic Sell Vol"]
    if resolution == "h":
        incl_columns.insert(1, "Hour")
    df_all = df_all[incl_columns]
    df_all.to_csv(out_path, index=False, float_format='%g')


def fix_error_in_year_hydro(df):
    number_of_weeks = df.shape[0]
    print(df.head())
    print(number_of_weeks)
    for i in range(0, number_of_weeks - 1):
        week = df.iloc[i, 0]
        next_week = df.iloc[i + 1, 0]
        year = int(week[-2:])
        next_year = int(next_week[-2:])
        difference = next_year - year
        if difference == 2:
            correct_year = year + 1
            correct_string = next_week[:-2] + str(correct_year)
            df.iloc[i + 1, df.columns.get_loc('Date')] = correct_string
    return df


# combine weekly hydro input to one csv file
def write_hydro_all_weekly(convert_to_csv, replace_commas):
    path = "..\\data\\input\\hydro_weekly"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution="w", make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        if "hydro_weekly.csv" not in str(p):
            df_year = pd.read_csv(p, sep=",")
            df_years.append(df_year)
    df_all = pd.concat(df_years, ignore_index=True)
    col_name_changes = {"NO": "NO Hydro", "SE": "SE Hydro", "FI": "FI Hydro"}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all = df_all.interpolate(method='linear', axis=0).ffill().bfill()  # affects only two last rows of 2019
    df_all = df_all[50:]  # from the last week of 2013
    df_all = fix_error_in_year_hydro(df_all)
    df_all.to_csv("..\\data\\input\\hydro_weekly\\hydro_weekly.csv", index=False, float_format='%g')


def interpolate_week(first, last, week_dates):
    values = np.linspace(first, last, 8).round(2)
    date_values = {}
    for i in range(len(week_dates)):
        date_values[np.datetime64(week_dates[i])] = values[i]
    return date_values


# combine hydro input to one csv file
def write_hydro_daily_to_combined():
    path = "..\\data\\input\\hydro_weekly\\hydro_weekly.csv"
    df_weekly = pd.read_csv(path)
    df_daily = pd.DataFrame(columns=['Date', 'NO Hydro', 'SE Hydro', 'FI Hydro'])
    dates = pd.date_range(start="2014-01-01", end="2021-01-01")
    for date in dates:
        df_daily = df_daily.append({'Date': date}, ignore_index=True)
    number_of_weeks = df_weekly.shape[0]
    for i in range(0, number_of_weeks - 1):
        week_row = df_weekly.iloc[i, :]
        next_week_row = df_weekly.iloc[i + 1, :]
        week_year = week_row[0]
        week = "W" + str(int(week_year[:2]))
        year = str(2000 + int(week_year[-2:]))
        minus_one_years = ["2014", "2015", "2019",
                           "2020"]  # Se hydro weekly data. Confusion with weeks. Some years starts at 0, some at 1.
        if year in minus_one_years:
            week = "W" + str(int(week_year[:2]) - 1)  # minus one. Weeks starts at 0 these years
        first_date = datetime.strptime(year + " " + week + " w1", "%Y W%W w%w").date()
        # print(
        #    "Year {}. W{}, firstdate {}, minus 1: {}".format(year, week_year[:2], first_date, year in minus_one_years))
        week_dates = [first_date + timedelta(days=x) for x in range(7)]
        no_date_values = interpolate_week(week_row[1], next_week_row[1], week_dates)
        se_date_values = interpolate_week(week_row[2], next_week_row[2], week_dates)
        fi_date_values = interpolate_week(week_row[3], next_week_row[3], week_dates)
        for date in week_dates:
            date = np.datetime64(date)
            after_start_date = df_daily["Date"] >= date
            before_end_date = df_daily["Date"] <= date
            between_two_dates = after_start_date & before_end_date
            filtered_date = df_daily.loc[between_two_dates]
            if len(filtered_date) == 1:
                index = filtered_date.index[0]
                df_daily.loc[index, "NO Hydro"] = no_date_values[date]
                df_daily.loc[index, "SE Hydro"] = se_date_values[date]
                df_daily.loc[index, "FI Hydro"] = fi_date_values[date]
    df_daily["Date"] = df_daily["Date"].dt.strftime("%d-%m-%Y")
    df_daily["Total Hydro"] = df_daily["NO Hydro"] + df_daily["SE Hydro"] + df_daily["FI Hydro"]
    df_daily["Total Hydro"] = pd.to_numeric(df_daily["Total Hydro"])
    df_daily.to_csv("..\\data\\input\\combined\\hydro_daily.csv", index=False, float_format='%g')


def write_hydro_hourly_to_combined():
    path = "..\\data\\input\\combined\\hydro_daily.csv"
    df_daily = pd.read_csv(path)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'], format="%d-%m-%Y")
    df_hourly = df_daily.set_index('Date').resample('H').interpolate('linear')
    df_hourly['Hour'] = [d.time() for d in df_hourly.index]
    df_hourly['Date'] = [d.date() for d in df_hourly.index]
    df_hourly["Date"] = pd.to_datetime(df_hourly["Date"], format="%Y-%m-%d")
    df_hourly["Date"] = df_hourly["Date"].dt.strftime("%d-%m-%Y")
    df_hourly['Hour'] = df_hourly["Hour"].apply(lambda x: x.hour)
    df_hourly = df_hourly[["Date", "Hour", "NO Hydro", "SE Hydro", "FI Hydro", "Total Hydro"]]
    df_hourly.drop(df_hourly.tail(1).index, inplace=True)
    df_hourly.to_csv("..\\data\\input\\combined\\hydro_hourly.csv", index=False, float_format='%g')


def plot_hydro(resolution):
    path = ""
    save_source_plot = ""
    if resolution == "w":
        path = "..\\data\\input\\hydro_weekly\\hydro_weekly.csv"
        save_source_plot = str("output\\plots\\hydro_week.png")
        title = "Hydro Week"
    elif resolution == "d":
        path = "..\\data\\input\\combined\\hydro_daily.csv"
        save_source_plot = str("output\\plots\\random\\hydro_day.png")
        title = "Hydro Day"
    elif resolution == "h":
        print("plot hourly hydro")
        path = "..\\data\\input\\combined\\hydro_hourly.csv"
        save_source_plot = str("output\\plots\\random\\hydro_hour.png")
        title = "Hydro Hour"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    plt.plot(df["Date"], df["NO Hydro"], label="NO")
    plt.plot(df["Date"], df["SE Hydro"], label="SE")
    plt.plot(df["Date"], df["FI Hydro"], label="FI")
    ax = plt.gca()
    ax.set_title(title, pad=20, size=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    ax.set(xlabel="Date",
           ylabel="Reservoir GWh")
    plt.tight_layout()
    plt.savefig(save_source_plot)
    plt.close()


def plot_hydro_dev(resolution):
    path = ""
    save_source_plot = ""
    if resolution == "d":
        print("plot daily hydro deviation..")
        path = "..\\data\\input\\combined\\hydro_dev_daily.csv"
        save_source_plot = str("output\\plots\\random\\hydro_dev_day.png")
        title = "Hydro Deviation Day"
    elif resolution == "h":
        print("plot hourly hydro deviation..")
        path = "..\\data\\input\\combined\\hydro_dev_hourly.csv"
        save_source_plot = str("output\\plots\\random\\hydro_dev_hour.png")
        title = "Hydro Deviation Hour"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    plt.plot(df["Date"], df["NO Hydro Dev"], label="NO")
    plt.plot(df["Date"], df["SE Hydro Dev"], label="SE")
    plt.plot(df["Date"], df["FI Hydro Dev"], label="FI")
    ax = plt.gca()
    ax.set_title(title, pad=20, size=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    ax.set(xlabel="Date",
           ylabel="Reservoir GWh")
    plt.tight_layout()
    plt.savefig(save_source_plot)
    plt.close()


def make_all_columns_integers_without_date(df, int_columns):
    cols = df.columns
    df[cols[int_columns:]] = df[cols[int_columns:]].apply(pd.to_numeric, errors='coerce')
    return df


def copy_last_days_hours_in_column(df, column):
    for i in range(len(df) - 1, 1, -1):
        if i > 23:
            value = df[column].iloc[i]
            if pd.isna(value):
                new_value = df[column].iloc[i - 24]
                df.at[i, column] = new_value
    return df


def write_supply_to_all_data(resolution, convert_to_csv, replace_commas):
    if resolution == "h":
        path = "..\\data\\input\\production_hourly"
        out_path = "..\\data\\input\\combined\\supply_hourly.csv"
        all_data = pd.read_csv("..\\data\\input\\combined\\all_data_hourly.csv.csv")
    elif resolution == "d":
        path = "..\\data\\input\\production_daily"
        out_path = "..\\data\\input\\combined\\supply_daily.csv"
        all_data = pd.read_csv("..\\data\\input\\combined\\all_data_daily.csv")
    all_data["Date"] = pd.to_datetime(all_data["Date"], format="%Y-%m-%d")
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
        df_all = copy_last_days_hours_in_column(df_all, "EE")  # affects only last 16 rows of 2019, column EE
        df_all["Supply"] = df_all["Nordic"] + df_all["Baltic"]
        df_all = df_all[["Date", "Hour", "Supply"]]
    elif resolution == "d":
        df_all['Nordic'] = df_all['Nordic'].fillna(df_all.iloc[:, 1:5].sum(axis=1))
        df_all['Baltic'] = df_all['Baltic'].fillna(df_all.iloc[:, 6:9].sum(axis=1))
        df_all["Supply"] = df_all["Nordic"] + df_all["Baltic"]
        df_all = df_all[["Date", "Supply"]]
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="%d-%m-%Y")
    assert [i == 0 for i in df_all.isna().sum().tolist()]
    df_all.to_csv(out_path, index=False, float_format='%g')
    assert len(df_all) == len(all_data)
    if resolution == "d":
        all_data = all_data.merge(df_all, on="Date", how="outer")
        assert [i == 0 for i in all_data.isna().sum().tolist()]
        all_data.to_csv("..\\data\\input\\combined\\all_data_daily.csv", index=False, float_format='%g')


def write_demand_to_all_data(resolution, convert_to_csv, replace_commas):
    if resolution == "h":
        path = "..\\data\\input\\consumption_hourly"
        out_path = "..\\data\\input\\combined\\consumption_hourly.csv"
        all_data = pd.read_csv("..\\data\\input\\combined\\all_data_hourly.csv")
    elif resolution == "d":
        path = "..\\data\\input\\consumption_daily"
        out_path = "..\\data\\input\\combined\\consumption_daily.csv"
        all_data = pd.read_csv("..\\data\\input\\combined\\all_data_daily.csv")
    all_data["Date"] = pd.to_datetime(all_data["Date"], format="%Y-%m-%d")
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="%d-%m-%Y")
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
        df_all = copy_last_days_hours_in_column(df_all, "EE")  # affects last 3 rows of 2019, column EE
        df_all['Baltic'] = df_all['EE'] + df_all['LV'] + df_all["LT"]
        df_all = copy_last_days_hours_in_column(df_all, "SE")  # affects only last 13 rows of 2019, column SE
        df_all['Nordic'] = df_all['NO'] + df_all['SE'] + df_all["FI"] + df_all["DK"]
        df_all['Total'] = df_all['Nordic'] + df_all['Baltic']
    elif resolution == "d":
        df_all['Nordic'] = df_all['Nordic'].fillna(df_all.iloc[:, 1:5].sum(axis=1))
        df_all['Baltic'] = df_all['Baltic'].fillna(df_all.iloc[:, 6:9].sum(axis=1))
        df_all["Demand"] = df_all["Nordic"] + df_all["Baltic"]
        df_all = df_all[["Date", "Demand"]]
    assert [i == 0 for i in df_all.isna().sum().tolist()]
    df_all.to_csv(out_path, index=False, float_format='%g')
    assert len(df_all) == len(all_data)
    if resolution == "d":
        all_data = all_data.merge(df_all, on="Date", how="outer")
        assert [i == 0 for i in all_data.isna().sum().tolist()]
        all_data.to_csv("..\\data\\input\\combined\\all_data_daily.csv", index=False, float_format='%g')


def combine_all_data(resolution):
    path = "..\\data\\input\\combined"
    all_paths = sorted(Path(path).iterdir())  # list all datasets paths
    paths = []
    for p in all_paths:
        if (resolution in str(p)) and ("all_data_" not in str(p)):
            paths.append(p)
    dfs = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        date_string = df.loc[0, "Date"].split("-")
        if len(date_string[0]) == 2:
            date_format = "%d-%m-%Y"
        else:
            date_format = "%Y-%m-%d"
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        dfs.append(df)
    for i in range(len(dfs) - 2, -1, -1):
        print("Merging in {}".format(paths[i]))
        if resolution == "daily":
            df = pd.merge(df, dfs[i], on='Date', how='outer')
        elif resolution == "hourly":
            df = pd.merge(df, dfs[i], on=['Date', 'Hour'], how='outer')
    if resolution == "daily":
        ordered_columns = ['Date', "System Price"]
        all_columns = df.columns.tolist()
        other_columns = [col for col in all_columns if col not in ordered_columns]
        ordered_columns.extend(other_columns)
        df = df[ordered_columns]
        df.to_csv("..\\data\\input\\combined\\all_data_daily.csv", index=False)
    elif resolution == "hourly":
        ordered_columns = ['Date', "Hour", "System Price"]
        all_columns = df.columns.tolist()
        other_columns = [col for col in all_columns if col not in ordered_columns]
        ordered_columns.extend(other_columns)
        df = df[ordered_columns]
        date_format = "%d-%m-%Y"
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        df.to_csv("..\\data\\input\\combined\\all_data_hourly.csv", index=False, float_format='%g')
    else:
        print("HAS TO BE HOURLY OG DAILY")
        assert False


def add_time_columns_to_all_data(resolution):
    if resolution == "d":
        data_path = "..\\data\\input\\combined\\all_data_daily.csv"
    else:
        data_path = "..\\data\\input\\combined\\all_data_hourly.csv"
    date_format = "%Y-%m-%d"
    df = pd.read_csv(data_path, sep=",")
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    for index, row in df.iterrows():
        date = row[0]
        week = datetime.date(date).isocalendar()[1]
        df.loc[index, "Week"] = week
        month = date.month
        df.loc[index, "Month"] = month
        season = math.floor(date.month / 3) + 1
        df.loc[index, "Season"] = season
        weekday = date.weekday() + 1
        df.loc[index, "Weekday"] = weekday
        if weekday in [1, 2, 3, 4, 5]:
            weekend = 0
        else:
            weekend = 1
        df.loc[index, "Weekend"] = weekend
    out_path = data_path
    df.to_csv(out_path, sep=",", index=False, float_format='%g')


def write_daily_hydro_deviations_to_combined():
    data_path = "..\\data\\input\\combined\\hydro_daily.csv"
    out_path = "..\\data\\input\\combined\\hydro_dev_daily.csv"
    out_columns = ["Date", "NO Hydro Dev", "SE Hydro Dev", "FI Hydro Dev", "Total Hydro Dev"]
    hydro_df = pd.read_csv(data_path, sep=",")
    date_format = "%d-%m-%Y"
    hydro_df['Date'] = pd.to_datetime(hydro_df['Date'], format=date_format)
    start = datetime(2014, 1, 1)
    end = datetime(2020, 12, 31)
    mask = (hydro_df['Date'] >= start) & (hydro_df['Date'] <= end)
    hydro_df = hydro_df.loc[mask].reset_index()
    average_year_df = hydro_df.groupby(
        [hydro_df["Date"].dt.month.rename("Month"), hydro_df["Date"].dt.day.rename("Day")]).mean()
    column_rename_dict = {"NO Hydro": "NO Mean", "SE Hydro": "SE Mean", "FI Hydro": "FI Mean",
                          "Total Hydro": "Total Mean"}
    average_year_df = rename_column_names(average_year_df, column_rename_dict)
    country_names = ["NO", "SE", "FI", "Total"]
    for index, row in hydro_df.iterrows():
        day = row["Date"].day
        month = row["Date"].month
        for country in country_names:
            mean_col_name = country + " Mean"
            day_row = average_year_df.loc[month, day]
            mean = day_row[mean_col_name]
            hydro = row[country + " Hydro"]
            dev = hydro - mean
            hydro_df.loc[index, country + " Hydro Dev"] = round(dev, 3)
    hydro_df = hydro_df[out_columns]
    hydro_df.to_csv(out_path, sep=",", index=False, float_format='%g')


def write_hourly_hydro_deviations_to_combined():
    daily_path = "..\\data\\input\\combined\\hydro_dev_daily.csv"
    day_df = pd.read_csv(daily_path)
    day_df["Date"] = pd.to_datetime(day_df["Date"], format="%Y-%m-%d")
    out_columns = ['Date', 'Hour', 'NO Hydro Dev', 'SE Hydro Dev', 'FI Hydro Dev', 'Total Hydro Dev']
    out_df = get_hourly_from_interpolated_daily(day_df, out_columns)
    out_path = "..\\data\\input\\combined\\hydro_dev_hourly.csv"
    out_df.to_csv(out_path, index=False, float_format='%g')


def get_hourly_from_interpolated_daily(day_df, out_columns):
    day_df = day_df.rename(columns={"Date": "DateTime"})
    out_df = pd.read_csv("..\\data\\input\\combined\\price_hourly.csv")
    out_df = out_df[["Date", "Hour"]]
    out_df["Date"] = pd.to_datetime(out_df["Date"], format="%d-%m-%Y")
    out_df["Hour"] = pd.to_datetime(out_df['Hour'], format="%H").dt.time
    out_df["DateTime"] = out_df.apply(lambda r: datetime.combine(r['Date'], r['Hour']), 1)
    out_df = out_df.merge(day_df, on="DateTime", how="outer")
    out_df = out_df.interpolate(method='linear', axis=0)
    out_df = out_df[out_columns]
    out_df['Hour'] = out_df['Hour'].apply(lambda x: x.hour)
    out_df = out_df.dropna()
    print(len(out_df))
    print(out_df)
    assert len(out_df) == 61375
    return out_df


def write_hourly_temperature_to_combined():
    daily_path = "..\\data\\input\\combined\\temp_daily.csv"
    day_df = pd.read_csv(daily_path)
    day_df["Date"] = pd.to_datetime(day_df["Date"], format="%Y-%m-%d")
    out_columns = ["Date", "Hour", "T Hamar", "T Krsand", "T Namsos", "T Troms", "T Bergen", "T Nor"]
    out_df = get_hourly_from_interpolated_daily(day_df, out_columns)
    out_path = "..\\data\\input\\combined\\temp_hourly.csv"
    out_df.to_csv(out_path, index=False, float_format='%g')


def remove_summer_winter_time():
    print("Fixing 'bugs' from summer time and winter time change")
    data_path = "..\\data\\input\\combined\\all_data_hourly.csv"
    fix_summer_and_winter_time(data_path)


def fix_summer_and_winter_time(data_path):
    date_format = "%Y-%m-%d"
    df = pd.read_csv(data_path, sep=",").dropna()
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    to_summer = {2014: 30, 2015: 29, 2016: 27, 2017: 26, 2018: 25, 2019: 31, 2020: 29}
    to_summer_dates = []
    to_winter = {2014: 26, 2015: 25, 2016: 30, 2017: 29, 2018: 28, 2019: 27, 2020: 25}
    to_winter_dates = []
    for year in range(2014, 2021):
        to_summer_dates.append(datetime(year, 3, to_summer[year]).date())
        to_winter_dates.append(datetime(year, 10, to_winter[year]).date())
    for index, row in df.iterrows():
        if row["Date"] in to_summer_dates and row["Hour"] == 2:
            print(index)
            print(row)
            prev_row = df.iloc[index - 1]
            next_row = df.iloc[index + 1]
            for col in df.columns[2:]:
                past_val = prev_row[col]
                next_val = next_row[col]
                value = round((past_val + next_val) / 2, 2)
                df.loc[index, col] = value
    years_covered = []
    indices_to_delete = []
    for index, row in df.iterrows():
        if row["Date"] in to_winter_dates and row["Hour"] == 2:
            year = row["Date"].year
            if year not in years_covered:
                years_covered.append(year)
            else:
                indices_to_delete.append(index)
    df = df.drop(indices_to_delete)
    df.to_csv(data_path, sep=",", index=False, float_format='%g')
    print("Date updated and saved to {}".format(data_path))


def write_daily_temperature_to_combined():
    coordinates = {
        "Temp Hamar": "SN12290",
        "Temp Krsand": "SN39100",
        "Temp Namsos": "SN72710",
        "Temp Troms": "SN91500",
        "Temp Bergen": "SN50500"
    }
    element = 'mean(air_temperature P1D)'
    time_offset = "PT0H"
    tol_col = "Temp Norway"
    tot_operation = "mean"
    out_file = "temp_daily"
    write_weather_df_to_combined_helping_method(element, time_offset, coordinates, tol_col, tot_operation, out_file)


def write_daily_precipitation_to_combined():
    codes = pd.read_csv("input/frost/frost_precip.csv", usecols=["Frost sted", "Frost kode"])
    coordinates = dict(zip(codes["Frost sted"], codes["Frost kode"]))
    element = 'sum(precipitation_amount P1D)'
    time_offset = "PT6H"
    tot_col = "Prec Norway"
    tot_operation = "add"
    out_file = "precip_daily_2"
    write_weather_df_to_combined_helping_method(element, time_offset, coordinates, tot_col, tot_operation, out_file)


def write_weather_df_to_combined_helping_method(element, time_offset, coordinates, tot_name, tot_op, out_file):
    from datetime import datetime as dt
    import requests
    years = [y for y in range(2014, 2021)]
    client_id = 'f2acda91-356a-4475-b815-17214a0c7f14'
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    weather_df = pd.DataFrame()
    for year in years:
        print("Year: {}".format(year))
        start = dt(year, 1, 1).date()
        end = dt(year, 12, 31).date()
        locations = ",".join(coordinates.values())
        parameters = {
            'sources': locations,
            'elements': element,
            'referencetime': '{}-01-01/{}-1-1'.format(year, year + 1),
        }
        r = requests.get(endpoint, parameters, auth=(client_id, ''))
        json = r.json()
        data = None
        if r.status_code == 200:
            data = json['data']
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json['error']['message'])
            print('Reason: %s' % json['error']['reason'])
        df = pd.DataFrame()
        for i in range(len(data)):
            row = pd.DataFrame(data[i]['observations'])
            row['referenceTime'] = data[i]['referenceTime']
            row['sourceId'] = data[i]['sourceId']
            df = df.append(row)
        df = df.reset_index()
        columns = ['sourceId', 'referenceTime', 'value', 'timeOffset']
        df = df[columns]
        df = df.rename(columns={"referenceTime": "Date"})
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df[df["timeOffset"].isin([time_offset])]
        dates = pd.date_range(start, end, freq='d')
        year_df = pd.DataFrame()
        year_df["Date"] = dates
        for city in coordinates.keys():
            id_number = [coordinates[city] + ":0"]
            sub_df = df[df["sourceId"].isin(id_number)]
            mask = (sub_df['Date'] >= start) & (sub_df['Date'] <= end)
            sub_df = sub_df.loc[mask].reset_index()
            sub_df = sub_df[["Date", "value"]]
            sub_df = sub_df.rename(columns={"value": city})
            sub_df['Date'] = pd.to_datetime(sub_df['Date'])
            year_df = pd.merge(year_df, sub_df, on="Date", how="outer")
        weather_df = weather_df.append(year_df, ignore_index=True)
    if tot_op == "mean":
        weather_df[tot_name] = weather_df[[y for y in coordinates.keys()]].mean(axis=1)
    elif tot_op == "add":
        weather_df[tot_name] = weather_df[[y for y in coordinates.keys()]].sum(axis=1)
    weather_df = weather_df.fillna(method="bfill")
    weather_df.to_csv("input/combined/{}.csv".format(out_file), index=False, float_format='%g')
    print("Number of nan rows: {}".format(weather_df.isna().sum()))


def append_weather_to_all_data():
    path_ = "..\\data\\input\\combined\\"
    df_all = pd.read_csv(path_ + "all_data_daily.csv", sep=",")
    df_all = df_all.drop(columns=["Prec Hamar", "Prec Kvin", "Prec Oppdal", "Prec Troms", "Prec Bergen", "Prec Norway", "Prec Norway 7"])
    #df_temp = pd.read_csv(path_ + "temp_daily.csv", sep=",")
    df_prec = pd.read_csv(path_ + "precip_daily.csv", sep=",")
    for df in [df_all, df_prec]:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    orig_len = len(df_all)
    #df_all = df_all.merge(df_temp, on="Date", how="outer")
    df_all = df_all.merge(df_prec[["Date", "Prec Norway"]], on="Date", how="outer")
    new_len = len(df_all)
    assert orig_len == new_len
    df_all.to_csv(path_ + "all_data_daily.csv", index=False, float_format='%g')


def add_prec_next_week_col():
    # method adding column for how much total rain norawy today + next 7 days
    path_ = "..\\data\\input\\combined\\"
    df = pd.read_csv(path_ + "all_data_daily.csv", sep=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    orig_len = len(df)
    days_ahead = 8
    for i in range(1, days_ahead):
        df["{} prec lag".format(i)] = df["Prec Norway"].shift(-i)
    prec_cols = ["{} prec lag".format(i) for i in range(1, days_ahead)]
    df["Prec Norway 7"] = df[prec_cols].sum(axis=1)
    mean_forecast = df["Prec Norway 7"].mean()
    for i in range(1, days_ahead):
        df.loc[len(df) - i, "Prec Norway 7"] = mean_forecast
    df = df.drop(columns=prec_cols)
    new_len = len(df)
    assert orig_len == new_len
    df.to_csv(path_ + "all_data_daily.csv", index=False, float_format='%g')


def write_wind_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution == "h":
        path = "..\\data\\input\\wind_hourly"
        out_path = "..\\data\\input\\combined\\wind_hourly.csv"
        out_col = ["Date", "Hour", "Wind"]
    elif resolution == "d":
        path = "..\\data\\input\\wind_daily"
        out_path = "..\\data\\input\\combined\\wind_daily.csv"
        out_col = ["Date", "Wind"]
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
    df_all["Wind DK"] = df_all["DK1"] + df_all["DK2"]
    print(df_all)
    assert False
    df_all = df_all[["Date", "Wind DK"]]
    print(df_all.columns)
    print(df_all)
    assert False
    df_all = df_all[out_col]
    df_all = df_all.interpolate(method='linear', axis=0)
    assert df_all["Wind DK"].isnull().sum(axis=0) == 0
    df_all.to_csv(out_path, index=False, float_format='%g')


def add_swedish_wind_to_all_data():
    warnings.filterwarnings("ignore")
    all_data = pd.read_csv("input/combined/all_data_daily.csv")
    orig_len = len(all_data)
    all_data["Date"] = pd.to_datetime(all_data["Date"], format="%Y-%m-%d")
    df = pd.DataFrame(columns=["Date", "Hour", "Wind SE"])
    for year in range(2014, 2021):
        df_year = pd.read_csv("input/entsoe/wind_{}.csv".format(year), usecols=["MTU", "Wind Onshore  - Actual Aggregated [MW]"])
        df_year["Date"] = pd.to_datetime(df_year["MTU"].str[0:10], format="%d.%m.%Y")
        df_year["Hour"] = df_year["MTU"].str[11:13].astype(int)
        df_year = df_year[["Date", "Hour", "Wind Onshore  - Actual Aggregated [MW]"]]
        df_year = df_year.rename(columns={"Wind Onshore  - Actual Aggregated [MW]": "Wind SE"})
        df = df.append(df_year, ignore_index=True)
    wind_dk = all_data[["Date", "Wind DK"]]
    df = df.groupby(by=["Date"]).sum()
    df = df.merge(wind_dk, on=["Date"], how="outer")
    df_complete = df[df["Wind SE"] != 0]
    df_complete["Ratio"] = df_complete["Wind SE"] / df_complete["Wind DK"]
    factor = df_complete["Ratio"].mean()
    print(factor)
    assert False
    first_phase = df[df["Date"] < datetime(2015, 1, 5)]
    first_phase["Wind SE"] = round(factor * first_phase["Wind DK"], 2)
    second_phase = df[df["Date"] > datetime(2015, 1, 4)]
    df = first_phase.append(second_phase, ignore_index=True)
    df = df.interpolate()
    assert [i == 0 for i in df.isnull().sum()]
    all_data["Wind SE"] = df["Wind SE"]
    all_data["Wind Prod"] = all_data["Wind DK"] + all_data["Wind SE"]
    assert orig_len == len(all_data)
    all_data.to_csv("input/combined/all_data_daily.csv", index=False, float_format="%g")


def append_data_to_all_data(data, resolution):
    df = pd.read_csv("..\\data\\input\\combined\\{}_{}.csv".format(data, resolution))
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    all_data = pd.read_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution))
    all_data["Date"] = pd.to_datetime(all_data["Date"], format="%Y-%m-%d")
    if resolution == "daily":
        all_data = all_data.merge(df, on="Date", how="outer")
    elif resolution == "hourly":
        all_data = all_data.merge(df, on=["Date", "Hour"], how="outer")
    all_data.to_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution), index=False, float_format='%g')


def append_arcshinh_to_all_data(col_name, resolution):
    df = pd.read_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution))
    orig_len = len(df)
    df, a, b = to_arcsinh(df, col_name)
    df = df.rename(columns={"Trans {}".format(col_name): "Asinh {}".format(col_name)})
    new_len = len(df)
    assert orig_len == new_len
    df.to_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution), index=False, float_format='%g')


def remove_column_from_all_data(col_name, resolution):
    df = pd.read_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution))
    orig_len = len(df)
    no_cols = len(df.columns)
    df = df.drop(columns=[col_name])
    new_len = len(df)
    assert orig_len == new_len
    assert len(df.columns) == no_cols - 1
    df.to_csv("..\\data\\input\\combined\\all_data_{}.csv".format(resolution), index=False, float_format='%g')


def add_lagged_col_to_all_data(col_name, res, lag):
    df = pd.read_csv("..\\data\\input\\combined\\all_data_{}.csv".format(res))
    orig_len = len(df)
    df[col_name + " {} lag".format(lag)] = df[col_name].shift(lag)
    df = df.fillna(method='ffill')
    new_len = len(df)
    assert orig_len == new_len
    df.to_csv("..\\data\\input\\combined\\all_data_{}.csv".format(res), index=False, float_format='%g')


def write_fossil_prices_to_combined():
    curr_df = get_df_end_path("/currency.xlsx")
    oil_df = get_df_end_path("/oil_prices.xlsx")
    oil_df["Oil"] = oil_df["Oil"] * (1 / curr_df["EUR to USD"])
    coal_df = get_df_end_path("/coal_prices.xlsx")
    coal_df["Coal"] = coal_df["Coal"] * (1 / curr_df["EUR to USD"])
    gas_df = get_df_end_path("/gas_prices_2.xlsx")
    carbon_df = get_df_end_path("/low_carbon_prices.xlsx")
    df = oil_df.merge(coal_df, on="Date", how="outer")
    df = df.merge(gas_df, on="Date", how="outer")
    df = df.merge(carbon_df, on="Date", how="outer")
    df.to_csv("input/combined/fossil_daily.csv", index=False, float_format='%g')


def get_df_end_path(end_path):
    dir_path = "input/eikon/raw"
    if end_path == "/currency.xlsx":
        head_no = 0
    else:
        head_no = 1
    df = pd.read_excel(dir_path + end_path, header=head_no, usecols=[0, 1])
    full_date = get_data(datetime(2014, 1, 1), datetime(2020, 12, 31), [], os.getcwd(), "d")
    df = full_date.merge(df, on="Date", how="outer").interpolate(method="linear")
    df = df[df["Date"].dt.year.isin(range(2014, 2021))]
    df = df.fillna(method="bfill")
    df = df.fillna(method="ffill")
    return df


def add_fossil_to_all_data():
    df_f = pd.read_csv("input/combined/fossil_daily.csv")
    df_all = pd.read_csv("input/combined/all_data_daily.csv")
    assert len(df_all) == len(df_f)
    df_all = df_all.merge(df_f, on="Date", how="outer")
    assert [i == 0 for i in df_all.isna().sum()]
    df_all.to_csv("..\\data\\input\\combined\\all_data_daily.csv", index=False, float_format='%g')


def add_european_prices_all_data():
    apx = get_price_df("apx")
    currency_df = get_price_df("pound_to_euro")
    apx["APX"] = apx["APX"] * currency_df["EUR"]
    omel = get_price_df("omel")
    eex = get_price_df("eex")
    df = apx.merge(omel, on="Date", how="outer")
    df = df.merge(eex, on="Date", how="outer")
    full_date = get_data(datetime(2014, 1, 1), datetime(2020, 12, 31), [], os.getcwd(), "d")
    df = full_date.merge(df, on="Date", how="outer").interpolate(method="linear")
    df = df[df["Date"].dt.year.isin(range(2014, 2021))]
    df = df.fillna(method="bfill")
    df = df.fillna(method="ffill")
    df = df.round(decimals=2)
    df_all = pd.read_csv("input/combined/all_data_daily.csv")
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="%Y-%m-%d")
    orig_len = len(df_all)
    df_all = df_all.merge(df, on="Date")
    assert len(df_all) == orig_len
    assert [i == 0 for i in df_all.isna().sum()]
    df_all.to_csv("input/combined/all_data_daily.csv", index=False)


def get_price_df(exchange):  # Helping method
    if exchange == "pound_to_euro":
        columns = ["Date", "GBP", "EUR".format(exchange.upper())]
        file_path = "input/eikon/raw/{}.xlsx".format(exchange)
    else:
        file_path = "input/eikon/raw/{}_prices.xlsx".format(exchange)
        columns = ["Date", "{}".format(exchange.upper())]
    df = pd.read_excel(file_path, header=1, usecols=columns)
    df = df[(df["Date"] >= datetime(2014, 1, 1)) & (df["Date"] <= datetime(2020, 12, 31))]
    return df


def write_est_demand_to_all_data():
    data = pd.read_csv("../data/input/auction/volume_analyses.csv", usecols=["Date", "Hour", "Est Vol", "Net flow"])
    print("Mean net flow auction data {:.2f} MWh".format(data["Net flow"].mean()))
    print("Abs mean net flow auction data {:.2f} MWh".format(abs(data["Net flow"]).mean()))
    data = data.drop(columns=["Net flow"])
    data = data.rename(columns={"Est Vol": "Curve Demand"})
    all_data = pd.read_csv("../data/input/combined/all_data_hourly.csv")
    all_data = all_data.drop(columns=["Curve Demand"])
    orig_len = len(all_data)
    all_data = all_data.merge(data, on=["Date", "Hour"], how="outer")
    all_data['Curve Demand'] = all_data.apply(
        lambda row: row["Total Vol"] if np.isnan(row['Curve Demand']) else row['Curve Demand'], axis=1)
    assert orig_len == len(all_data)
    assert [i == 0 for i in all_data.isna().sum()]
    # all_data.to_csv("../data/input/combined/all_data_hourly.csv", index=False, float_format='%g')
    all_data["Curve MAE"] = abs(all_data["Total Vol"] - all_data["Curve Demand"])
    print("Mean curve difference hour {:.2f} MWh".format(all_data["Curve MAE"].mean()))
    daily_curve_demand = all_data[["Date", "Curve Demand"]].groupby(by="Date").sum()
    all_data_daily = pd.read_csv("../data/input/combined/all_data_daily.csv")
    orig_len = len(all_data_daily)
    all_data_daily = all_data_daily.merge(daily_curve_demand, on="Date")
    assert orig_len == len(all_data_daily)
    assert [i == 0 for i in all_data_daily.isna().sum()]
    #all_data_daily.to_csv("../data/input/combined/all_data_daily.csv", index=False, float_format='%g')
    all_data_daily = all_data_daily[(all_data_daily["Date"] >= datetime(2014, 7, 1)) & (all_data_daily["Date"] < datetime(2020, 6, 3))]
    all_data_daily["Curve MAE"] = abs(all_data_daily["Total Vol"] - all_data_daily["Curve Demand"])
    print("Mean curve difference day {:.2f} MWh".format(all_data_daily["Curve MAE"].mean()))


def winter_summer_time_volumes_analyses():
    data_path = "../data/input/auction/volume_analyses.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    all_hours = pd.read_csv("../data/input/combined/all_data_hourly.csv", usecols=["Date", "Hour"])
    all_hours["Date"] = pd.to_datetime(all_hours["Date"], format="%Y-%m-%d")
    all_hours = all_hours[(all_hours["Date"] >= datetime(2014, 7, 1)) & (all_hours["Date"] < datetime(2020, 6, 3))]
    df = all_hours.merge(df, on=["Date", "Hour"], how="outer")
    df = df.interpolate().dropna()
    df.to_csv(data_path, index=False, float_format='%g')


def write_holiday_to_all_data():
    import holidays
    no_holydays = holidays.Norway()
    paths = ["daily", "hourly"]
    for p in paths:
        all_data = pd.read_csv("../data/input/combined/all_data_{}.csv".format(p))
        all_data["Date"] = pd.to_datetime(all_data["Date"], format="%Y-%m-%d")
        all_data["Holiday"] = all_data.apply(
            lambda row: 1 if row["Date"] in no_holydays and row["Weekday"] != 7 else 0, axis=1)
        all_data.to_csv("../data/input/combined/all_data_{}.csv".format(p), index=False, float_format='%g')


def shift_total_hydro_dev_test():
    df = pd.read_csv("../data/input/combined/all_data_daily.csv")
    orig_len = len(df)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    add_future = False
    if add_future:
        df["Total Hydro Dev Shift"] = df["Total Hydro Dev"].shift(7)
        df["Total Hydro Dev Shift"] = df["Total Hydro Dev Shift"].fillna(method='bfill')
        assert orig_len == len(df)
        assert [i == 0 for i in df.isna().sum()]
    remove_future = True
    if remove_future:
        df = df.drop(columns=["Total Hydro Dev Shift"])
        assert orig_len == len(df)
        assert [i == 0 for i in df.isna().sum()]
        df.to_csv("../data/input/combined/all_data_daily.csv", index=False, float_format='%g')


def update_season():
    change = {1: [12, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9, 10, 11]}
    for res in ["daily", "hourly"]:
        df = pd.read_csv("../data/input/combined/all_data_{}.csv".format(res))
        orig_len = len(df)
        df["Season"] = df.apply(lambda row: 1 if row["Month"] in change[1] else 2 if row["Month"] in change[2] else 3 if
                                row["Month"] in change[3] else 4, axis=1)
        assert len(df) == orig_len
        assert [i == 0 for i in df.isna().sum()]
        df.to_csv("../data/input/combined/all_data_{}.csv".format(res), index=False, float_format='%g')


if __name__ == '__main__':
    print("Running method.." + "\n")
    # write_price_to_combined("h", convert_to_csv=True, replace_commas=True)
    # write_price_to_combined("d", convert_to_csv=True, replace_commas=True)
    # write_volume_to_combined("d", convert_to_csv=True, replace_commas=True)
    # write_supply_to_all_data("d", convert_to_csv=False, replace_commas=True)
    # write_demand_to_all_data("d", convert_to_csv=False, replace_commas=True)
    # write_volume_to_combined("h", convert_to_csv=True, replace_commas=True)
    # write_hydro_all_weekly(convert_to_csv=True, replace_commas=False)  # replace_commas=False, always
    # write_hydro_daily_to_combined()
    # write_hydro_hourly_to_combined()
    # write_daily_hydro_deviations_to_combined()
    # write_hourly_hydro_deviations_to_combined()
    # write_daily_temperature_to_combined()
    # write_hourly_temperature_to_combined()
    # combine_all_data("daily")
    # combine_all_data("hourly")
    # remove_summer_winter_time()
    # add_time_columns_to_all_data("d")
    # add_time_columns_to_all_data("h")
    # write_daily_temperature_to_combined()
    # write_daily_precipitation_to_combined()
    # append_weather_to_all_data()
    # add_prec_next_week_col()
    # write_wind_to_combined("h", convert_to_csv=False, replace_commas=True)
    # add_swedish_wind_to_all_data()
    # write_wind_to_combined("h", convert_to_csv=False, replace_commas=True)
    # append_data_to_all_data("wind", "daily")
    # append_arcshinh_to_all_data("Total Hydro Dev", "daily")
    # add_lagged_col_to_all_data("Total Hydro Dev", "daily", -11)
    # remove_column_from_all_data("Total Hydro Dev -11 lag", "daily")
    # add_fossil_to_all_data()
    # add_european_prices_all_data()
    # winter_summer_time_volumes_analyses()
    # write_est_demand_to_all_data()
    # write_holiday_to_all_data()
    # shift_total_hydro_dev_test()
    update_season()


# script connecting src and data folder.
import datetime
import pandas as pd
import numpy as np
import os


def get_data(from_date, to_date, column_list, work_dir):  # dates on form "dd.mm.yyyy", columns in list ["System Price", ...]
    # print("--Retrieving data from {} to {}. Columns: {} --\n".format(from_date, to_date, ", ".join(column_list)))
    path = get_path_to_all_data(work_dir)
    if isinstance(from_date, str):
        first_date = datetime.datetime.strptime(from_date, '%d.%m.%Y')
    else:
        first_date = datetime.datetime(from_date.year, from_date.month, from_date.day)
    if isinstance(to_date, str):
        last_date = datetime.datetime.strptime(to_date, '%d.%m.%Y')
    else:
        last_date = datetime.datetime(to_date.year, to_date.month, to_date.day)
    valid_dates = check_valid_dates(first_date, last_date)
    if valid_dates:
        columns = ["Date", "Hour"] + column_list
        all_data = pd.read_csv(path, usecols=columns)
        all_data["Date"] = pd.to_datetime(all_data["Date"], format='%Y-%m-%d')
        mask = (all_data['Date'] >= first_date) & (all_data['Date'] <= last_date)
        df = all_data.loc[mask]
        df = df.reset_index(drop=True)
        return df
    else:
        assert False


def get_index(string_start_date):
    return 100

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


def get_path_to_all_data(work_dir):
    path_to_project = "/".join(work_dir.split("\\")[0:5])
    path_to_all_data = "/data/input/combined/all_data_hourly.csv"
    path = path_to_project + path_to_all_data
    return path


if __name__ == '__main__':
    date_string_from = "01.01.2019"
    date_string_do = "31.12.2019"
    col = ["System Price"]
    data = get_data(date_string_from, date_string_do, col, os.getcwd())
    print(data)

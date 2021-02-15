# script connecting src and data folder.
import datetime
import pandas as pd
import numpy as np
import os


# TODO: write method so that get_data takes in resolution as parameter
def get_data(from_date, to_date, column_list, work_dir):  # dates on form "dd.mm.yyyy", columns in list ["System Price", ...]
    # print("--Retrieving data from {} to {}. Columns: {} --\n".format(from_date, to_date, ", ".join(column_list)))
    path = get_path_to_all_data(work_dir)
    first_row_in_data = datetime.datetime.strptime("01.01.2014", '%d.%m.%Y')
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
        difference = first_date - first_row_in_data
        years = difference.days // 365
        hours_less = years - 1
        skip_rows_no = int(difference.days * 24 + difference.seconds / 3600) + (16 * years) - hours_less
        columns = ["Date", "Hour"] + column_list
        if len(column_list)==1 and column_list[0] == "all":
            dataset = pd.read_csv(path, skiprows=range(1, skip_rows_no))
        else:
            dataset = pd.read_csv(path, skiprows=range(1, skip_rows_no), usecols=columns)
        dataset["Date"] = pd.to_datetime(dataset["Date"], format='%Y.%m.%d')
        dataset = dataset[dataset['Date'] <= np.datetime64(last_date.date())]
        return dataset
    else:
        assert False


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

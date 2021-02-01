# script connecting src and data folder.
import datetime
import pandas as pd
import numpy as np


def get_data(from_date, to_date, column_list):  # dates on form "dd.mm.yyyy", columns in list ["System Price", ...]
    print("--Retrieving data from {} to {}. Columns: {} --\n".format(from_date, to_date, ", ".join(column_list)))
    valid_dates = check_valid_dates(from_date, to_date)
    if valid_dates:
        first_row_in_data = datetime.datetime.strptime("01.01.2014", '%d.%m.%Y')
        first_date = datetime.datetime.strptime(from_date, '%d.%m.%Y')
        last_date = datetime.datetime.strptime(to_date, '%d.%m.%Y')
        difference = first_date - first_row_in_data
        years = difference.days // 365
        hours_less = years - 1
        skip_rows_no = int(difference.days * 24 + difference.seconds / 3600) + (16 * years) - hours_less
        columns = ["Date", "Hour"] + column_list
        path = "../../data/input/combined/all_data_hourly.csv"  # only works when run from src
        if len(column_list) > 0:
            dataset = pd.read_csv(path, skiprows=range(1, skip_rows_no), usecols=columns)
        else:
            dataset = pd.read_csv(path, skiprows=range(1, skip_rows_no))
        dataset["Date"] = pd.to_datetime(dataset["Date"], format='%Y.%m.%d')
        dataset = dataset[dataset['Date'] <= np.datetime64(last_date.date())]
        return dataset
    else:
        assert False


def check_valid_dates(from_date, to_date):  # helping method checking that dates are valid
    valid = True
    try:
        first_date = datetime.datetime.strptime(from_date, '%d.%m.%Y')
        last_date = datetime.datetime.strptime(to_date, '%d.%m.%Y')
        if last_date < first_date:
            print("First date must come before last date")
            valid = False
    except Exception as e:
        print(e)
    finally:
        return valid


if __name__ == '__main__':
    date_string_from = "01.01.2019"
    date_string_do = "31.12.2019"
    col = ["System Price"]
    data = get_data(date_string_from, date_string_do, col)
    print(data)

from sklearn.preprocessing import MinMaxScaler
from data.data_handler import get_data
import os

train_df = get_data("01.01.2014", "31.12.2018", ["System Price", "Total Vol", "Total Hydro Dev"], os.getcwd(), "d")
x = train_df[["Total Vol", "Total Hydro Dev"]]
y = train_df[["System Price"]]
scaler = MinMaxScaler()
scaler.fit(x)
test_df = get_data("01.01.2019", "31.12.2019", ["Total Vol", "Total Hydro Dev"], os.getcwd(), "d")
x_test = test_df[["Total Vol", "Total Hydro Dev"]]
x_test_scaled = scaler.transform(x)
print(x_test_scaled)

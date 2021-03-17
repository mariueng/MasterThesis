from data import data_handler
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
from pathlib import Path
from src.system.scores import get_all_point_metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid warnings from Tensorflow or any {'0', '1', '2'}
import tensorflow
tensorflow.random.set_seed(2)
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.models.expert_day.expert_day import calculate_performance


def get_data_set(s_date, e_date):
    day_df = data_handler.get_data(s_date, e_date, ["System Price", "Weekday", "Month", "T Nor"], os.getcwd(), "d")
    hour_df = data_handler.get_data(s_date, e_date, ["System Price"], os.getcwd(), "h")
    weekdays = ["d{}".format(i) for i in range(1, 8)]
    months = ["m{}".format(i) for i in range(1, 13)]
    hours = ["h{}".format(i) for i in range(1, 25)]
    col = ["Price"] + [w for w in weekdays] + [m for m in months] + [h for h in hours]
    data = pd.DataFrame(columns=col)
    for i in range(len(day_df)):
        row = {}
        date = day_df.loc[i, "Date"].date()
        row["Price"] = day_df.loc[i, "System Price"]
        row["Temp Norway"] = day_df.loc[i, "T Nor"]
        weekday = day_df.loc[i, "Weekday"]
        month = day_df.loc[i, "Month"]
        hourly_prices = hour_df["System Price"][hour_df["Date"].isin([date])].tolist()
        if len(hourly_prices) != 24:
            print(hourly_prices)
            print(len(hourly_prices))
            print(day_df.iloc[i])
        assert len(hourly_prices) == 24
        for j in range(1, 8):
            if weekday == j:
                row["d{}".format(j)] = 1
            else:
                row["d{}".format(j)] = 0
        for j in range(1, 13):
            if month == j:
                row["m{}".format(j)] = 1
            else:
                row["m{}".format(j)] = 0
        for j in range(1, 25):
            row["h{}".format(j)] = hourly_prices[j - 1] - row["Price"]
        data = data.append(row, ignore_index=True)
    return data


def fit_model(train):
    train_x, train_y = convert2matrix(train)
    number_of_inputs = train_x.shape[1]
    model = get_model(number_of_inputs)
    model_fit = model.fit(train_x, train_y, epochs=50, batch_size=30, verbose=0,
                          shuffle=False, callbacks=EarlyStopping(monitor="loss", patience=10))
    evaluate_model(model, model_fit, train_x, train_y)
    model.save(r"first_model")
    return model


def evaluate_model(model, model_fit, train_x, train_y):
    plt.figure(figsize=(8, 4))
    plt.plot(model_fit.history['loss'], label='Train Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.show()
    train_score = model.evaluate(train_x, train_y, verbose=0)
    print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
          % (np.sqrt(train_score[1]), train_score[2]))


def get_model(number_of_inputs):
    model = Sequential()
    # model.add(Dense(units=32, input_dim=look_back, activation='relu'))
    model.add(Dense(units=8, input_dim=number_of_inputs, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(24))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model


# turn dataset into matrix
def convert2matrix(data_arr):
    x, y = [], []
    for i in range(len(data_arr)):
        row = data_arr.loc[i]
        all_keys = row.keys().tolist()
        output_keys = ["h{}".format(i) for i in range(1, 25)]
        input_keys = [i for i in all_keys if i not in output_keys]
        x_list = row[[i for i in input_keys]].tolist()
        y_list = row[[i for i in output_keys]].tolist()
        x.append(x_list)
        y.append(y_list)
    return np.asarray(x).astype('float32'), np.asarray(y).astype('float32')


def test_model(test, start, end):
    model = load_model(r'first_model')
    test_x, test_y = convert2matrix(test)
    predictions = []
    for x in test_x:
        x = x.reshape(1, len(x))
        prediction = model.predict(x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                                   workers=1, use_multiprocessing=False)
        predictions.append(prediction[0])
    analyze_results(predictions, test_y, start, end)


def analyze_results(pred, true, start, end):
    [f.unlink() for f in Path("validation/plots").glob("*") if f.is_file()]
    df = pd.DataFrame(columns=("Date", "System Price", "Forecast"))
    results = pd.DataFrame(columns=["Period", "mape", "smape", "mae", "rmse"])
    day_df = data_handler.get_data(start, end, ["System Price"], os.getcwd(), "d")
    for i in range(len(pred)):
        day_price = day_df.loc[i, "System Price"]
        prediction = [pred[i][j] + day_price for j in range(len(pred[i]))]
        sys_price = [true[i][j] + day_price for j in range(len(true[i]))]
        d_df = pd.DataFrame(columns=df.columns)
        d_df["System Price"] = sys_price
        d_df["Forecast"] = prediction
        d_df["Date"] = day_df.loc[i, "Date"]
        df = df.append(d_df, ignore_index=True)
        plot_forecast(i+1, d_df.loc[0, "Date"], d_df["System Price"], d_df["Forecast"])
        point_performance = get_all_point_metrics(d_df)
        point_performance["Period"] = i
        results = results.append(point_performance, ignore_index=True)
    calculate_performance(df, results, "MLP", dt.today())


def plot_forecast(period, date, true, forecast):
    plt.subplots(figsize=(6.5, 7))
    plt.plot(true, label="True", color="steelblue")
    plt.plot(forecast, label="Forecast", color="firebrick")
    plt.title("Result from MLP, {}".format(date.date()), pad=20)
    plt.xlabel("Hour of day", labelpad=12)
    plt.ylabel("System Price [€]", labelpad=12)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("validation/plots/{}.png".format(period))
    plt.close()


if __name__ == '__main__':
    start_date = dt(2014, 1, 1)
    end_date = dt(2018, 12, 31)
    # train_ = get_data_set(start_date, end_date)
    start_test = end_date+timedelta(days=1)
    end_test = dt(2019, 12, 31)
    # fit_model(train_)
    test_ = get_data_set(start_test, end_test)
    test_model(test_, start_test, end_test)

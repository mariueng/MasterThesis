from data import data_handler
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os

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


def test_model(test, start_date, end_date):
    model = load_model(r'first_model')
    test_x, test_y = convert2matrix(test)
    predictions = []
    for x in test_x:
        x = x.reshape(1, len(x))
        prediction = model.predict(x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                                   workers=1, use_multiprocessing=False)
        predictions.append(prediction[0])
    mapes = []
    maes = []
    for i in range(len(test_y)):
        pred = predictions[i]
        true = test_y[i]
        aes = []
        apes = []
        for j in range(24):
            aes.append(abs(true[j] - pred[j]))
            if true[j] != 0:
                apes.append(100 * abs(true[j] - pred[j])/true[j])
        mapes.append(sum(apes)/len(apes))
        maes.append(sum(aes)/len(aes))
    print("MAE: {}, MAPE: {}".format(sum(maes)/len(maes), sum(mapes)/len(mapes)))
    df = data_handler.get_data(start_date, end_date, ["System Price"], os.getcwd(), "h")
    df["Hour"] = pd.to_datetime(df['Hour'], format="%H").dt.time
    df["DateTime"] = df.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    prediction_list = []
    day_prices = data_handler.get_data(start_date, end_date, ["System Price"], os.getcwd(), "d")["System Price"].tolist()
    for i in range(len(day_prices)):
        pred = predictions[i]
        day_price = day_prices[i]
        all_prices = [pred[j] + day_price for j in range(24)]
        prediction_list.extend(all_prices)
    df["Forecast"] = prediction_list
    plt.plot(df["DateTime"], df["System Price"], label="True")
    plt.plot(df["DateTime"], df["Forecast"], label="Forecast")
    plt.show()



def prediction_plot(test_y, test_predict, last_train):
    df = pd.DataFrame(columns=["Hour", "True", "Forecast"])
    for i in range(len(last_train)):
        row = {"Hour": i, "True": last_train[i], "Forecast": None}
        df = df.append(row, ignore_index=True)
    for i in range(len(test_y)):
        row = {"Hour": i + len(last_train), "True": test_y[i], "Forecast": test_predict[i]}
        df = df.append(row, ignore_index=True)
    plt.figure(figsize=(13, 7))
    plt.plot(df["Hour"], df["True"], marker='.', label="True")
    plt.plot(df["Hour"], df["Forecast"], 'r', label="Forecast")
    plt.tight_layout()
    sns.despine(top=True)
    plt.ylabel('Price', size=10)
    plt.xlabel('Time step', size=10)
    plt.legend(fontsize=15)
    plt.ylim(20, 60)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_date = "01.01.2014"
    end_date = "31.12.2018"
    train_ = get_data_set(start_date, end_date)
    end_test = "31.01.2019"
    fit_model(train_)
    test_ = get_data_set(end_date, end_test)
    test_model(test_, end_date, end_test)

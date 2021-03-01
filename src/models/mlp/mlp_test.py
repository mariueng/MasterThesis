from data import data_handler
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Avoid warnings from Tensorflow or any {'0', '1', '2'}
import tensorflow
tensorflow.random.set_seed(2)
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_training_set():
    start_date = "01.01.2019"
    end_date = "10.03.2019"
    data = data_handler.get_data(start_date, end_date, ["System Price"], os.getcwd())
    data["Hour"] = pd.to_datetime(data['Hour'], format="%H").dt.time
    data["DateTime"] = data.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    data = data[["DateTime", "System Price"]]
    data = data.values[0:len(data), :]
    return data


def get_test_set(train):
    last_day_of_training = train[-24, 0].date()
    first_day_of_testing = last_day_of_training + timedelta(days=1)
    last_day_of_testing = first_day_of_testing + timedelta(days=13)
    data = data_handler.get_data(first_day_of_testing, last_day_of_testing, ["System Price"], os.getcwd())
    data["Hour"] = pd.to_datetime(data['Hour'], format="%H").dt.time
    data["DateTime"] = data.apply(lambda r: dt.combine(r['Date'], r['Hour']), 1)
    data = data[["DateTime", "System Price"]]
    data = data.values[0:len(data), :]
    return data


def fit_model(train):
    look_back = 4*24
    train_x, train_y = convert2matrix(train, look_back)
    model = get_model(look_back)
    model_fit = model.fit(train_x, train_y, epochs=100, batch_size=30, verbose=0,
                          shuffle=False, callbacks=EarlyStopping(monitor="loss", patience=10), )
    evaluate_model(model, model_fit, train_x, train_y)
    return model, look_back


def evaluate_model(model, model_fit, train_x, train_y):
    plt.figure(figsize=(8, 4))
    plt.plot(model_fit.history['loss'], label='Train Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.ylim(0, 30)
    # plt.show()
    train_score = model.evaluate(train_x, train_y, verbose=0)
    print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
          % (np.sqrt(train_score[1]), train_score[2]))


def get_model(look_back):
    model = Sequential()
    model.add(Dense(units=32, input_dim=look_back, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model


# turn dataset into matrix
def convert2matrix(data_arr, look_back):
    x, y = [], []
    for i in range(len(data_arr) - look_back):
        d = i + look_back
        x.append(data_arr[i:d, 1])
        y.append(data_arr[d, 1])
    return np.asarray(x).astype('float32'), np.asarray(y).astype('float32')


def forecast(train, model, test, look_back):
    x = train[-look_back:, 1]
    x = x.reshape((1, look_back))
    predictions = []
    for i in range(len(test)):
        input_x = np.asarray(x).astype('float32')
        prediction = model.predict(input_x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10,
                                   workers=1, use_multiprocessing=False)
        pred = prediction[0, 0]
        predictions.append(pred)
        x = np.append(x, prediction)
        x = np.delete(x, 0)
        x = x.reshape((1, look_back))
    test_y = test[:, 1]
    days_back = 5
    last_days_train = train[-24*days_back:, 1]
    prediction_plot(test_y, predictions, last_days_train)


def prediction_plot(test_y, test_predict, last_train):
    df = pd.DataFrame(columns=["Hour", "True", "Forecast"])
    for i in range(len(last_train)):
        row = {"Hour": i, "True": last_train[i], "Forecast": None}
        df = df.append(row, ignore_index=True)
    for i in range(len(test_y)):
        row = {"Hour": i+len(last_train), "True": test_y[i], "Forecast": test_predict[i]}
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
    train_ = get_training_set()
    test_ = get_test_set(train_)
    model_fit_, look_back_ = fit_model(train_)
    forecast(train_, model_fit_, test_, look_back_)
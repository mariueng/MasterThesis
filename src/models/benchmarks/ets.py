from src.models.model_handler import *
from data import data_formatter, data_handler
from src.system.metrics import evaluate_point_and_interval_forecast
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


# Get data
def load_data(start_time, end_time):
    df = data_handler.get_data(start_time, end_time, ['System Price'], os.getcwd())
    df = data_formatter.combine_hour_day_month_year_to_datetime_index(df)
    return df


# Fit new model to new data
def fit(train, verbose=False):
    # TODO: Check whether frequency should be passsed or if it is inferred correctly. See ValueWarning for more info.
    # TODO: Check whether parameters are correct. Uncertain about if correct model is implemented.
    ets = ETSModel(train.squeeze('columns'), error='add', trend='add', seasonal='add', damped_trend=True,
                   seasonal_periods=4)
    ets = ets.fit(disp=False)
    if verbose:
        print(ets.summary())
    return ets


# Use model to predict
def predict(ets_, test):
    forecast = ets_.get_prediction(start=test.index[0], end=test.index[-1])
    forecast = forecast.summary_frame(alpha=0.05)
    forecast.columns = ['System Price', 'Lower bound', 'Upper bound']
    return forecast


def score(forecast, actual, train, nominal_coverage=0.95):
    return evaluate_point_and_interval_forecast(forecast, actual, train, nominal_coverage)


# Save model to file
def save_model(id_, name, model, forecast, scores, folder):
    save_pickle(id_, name, model, forecast, scores, folder)


if __name__ == '__main__':
    """
    start_time_ = "01.01.2018"
    end_time_ = "31.01.2018"
    data_ = load_data(start_time_, end_time_)
    split_ = 150
    train_, test_ = data_[:split_], data_[split_:]
    model_ = fit(train_)
    forecast_ = predict(model_, test_)
    scores = score(forecast_, test_, train_)
    print(scores)
    """

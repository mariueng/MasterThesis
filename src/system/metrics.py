import numpy as np
import pandas as pd
import sklearn.metrics
import properscoring as ps
from scipy import stats

np.seterr(invalid="ignore", divide="ignore")


# Point forecasting metrics

def mape(forecast: pd.DataFrame, actual: pd.DataFrame) -> (float, pd.DataFrame):
    apes = pd.DataFrame(
        (abs(actual['System Price'].values - forecast['System Price'].values) / actual[
            'System Price'].values) * 100,
        columns=actual.columns, index=actual.index)
    if 0 in actual.values:
        mape_ = -1
        print("Not possible to calculate MAPE when actual contains zero-values. MAPE set to -1.")
    else:
        mape_ = apes['System Price'].mean()
    return mape_, apes


def smape(forecast: pd.DataFrame, actual: pd.DataFrame) -> (float, pd.DataFrame):
    sapes = pd.DataFrame(
        (abs(actual['System Price'].values - forecast['System Price'].values) / (actual[
                                                                                     'System Price'].values + forecast[
                                                                                     'System Price'].values) / 2) * 100,
        columns=actual.columns, index=actual.index)
    if 0 in actual.values + forecast['System Price'].values:
        smape_ = -1
        print("Not possible to calculate sMAPE when actual + forecasts equals zero. sMAPE set to -1")
    else:
        smape_ = sapes['System Price'].mean()
    return smape_, sapes


def mae(forecast: pd.DataFrame, actual: pd.DataFrame) -> (float, pd.DataFrame):
    aes = pd.DataFrame(
        abs(actual['System Price'].values - forecast['System Price'].values),
        columns=actual.columns, index=actual.index)
    mae_ = aes['System Price'].mean()
    return mae_, aes


def rmse(forecast: pd.DataFrame, actual: pd.DataFrame) -> (float, pd.DataFrame):
    ses = pd.DataFrame(
        abs(actual['System Price'].values - forecast['System Price'].values) ** 2,
        columns=actual.columns, index=actual.index)
    rmse_ = (ses['System Price'].mean()) ** (1 / 2)
    return rmse_, ses


def evaluate_point_forecast(forecast, actual):
    mape_, _ = mape(forecast, actual)
    smape_, _ = smape(forecast, actual)
    mae_, _ = mae(forecast, actual)
    rmse_, _ = rmse(forecast, actual)
    scores = {'mape': mape_,
              'smape': smape_,
              'mae': mae_,
              'rmse': rmse_}
    return scores


# Prediction Interval metrics
# TODO: Implement CRPS, Pinball loss, Brier Score, winkler score, UC and CC, CWC, PINAW, OWA (?)

# TODO: check whether this 'noinspection' is dangerous, or if PyCharm is just being an idiot
# noinspection PyTypeChecker
def msis(forecast: pd.DataFrame, actual: pd.DataFrame, in_sample: pd.DataFrame) -> (float, int):
    alpha = 0.05
    u, l, y = forecast['Upper bound'], forecast['Lower bound'], actual['System Price']
    i_lower, i_upper = pd.concat([l, y], axis=1), pd.concat([u, y], axis=1)
    # TODO: Check if this is correct or if a single two-sided indicator function is intended. Both are implemented.
    i_lower = i_lower.apply(lambda x: lower_indicator_function(x['Lower bound'], x['System Price']), axis=1)
    i_upper = i_upper.apply(lambda x: upper_indicator_function(x['Upper bound'], x['System Price']), axis=1)
    h, n = len(forecast.index), len(in_sample.index)
    m = 24  # Time interval between successive observations, i.e. 24 for hourly, 12 for monthly, four for quarterly 1
    # for one year, weekly and daily data
    mis = (1 / h) * (sum((u - l) + (2 / alpha) * ((l - y) * i_lower + (y - u) * i_upper)))
    scaling = sum([abs(in_sample.iat[t, 0] - in_sample.iat[t - m, 0]) for t in range(m, n)]) / (n - m)
    # TODO: implement return value to be able to adapt and calculate MSIS based on selected range (dummy -1 return now)
    if scaling <= 0:
        print("Time series too short, scaling became zero. Returning MIS.")
        return mis, -1
    return mis / scaling, -1


# Helper method to perform row wise check whether the observation is contained within the interval
def indicator_function(lower_bound, upper_bound, observation):
    if lower_bound <= observation <= upper_bound:
        return 1
    else:
        return 0


# Helper method to perform row wise check whether the observation is above the lower bound
def lower_indicator_function(lower_bound, observation):
    if lower_bound <= observation:
        return 1
    else:
        return 0


# Helper method to perform row wise check whether the observation is below the upper bound
def upper_indicator_function(upper_bound, observation):
    if observation <= upper_bound:
        return 1
    else:
        return 0


# noinspection PyTypeChecker
def absolute_coverage_error(forecast: pd.DataFrame, actual: pd.DataFrame, nominal_value: float) -> (
        float, pd.DataFrame):
    u = forecast['Upper bound']
    l = forecast['Lower bound']
    y = actual['System Price']
    i = pd.concat([l, u, y], axis=1).apply(
        lambda x: indicator_function(x['Lower bound'], x['Upper bound'], x['System Price']), axis=1)
    return abs((i.sum() / len(i.index)) - nominal_value), i


def crps():
    return


def brier_score():
    return


# noinspection PyTypeChecker
def pinball_loss(quantile_forecast: float, actual: pd.DataFrame, target_quantile: float) -> float:
    y = actual['System Price']
    quantiles = [0.05, 0.10, 0.5, 0.90, 0.95]
    tau = 0.95
    i_lower, i_upper = pd.concat([l, y], axis=1), pd.concat([u, y], axis=1)
    i_lower = i_lower.apply(lambda x: lower_indicator_function(x['Lower bound'], x['System Price']), axis=1)
    i_upper = i_upper.apply(lambda x: upper_indicator_function(x['Upper bound'], x['System Price']), axis=1)
    pinball_score = 0
    if y >= quantile_forecast:
        pinball_score += (y - quantile_forecast) * target_quantile
    elif quantile_forecast > y:
        pinball_score += (quantile_forecast - y) * target_quantile
    return


def evaluate_interval_forecast(forecast, actual, in_sample, nominal_coverage):
    msis_, _ = msis(forecast, actual, in_sample)
    ace_, _ = absolute_coverage_error(forecast, actual, nominal_coverage)
    scores = {'msis': msis_,
              'ace': ace_,
              }
    return scores


def evaluate_point_and_interval_forecast(forecast, actual, in_sample, nominal_coverage):
    point = evaluate_point_forecast(forecast.copy(), actual.copy())
    interval = evaluate_interval_forecast(forecast, actual, in_sample, nominal_coverage)
    return {**point, **interval}


if __name__ == '__main__':

    """
    a = pd.DataFrame([1, 2, 3.5, 1, 3, 1, 2], columns=['System Price'])
    f = pd.DataFrame([[1, 0, 2],
                      [0, -1, 1],
                      [3, 2, 4],
                      [2, 1, 1.5],
                      [3, 2, 4],
                      [1, 0, 0],
                      [2, 1, 3]], columns=['System Price', 'Lower bound', 'Upper bound'])
    in_sample_ = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['System Price'])
    result = evaluate_interval_forecast(f, a, in_sample_, 0.95)
    print(result)
    """

    tau = 0.95  # Target quantile
    target_quantiles = [0.05, 0.10, 0.50, 0.90, 0.95]
    z = 0  # Forecasted value
    Y = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Possible actual values, i.e. actual distribution
    n = len(Y)  # Number of points
    for tau in target_quantiles:
        print(f"-------- Quantile: {tau} --------")
        for z in Y:
            loss = (tau / n) * sum([(Y[t] - z) for t in range(n) if Y[t] >= z]) + ((tau - 1) / n) * sum(
                [(z - Y[t]) for t in range(n) if Y[t] < z])
            print(loss)
        print("-------------------------------")

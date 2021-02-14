import numpy as np
import pandas as pd
from scipy import stats
np.seterr(invalid="ignore", divide="ignore")


# Point forecasting metrics

def mape(forecast, actual):
    """
    Method calculating MAPE score over a point forecast result
    :param forecast: pd.DataFrame
    :param actual: pd.DataFrame
    :return: numpy.float64, pd.DataFrame
    """
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


def smape(forecast, actual):
    """
    Method calculating sMAPE score over a point forecast result
    :param forecast: pd.DataFrame
    :param actual: pd.DataFrame
    :return: numpy.float64, pd.DataFrame
    """
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


def mae(forecast, actual):
    """
    Method calculating MAE score over a point forecast result
    :param forecast: pd.DataFrame
    :param actual: pd.DataFrame
    :return: numpy.float64, pd.DataFrame
    """
    aes = pd.DataFrame(
        abs(actual['System Price'].values - forecast['System Price'].values),
        columns=actual.columns, index=actual.index)
    mae_ = aes['System Price'].mean()
    return mae_, aes


def rmse(forecast, actual):
    """
    Method calculating RMSE score over a point forecast result
    :param forecast: pd.DataFrame
    :param actual: pd.DataFrame
    :return: numpy.float64, pd.DataFrame
    """
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
# TODO: Implement Pinball loss, Brier Score, winkler score, UC and CC, CWC, PINAW, OWA (?)

# TODO: check whether this 'noinspection' is dangerous, or if PyCharm is just being an idiot
# noinspection PyTypeChecker
def msis(forecast, actual, in_sample):
    alpha = 0.05
    u = forecast['Upper bound']
    l = forecast['Lower bound']
    y = actual['System Price']
    i_lower = pd.concat([l, y], axis=1)
    i_upper = pd.concat([u, y], axis=1)
    # TODO: Check if this is correct or if a single two-sided indicator function is intended. Both are implemented.
    i_lower = i_lower.apply(lambda x: lower_indicator_function(x['Lower bound'], x['System Price']), axis=1)
    i_upper = i_upper.apply(lambda x: upper_indicator_function(x['Upper bound'], x['System Price']), axis=1)
    h = len(forecast.index)
    n = len(in_sample.index)
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


def absolute_coverage_error(forecast, actual, nominal_value):
    u = forecast['Upper bound']
    l = forecast['Lower bound']
    y = actual['System Price']
    i = pd.concat([l, u, y], axis=1).apply(lambda x: indicator_function(x['Lower bound'], x['Upper bound'], x['System Price']), axis=1)
    return abs((i.sum() / len(i.index)) - nominal_value)



def lr_bt(self):
    """Likelihood ratio framework of Christoffersen (1998)"""
    hits = self.hit_series()  # Hit series
    tr = hits[1:] - hits[:-1]  # Sequence to find transitions

    # Transitions: nij denotes state i is followed by state j nij times
    n01, n10 = (tr == 1).sum(), (tr == -1).sum()
    n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

    # Times in the states
    n0, n1 = n01 + n00, n10 + n11
    n = n0 + n1

    # Probabilities of the transitions from one state to another
    p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
    p = n1 / n

    if n1 > 0:
        # Unconditional Coverage
        uc_h0 = n0 * np.log(1 - self.alpha) + n1 * np.log(self.alpha)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        # Independence
        ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
        if p11 > 0:
            ind_h1 += n11 * np.log(p11)
        ind = -2 * (ind_h0 - ind_h1)

        # Conditional coverage
        cc = uc + ind

        # Stack results
        df = pd.concat([pd.Series([uc, ind, cc]),
                        pd.Series([1 - stats.chi2.cdf(uc, 1),
                                   1 - stats.chi2.cdf(ind, 1),
                                   1 - stats.chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    # Assign names
    df.columns = ["Statistic", "p-value"]
    df.index = ["Unconditional", "Independence", "Conditional"]

    return df

def evaluate_interval_forecast(forecast, actual, in_sample):
    msis_, _ = msis(forecast, actual, in_sample)
    ace_, _ = absolute_coverage_error(forecast, actual)
    scores = {'msis': msis_,
              'ace': ace_,
              }
    return scores

if __name__ == '__main__':
    a = pd.DataFrame([1, 2, 3.5, 1, 3, 1, 2], columns=['System Price'])
    f = pd.DataFrame([[1, 0, 2],
                      [0, -1, 1],
                      [3, 2, 4],
                      [2, 1, 1.5],
                      [3, 2, 4],
                      [1, 0, 0],
                      [2, 1, 3]], columns=['System Price', 'Lower bound', 'Upper bound'])
    in_sample_ = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['System Price'])
    result = absolute_coverage_error(f, a, 0.95)
    print(result)

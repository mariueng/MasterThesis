import numpy as np
import pandas as pd

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
            'System Price'].values + forecast['System Price'].values) / 2) * 100,
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
    rmse_ = (ses['System Price'].mean()) ** (1/2)
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
# TODO: Implement MSIS, ACE, Pinball loss, Brier Score, winkler score, UC and CC, CWC, PINAW


def msis(forecast, actual):

    return msis


if __name__ == '__main__':
    a = pd.DataFrame([1, 2, 3.5, 1, 3, 1, 2], columns=['System Price'])
    f = pd.DataFrame([1, 0, 3, 2, 3, 1, 2], columns=['System Price'])
    s = evaluate_point_forecast(f, a)
    print(s)

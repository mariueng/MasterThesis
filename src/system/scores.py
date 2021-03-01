import math


def get_all_point_metrics(result):
    return {"mape": calculate_mape(result), "smape": calculate_smape(result),
            "mae": calculate_mae(result), "rmse": calculate_rmse(result)}


def calculate_coverage_error(result):
    result['Hit'] = result.apply(lambda row: 1 if row["Lower"] <= row["System Price"] <=
                                                  row["Upper"] else 0, axis=1)
    coverage = sum(result["Hit"]) / len(result)
    cov_error = abs(0.95 - coverage)
    return coverage * 100, cov_error * 100


def calculate_interval_score(result):
    t_values = []
    for index, row in result.iterrows():
        u = row["Upper"]
        l = row["Lower"]
        y = row["System Price"]
        func_l = 1 if y < l else 0
        func_u = 1 if y > u else 0
        t = (u - l) + (2 / 0.05) * (l - y) * func_l + (2 / 0.05) * (y - u) * func_u
        t_values.append(t)
    interval_score = sum(t_values) / len(t_values)
    return interval_score


def calculate_mape(result):
    apes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        if true != 0:
            forecast = row["Forecast"]
            ape = (abs(true - forecast) / true) * 100
            apes.append(ape)
    mape = sum(apes) / len(apes)
    return mape


def calculate_smape(result):
    sapes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        if (true + forecast)/2 != 0:
            sape = 100 * abs(abs(true - forecast) / abs((true + forecast) / 2))
            sapes.append(sape)
    smape = sum(sapes) / len(sapes)
    return smape


def calculate_mae(result):
    aes = []
    for index, row in result.iterrows():
        true = row["System Price"]
        forecast = row["Forecast"]
        ae = abs(true - forecast)
        aes.append(ae)
    mae = sum(aes) / len(aes)
    return mae


def calculate_rmse(result):
    ses = []
    for index, row in result.iterrows():
        true = row["System Price"]
        true = row["System Price"]
        forecast = row["Forecast"]
        se = (true - forecast) ** 2
        ses.append(se)
    rmse = math.sqrt(sum(ses) / len(ses))
    return rmse
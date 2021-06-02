from epftoolbox.evaluation import DM, plot_multivariate_DM_test
import pandas as pd


def get_all_results():
    """
    Method to obtain all results from test set and real prices for test set periods (351x336)
    :return:
    """
    base = r'../results/test/'
    models = ['NaiveDay', 'NaiveWeek', 'Sarima', 'ETS', 'ExpertModel', 'ExpertDay', 'ExpertMLP', 'CurveModel']
    models.reverse()
    paths = []
    # Retrieve all paths
    for model in models:
        paths.append(base + model + '/forecast.csv')

    # Read real price, dates and periods
    forecasts = pd.read_csv(r'../results/test/CurveModel/forecast.csv', delimiter=',', header=0, usecols=['System Price'])

    # Retrieve real prices from forecasts df
    real_price = forecasts.rename(columns={'System Price': 'Price'})

    # Retrieve all forecasts
    for path, model_name in zip(paths, models):
        forecasts[model_name] = pd.read_csv(path, delimiter=',', header=0, usecols=['Forecast'])

    # Remove real price from forecasts, not a forecast
    forecasts.drop(columns=['System Price'], inplace=True)
    return forecasts, real_price


def dm_test(norm):
    # Generating forecasts of multiple models
    # Download available forecast of the NP market available in the library repository
    # These forecasts accompany the original paper

    # Retrieve forecasts and real price for all 351 test periods
    forecasts, real_price = get_all_results()

    # Transforming indices to datetime format
    # forecasts.index = pd.to_datetime(forecasts.index)

    # Generating a plot to compare the models using the multivariate DM test
    plot_multivariate_DM_test(real_price=real_price, forecasts=forecasts, norm=norm)


if __name__ == '__main__':
    # Norm 1: Absolute errors
    dm_test(norm=1)
    # Norm 2: Squared errors
    dm_test(norm=2)
    # Norm 3: Absolute percentage errors
    # This is the attempt to implement MAPE as well, to do this you need to change the source code of the
    # package as well
    # dm_test(norm=3)

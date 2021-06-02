"""
Functions to compute and plot the univariate and multivariate versions of the Diebold-Mariano (DM) test.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def DM(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2

    # Computing the test statistic
    if version == 'univariate':

        # Computing the loss differential series for the univariate test
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2

        # Computing the loss differential size
        N = d.shape[0]

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    elif version == 'multivariate':

        # Computing the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

        # Computing the loss differential size
        N = d.size

        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value


def get_multi_DM_p_values(real_price, forecasts, norm=1):
    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns)

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p = DM(p_real=real_price.values.reshape(-1, 24),
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24),
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24),
                                                  norm=norm, version='multivariate')
                p_values.loc[model1, model2] = p
    return p_values



def get_all_results():
    models = ['NaiveDay', 'NaiveWeek', 'Sarima', 'ETS', 'ExpertModel', 'ExpertDay', 'ExpertMLP', 'CurveModel']
    models.reverse()
    paths = []
    # Retrieve all paths
    for model in models:
        paths.append('../results/test/' + model + '/forecast.csv')
    forecasts = pd.read_csv(r'../results/test/CurveModel/forecast.csv', delimiter=',', header=0, usecols=['System Price'])
    real_price = forecasts.rename(columns={'System Price': 'Price'})
    for path, model_name in zip(paths, models):
        forecasts[model_name] = pd.read_csv(path, delimiter=',', header=0, usecols=['Forecast'])
    forecasts.drop(columns=['System Price'], inplace=True)
    return forecasts, real_price


def mape_test():
    forecasts, real_price = get_all_results()
    for col in forecasts.columns:
        forecasts[col] = 100*(abs(forecasts[col]-real_price["Price"]))/real_price["Price"]
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns)
    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                d = np.abs(forecasts[model1]) - np.abs(forecasts[model2])
                N = d.size
                mean_d = np.mean(d)
                var_d = np.var(d, ddof=0)
                DM_stat = mean_d / np.sqrt((1 / N) * var_d)
                p_value = 1 - stats.norm.cdf(DM_stat)
                check_curve = False
                if check_curve:
                    if model1=="CurveModel" and model2=="Sarima":
                        print("Mean curve model: {:.2f}".format(forecasts[model1].mean()))
                        print("Mean sarima: {:.2f}".format(forecasts[model2].mean()))
                        print(d)
                        print("Difference {:.4f}".format(np.mean(d)))
                        print("Var d {:.4f}".format(np.var(d, ddof=0)))
                        print("DM stat {:.4f}".format(mean_d / np.sqrt((1 / N) * var_d)))
                        print("{} {}: {:.6f}".format(model1, model2, p_value))
                p_values.loc[model1, model2] = p_value
    return p_values


def plot_all():
    forecasts, real_price = get_all_results()
    mae_p = get_multi_DM_p_values(real_price, forecasts, norm=1)
    rmse_p = get_multi_DM_p_values(real_price, forecasts, norm=2)
    mape_p = mape_test()
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1),
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5.5))
    dfs = [mae_p, rmse_p, mape_p]
    axs = [ax1, ax2, ax3]
    titles = ["MAE", "RMSE", "MAPE"]
    model_names = [c for c in forecasts.columns]
    for i in range(3):
        ax = axs[i]
        p_values = dfs[i]
        im = ax.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
        ax.set_xticklabels(labels=model_names, rotation=90, size=11)
        ax.xaxis.set_ticks(range(len(model_names)))
        ax.set_yticklabels(model_names, size=11)
        ax.yaxis.set_ticks(range(len(model_names)))
        ax.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
        ax.set_title("DM - {}".format(titles[i]), y=1.02, size=14)
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar1 = fig.add_axes([p1[0]+0.02, 0.95, p1[1]+0.01, 0.04])
    plt.colorbar(im, cax=ax_cbar1, orientation='horizontal')
    plt.tight_layout()
    plt.savefig("../analysis/CurveModel/dm_last.png")

def check_mean():
    forecasts, real_price = get_all_results()
    for col in forecasts.columns:
        forecasts[col] = 100*(abs(forecasts[col]-real_price["Price"]))/real_price["Price"]
    print(forecasts)

if __name__ == '__main__':
    #check_mean()
    plot_all()


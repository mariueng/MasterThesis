import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from data.data_handler import get_data
import os


def read_cm_results(model):
    df = pd.read_csv(r'../results/test/' + model + '/forecast.csv', delimiter=',', header=0,
                     usecols=['System Price', 'Forecast', 'Upper', 'Lower'])
    return df


def get_error_distribution_by_horizon(model):
    df = read_cm_results(model)
    # parse_dates=[['Date', 'Hour']]
    # df.rename(columns={'Date_Hour': 'Datetime'}, inplace=True)

    # Calculate errors for each data point
    df['ae'] = df.apply(lambda x: ae(x['System Price'], x['Forecast']), axis=1)
    df['se'] = df.apply(lambda x: se(x['System Price'], x['Forecast']), axis=1)
    df['ape'] = df.apply(lambda x: ape(x['System Price'], x['Forecast']), axis=1)
    df['sape'] = df.apply(lambda x: sape(x['System Price'], x['Forecast']), axis=1)
    df['is'] = df.apply(lambda x: int_score(x['Lower'], x['Upper'], x['System Price']), axis=1)
    df['ind'] = df.apply(lambda x: indicator_function(x['Lower'], x['Upper'], x['System Price']), axis=1)

    # Drop unecessary data
    df.drop(columns=['Lower', 'Upper', 'System Price', 'Forecast'], inplace=True)

    # Assign what day in the forecasting horizon each entry is
    # for each entry in each period add entry + i to period in new column to make hour and day forecasted in forecasting
    # horizon
    # E.g.period 1, day 1, hour 0 is the first forecasted data point
    df = indecise(df)
    # Analyze errors based on hour and day
    # Across all forecasted hours
    df_hour = df.groupby(['forecast_hour']).mean()
    df_hour['se'] = np.sqrt(df_hour[['se']])
    df_hour.rename(columns={'ae': 'MAE', 'se': 'RMSE', 'ape': 'MAPE', 'sape': 'SMAPE', 'is': 'MIS', 'ind': 'COV'},
                   inplace=True)
    df_hour.drop(columns={'forecast_day'}, inplace=True)
    # Across all forecasted days
    df_day = df.groupby(['forecast_day']).mean()
    df_day['se'] = np.sqrt(df_day[['se']])
    df_day.rename(columns={'ae': 'MAE', 'se': 'RMSE', 'ape': 'MAPE', 'sape': 'SMAPE', 'is': 'MIS', 'ind': 'COV'},
                  inplace=True)
    df_day.drop(columns={'forecast_hour'}, inplace=True)
    return df_hour, df_day


# Helper method to index day and hour for forecasted value
def indecise(df):
    df['forecast_hour'] = 0
    df['forecast_day'] = 0
    hour = 0
    day = 1
    for index in df.index:
        df.at[index, 'forecast_hour'] = hour
        df.at[index, 'forecast_day'] = day
        hour = hour + 1
        if hour % 24 == 0:
            day = day + 1
        if hour == 336:
            hour = 0
        if day == 15:
            day = 1
    return df


# Metric helper functions
def ape(a, f):
    return abs(a - f) / a * 100


def sape(a, f):
    return abs(a - f) / ((a + f) / 2) * 100


def ae(a, f):
    return abs(a - f)


def se(a, f):
    return abs(a - f) ** 2


# noinspection PyTypeChecker
def int_score(l, u, a):
    i_upper = upper_indicator_function(u, a)
    i_lower = lower_indicator_function(l, a)
    alpha = 0.05
    return (u - l) + (2 / alpha) * ((l - a) * i_lower + (a - u) * i_upper)


# Helper method to perform row wise check whether the observation is contained within the interval
def indicator_function(lower_bound, upper_bound, observation):
    if lower_bound <= observation <= upper_bound:
        return 1
    else:
        return 0


# Helper method to perform row wise check whether the observation is above the lower bound
def lower_indicator_function(lower_bound, observation):
    if lower_bound >= observation:
        return 1
    else:
        return 0


# Helper method to perform row wise check whether the observation is below the upper bound
def upper_indicator_function(upper_bound, observation):
    if observation >= upper_bound:
        return 1
    else:
        return 0


# noinspection PyTypeChecker
def absolute_coverage_error(df, nominal_value=0.95):
    y = df['System Price']
    u = df['Upper']
    l = df['Lower']
    i = pd.concat([l, u, y], axis=1).apply(lambda x: indicator_function(x['Lower bound'], x['Upper bound'], x['System Price']), axis=1)
    return abs((i.sum() / len(i.index)) - nominal_value), i


def plot_horizon_error(df1, df2, x_label):
    img_size = (13, 7)
    true_color = "steelblue"
    fc_color = "firebrick"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=img_size)
    metrics = df1.columns.values
    hourly_x_values = df1.index
    daily_x_index = df2.index * 24 - 12
    for metric, ax in zip(metrics, axes.flat):
        ax.plot(hourly_x_values, df1[metric], label='Hourly', color=true_color)
        ax.plot(daily_x_index, df2[metric], label='Daily', color=fc_color)
        # ax.plot(df2.index, df2[metric], label='Naive Day', color=fc_color)
        # ax.plot(df3.index, df3[metric], label='SARIMA', color="darkorange")
        ax.set(title=f'{metric}')
        ax.set(xlabel=x_label, ylabel=metric)
        ax.legend()
        # ax.title(metric, pad=title_pad)
    fig.tight_layout()
    plt.show()
    # plt.close()


def get_demand_forecast_errors():
    df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\masteroppgave\src\models\curve_model\demand_scores_test_period\day_demand_results.csv')
    df['forecast_day'] = 0
    day = 1
    for index in df.index:
        df.at[index, 'forecast_day'] = day
        day = day + 1
        if day == 15:
            day = 1
    df.drop(columns={'Period', 'Curve Demand', 'Demand Forecast'}, inplace=True)
    df['se'] = np.square(df['Error'])
    df_day = df.groupby(['forecast_day']).mean()
    df_day['se'] = np.sqrt(df_day[['se']])
    return df_day


def get_spikes_per_day(model='CurveModel'):
    df = pd.read_csv(r'../results/test/' + model + '/forecast.csv', delimiter=',', header=0,
                     usecols=['Date', 'Hour', 'Period', 'System Price', 'Forecast', 'Upper', 'Lower'],
                     parse_dates=[['Date', 'Hour']])
    df.rename(columns={'Date_Hour': 'Datetime'}, inplace=True)
    df = indecise(df)
    df['cm_pred_spike'] = 0
    df['cm_pred_spike_dir'] = 0

    # Get data + 60 days before tesing period
    df_test_plus_60 = get_data("04.04.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    df_test_plus_60['Datetime'] = pd.to_datetime(df_test_plus_60.Date) + df_test_plus_60.Hour.astype('timedelta64[h]')
    df_test_plus_60.drop(columns={'Date', 'Hour'}, inplace=True)

    # Get predicted spikes by the model
    for period in df["Period"].unique():
        sub_df = df[df["Period"] == period][['Datetime', 'Forecast']].reset_index(drop=True)
        start_date = sub_df.loc[0, 'Datetime']
        from datetime import timedelta
        window = df_test_plus_60[(df_test_plus_60['Datetime'] >= start_date - timedelta(days=60)) & (df_test_plus_60['Datetime'] < start_date)]
        assert len(window) == 60 * 24
        mean = window["System Price"].mean()
        std_dev = window["System Price"].std()
        sub_df["Pos"] = sub_df.apply(lambda r: 1 if r["Forecast"] > mean + 1.96 * std_dev else 0, axis=1)
        sub_df["Neg"] = sub_df.apply(lambda r: -1 if r["Forecast"] < mean - 1.96 * std_dev else 0, axis=1)
        sub_df['spike'] = sub_df.apply(lambda r: 1 if r['Pos'] == 1 or r['Neg'] == -1 else 0, axis=1)
        df.loc[df['Period'] == period, 'cm_pred_spike'] = sub_df.loc[:, 'spike'].values
        df.loc[df['Period'] == period, 'cm_pred_spike_dir'] = sub_df.loc[:, 'Pos'].values + sub_df.loc[:, 'Neg'].values

    # Extract test dataframe, reset index to 0 - 8735 instead of 1440 - 10775
    df_test = df_test_plus_60[1440:].reset_index()
    # Extract array of test plus 60 days data
    data = df_test_plus_60['System Price'].values

    # Find true spikes with direction
    for i in range(0, len(df_test)):
        df_test.loc[i, 'true_spike_dir'] = check_if_spike(df_test.loc[i, 'System Price'], data[i:(i + 1440)])
    df_test.drop(columns={'System Price', 'index'}, inplace=True)
    # Add true spikes without direction
    df_test['true_spike'] = df_test.apply(lambda c: 1 if c['true_spike_dir'] != 0 else 0, axis=1)

    # Add true spike data to forecasted data df
    df_new = df.merge(df_test, on='Datetime', how='left')

    # Check whether price covered by PI
    df_new['ind'] = df_new.apply(lambda c: indicator_function(c['Lower'], c['Upper'], c['System Price']), axis=1)

    # Check whether spike predicted correctly
    df_new['spike_pred_correct'] = df_new.apply(lambda c: 1 if c['cm_pred_spike'] == c['true_spike'] else 0, axis=1)

    # Check whether spike covered by interval
    df_new['spike_cov'] = df_new.apply(lambda c: 1 if c['ind'] == 1 and c['true_spike'] != 0 else 0, axis=1)

    # Drop unnecessary data
    df_new.drop(columns={'Period', 'System Price', 'Forecast', 'Upper', 'Lower', 'forecast_hour'}, inplace=True)

    #Analyze whether spikes are more frequent further into the horizon in both predicted and analyzed data
    print('Spike occurence frequency in Curve Model forecast and true price over all data')
    spike_hor_freq = df_new.groupby(['forecast_day']).mean()
    print(spike_hor_freq.round(4))
    # spike_hor_freq[['cm_pred_spike', 'true_spike', 'spike_cov']]
    fig, ax = plt.subplots(figsize=(13, 4))
    x_values = spike_hor_freq.index
    ax.plot(x_values, spike_hor_freq['true_spike']*100, label='True', color='steelblue')
    ax.plot(x_values, spike_hor_freq['cm_pred_spike']*100, label='Curve Model', color='firebrick')
    #ax.plot(x_values, spike_hor_freq['spike_cov'], label='Spike coverage', color='darkorange')
    ax.set_title("True Average vs. Model Average Spike Frequency per Day", size=14, y=1.08)
    ax.set_xticks(range(1, 15))
    ax.set_xlabel('Day of prediction horizon', size=11, labelpad=12)
    ax.set_ylabel('Frequency [%]', size=11, labelpad=12)
    ax.set_ylim(ax.get_ylim()[0]*0.9, ax.get_ylim()[1]*1.1)
    for line in ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.07),
                          fancybox=True, shadow=True, prop={'size': 11}).get_lines():
        line.set_linewidth(2)
    # ax.title(metric, pad=title_pad)
    fig.tight_layout()
    plt.savefig("../analysis/CurveModel/spike_freq.png")
    assert False
    # Analysis from here on only with spike data points

    # Only predicted spikes på CM
    df_cm_spikes = df_new[df_new['cm_pred_spike'] > 0]
    print('-------------------------------------')
    print('Predicted spikes analysis')
    #print(df_cm_spikes)
    df_cm_spikes_mean = df_cm_spikes[['true_spike', 'ind', 'spike_pred_correct', 'spike_cov', 'forecast_day']].groupby(['forecast_day']).mean()
    print(df_cm_spikes_mean.round(4))
    print(df_cm_spikes_mean.describe())
    #df_cm_spikes_cnt = df_cm_spikes[['true_spike', 'spike_pred_correct']].groupby(['forecast_day']).count()
    # print(df_cm_spikes_cnt)

    # Only true spikes
    print('-------------------------------------')
    print('True spikes analysis')
    df_true_spikes = df_new[df_new['true_spike'] != 0]
    #print(df_true_spikes)
    print(df_true_spikes[['true_spike', 'ind', 'spike_pred_correct', 'spike_cov', 'forecast_day']].groupby(['forecast_day']).mean())


    # Drop all data points not predicted a spike by CM or that is a true spike
    print('-------------------------------------')
    print('True and predicted spikes analysis')
    df_spikes = df_new[(df_new['cm_pred_spike'] > 0) | (df_new['true_spike'] > 0)]
    # print(df_spikes)
    print('MEAN')
    print(df_spikes.groupby(['forecast_day']).mean())


    #df_day = df_spikes.groupby(['forecast_day']).mean()
    return df_spikes


def check_if_spike(price, window_prices):  # Helping method
    spike = 0
    threshold = 1.96
    mean = window_prices.mean()
    std_dev = window_prices.std()
    if price > mean + threshold * std_dev:
        spike = 1
    elif price < mean - threshold * std_dev:
        spike = -1
    return spike


def find_trends():
    df = pd.read_csv(r'../results/test/CurveModel/forecast.csv', delimiter=',', header=0,
                    usecols=['Date', 'Hour', 'Period', 'System Price', 'Forecast', 'Upper', 'Lower'],
                    parse_dates=[['Date', 'Hour']])
    df.rename(columns={'Date_Hour': 'Datetime'}, inplace=True)
    df = indecise(df)
    df[['trend', 'cm_trend']] = 0
    # 'trend_w1', 'trend_w2', 'cm_trend', 'cm_trend_w1', 'cm_trend_w2'
    # find and remove periods with spikes
    df_test_plus_60 = get_data("04.04.2019", "31.05.2020", ["System Price"], os.getcwd(), "h")
    df_test_plus_60['Datetime'] = pd.to_datetime(df_test_plus_60.Date) + df_test_plus_60.Hour.astype('timedelta64[h]')
    df_test_plus_60.drop(columns={'Date', 'Hour'}, inplace=True)
    df_test = df_test_plus_60[1440:].reset_index()
    # Extract array of test plus 60 days data
    data = df_test_plus_60['System Price'].values
    for i in range(0, len(df_test)):
        df_test.loc[i, 'true_spike_dir'] = check_if_spike(df_test.loc[i, 'System Price'], data[i:(i + 1440)])
    df_test.drop(columns={'System Price', 'index'}, inplace=True)

    # Add true spikes without direction
    df_test['true_spike'] = df_test.apply(lambda c: 1 if c['true_spike_dir'] != 0 else 0, axis=1)
    df_test.drop(columns={'true_spike_dir'}, inplace=True)
    df = df.merge(df_test, on='Datetime', how='left')
    for period in df["Period"].unique():
        sub_df = df[df["Period"] == period].reset_index(drop=True)
        if sub_df['true_spike'].any() > 0:
            df = df[df['Period'] != period]
            continue
        prev_lvl = sub_df.head(168)["System Price"].median()
        df.loc[df['Period'] == period, "trend"] = sub_df.tail(336)["System Price"].median() - prev_lvl
        # df.loc[df['Period'] == period, "trend_w1"] = sub_df.loc[168:335, "System Price"].median() - prev_lvl
        # df.loc[df['Period'] == period, "trend_w2"] = sub_df.tail(168)["System Price"].median() - prev_lvl
        df.loc[df['Period'] == period, "cm_trend"] = sub_df["Forecast"].median() - prev_lvl
        # df.loc[df['Period'] == period, "cm_trend_w1"] = sub_df.head(168)["Forecast"].median() - prev_lvl
        # df.loc[df['Period'] == period, "cm_trend_w2"] = sub_df.tail(168)["Forecast"].median() - prev_lvl

    # Check whether price covered by PI
    df['ind'] = df.apply(lambda c: indicator_function(c['Lower'], c['Upper'], c['System Price']), axis=1)
    df.drop(columns={'true_spike', 'Upper', 'Lower', 'Forecast', 'System Price'}, inplace=True)
    # Calculate difference in model forecast trend and true trend
    df['diff'] = df.apply(lambda c: abs(c['trend'] - c['cm_trend']), axis=1)
    print('-------------------------------------')
    print('Trend analysis')
    print(df.describe())
    print(df.groupby(['forecast_day']).mean())


if __name__ == '__main__':
    df_day_ = get_spikes_per_day()
    assert False
    find_trends()
    pd.set_option('display.width', 100000)
    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.min_rows', 1000)
    pd.set_option('display.max_columns', 100)
    # df_hour_cm, df_day_cm = get_error_distribution_by_horizon('CurveModel')
    # df_hour_nd, df_day_nd = get_error_distribution_by_horizon('NaiveDay')
    # df_hour_sa, df_day_sa = get_error_distribution_by_horizon('Sarima')
    # plot_horizon_error(df_hour_cm, df_day_cm, 'Forecast hour')
    # plot_horizon_error(df_day_cm, df_day_nd, 'Forecast day')
    assert False

    df_day_ = get_demand_forecast_errors()

    np.set_printoptions(linewidth=50000000)
    # print(df_day_['AE'].values.transpose())
    # print(df_day_['APE'].values.transpose())
    mae = np.around(df_day_['AE'].values.transpose(), 0)
    mape = np.around(df_day_['APE'].values.transpose(), 2)
    rmse = np.around(df_day_['se'].values.transpose(), 0)
    print(mape)

    week_1 = [np.mean(mae[:7]), np.mean(mape[:7]), np.mean(rmse[:7])]
    week_2 = [np.mean(mae[7:]), np.mean(mape[7:]), np.mean(rmse[7:])]
    print(week_1)
    print(week_2)

    fig, ax1 = plt.subplots(figsize=(13, 7))

    x = df_day_.index

    label_col = 'black'
    color1 = 'steelblue'
    ax1.set_xlabel('Forecast hour')
    ax1.set_ylabel('€', color=label_col)
    ax1.plot(x, df_day_['AE'], color=color1)
    ax1.plot(x, df_day_['se'], color='firebrick')
    ax1.tick_params(axis='y', labelcolor=label_col)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('%', color=label_col)  # we already handled the x-label with ax1
    ax2.plot(x, df_day_['APE'], color='darkorange')
    ax2.tick_params(axis='y', labelcolor=label_col)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

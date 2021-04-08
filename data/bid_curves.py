import os
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_handler import get_data
from data_handler import get_auction_data
from shapely.geometry import LineString
import random

label_pad = 12
title_pad = 20
full_fig = (13, 7)
half_fig = (6.5, 7)

first_color = "steelblue"
sec_color = "firebrick"
third_color = "darkorange"
fourth_color = "mediumseagreen"
fifth_color = "silver"
sixth_color = "palevioletred"
seventh_color = "teal"
seven_colors = [first_color, sec_color, third_color, fourth_color, fifth_color, sixth_color, seventh_color]


def auction_data():
    raw_folder = "input/auction/raw"
    vol_price_df = get_data("01.07.2014", "31.12.2020", ["Total Vol", "System Price"], os.getcwd(), "h")
    net_flow_df = pd.DataFrame(
        columns=["Date", "Hour", "Acs demand", "Acs supply", "Net flow", "Total Volume", "Int Volume"])
    vol_price_df["Full vol"] = np.nan
    vol_price_df["Full price"] = np.nan
    vol_price_df["Disc vol"] = np.nan
    vol_price_df["Disc price"] = np.nan
    for file in sorted(Path(raw_folder).iterdir()):  # list all raw xls files
        date_string = str(file).split("\\")[3].split("_")[-1][:-4]
        date = dt.strptime(date_string, "%Y-%m-%d").date()
        print(date)
        vol_price_df_date = vol_price_df[vol_price_df["Date"].dt.date == date]
        if not os.path.exists('output/plots/all_bids/{}'.format(date)):
            os.makedirs('output/plots/all_bids/{}'.format(date))
        df_result = pd.DataFrame()
        lower_lim = -10
        upper_lim = 210
        df_result["Price"] = range(lower_lim, upper_lim + 1)
        df_raw = pd.read_excel(file)
        df_raw.columns = [str(i) for i in df_raw.columns]
        all_columns = [i for i in df_raw.columns]
        all_hour_columns = [i for i in all_columns if "Bid curve chart data (Reference time)" not in i]
        for i in range(len(all_hour_columns)):
            cols = [i * 2, i * 2 + 1]
            df_h = df_raw[df_raw.columns[cols]]
            hour = int(df_h.columns[1].split(" ")[1][0:2])
            print("Hour {}".format(hour))
            row_index = vol_price_df.loc[(vol_price_df["Date"].dt.date == date) & (vol_price_df["Hour"] == hour)].index[
                0]
            exc_volume = vol_price_df_date[vol_price_df_date["Hour"] == hour]["Total Vol"].tolist()[0]
            exc_price = vol_price_df_date[vol_price_df_date["Hour"] == hour]["System Price"].tolist()[0]
            idx_vol_demand, idx_vol_supply, idx_net_flow, acs_demand, acs_supply, net_flows, idx_bid_start, \
                = get_initial_info_raw(df_h)
            add_flow_to_demand = net_flows < 0
            add_flow_to_supply = net_flows > 0
            df_buy, df_sell = get_buy_and_sell_dfs(df_h, idx_bid_start)
            df_d = get_auction_df(df_buy, add_flow_to_demand, lower_lim, upper_lim, acs_demand, net_flows)
            demand_line_full = LineString(np.column_stack((df_d["Volume"], df_d["Price"])))
            df_d = get_discrete_bid_df(df_d, lower_lim, upper_lim)
            demand_line_discrete = LineString(np.column_stack((df_d["Volume"], df_d["Price"])))
            plt.subplots(figsize=full_fig)
            plt.plot(df_d["Volume"], df_d["Price"], label="Demand", zorder=1)
            df_s = get_auction_df(df_sell, add_flow_to_supply, lower_lim, upper_lim, acs_supply, net_flows)
            supply_line_full = LineString(np.column_stack((df_s["Volume"], df_s["Price"])))
            df_s = get_discrete_bid_df(df_s, lower_lim, upper_lim)
            demand_col_name = "Demand h {}".format(hour)
            if demand_col_name in df_result.columns:
                demand_col_name = demand_col_name + "_2"
            supply_col_name = "Supply h {}".format(hour)
            if supply_col_name in df_result.columns:
                supply_col_name = demand_col_name + "_2"
            df_result[demand_col_name] = df_d["Volume"]
            df_result[supply_col_name] = df_s["Volume"]
            supply_line_discrete = LineString(np.column_stack((df_s["Volume"], df_s["Price"])))
            plt.plot(df_s["Volume"], df_s["Price"], label="Supply", zorder=2)
            full_intersect = supply_line_full.intersection(demand_line_full)
            full_vol = full_intersect.x
            full_price = full_intersect.y
            vol_price_df.loc[row_index, "Full vol"] = full_vol
            vol_price_df.loc[row_index, "Full price"] = full_price
            discrete_intersect = supply_line_discrete.intersection(demand_line_discrete)
            if type(discrete_intersect) == LineString:
                disc_price = upper_lim
                disc_vol = max(df_d["Volume"])
                print("In date {} hour {} we found no intersection".format(date, hour))
            else:
                disc_vol = discrete_intersect.x
                disc_price = discrete_intersect.y
            vol_price_df.loc[row_index, "Disc vol"] = disc_vol
            vol_price_df.loc[row_index, "Disc price"] = disc_price
            plt.scatter(disc_vol, disc_price, color="red", zorder=3, s=140,
                        label="Disc. intersect ({:.2f}, {:.2f})".format(disc_vol, disc_price))
            plt.scatter(exc_volume, exc_price, color="green", zorder=4,
                        label="True intersect ({:.2f}, {:.2f})".format(exc_volume, exc_price))
            plt.legend()
            plt.ylim(lower_lim, max(100, disc_price * 1.1))
            plt.ylabel("Price")
            plt.xlabel("Volume")
            plt.title("Discrete Bid Curves {} - {}".format(date, hour))
            plt.tight_layout()
            save_curve_path = 'output/plots/all_bids/{}/{}.png'.format(date, hour)
            if os.path.exists(save_curve_path):
                plt.savefig('output/plots/all_bids/{}/{}_2'.format(date, hour))
            else:
                plt.savefig('output/plots/all_bids/{}/{}'.format(date, hour))
            plt.close()
            flow_dict = {"Date": date, "Hour": hour, "Acs demand": acs_demand, "Acs supply": acs_supply,
                         "Net flow": net_flows, "Total Volume": exc_volume, "Int Volume": disc_vol}
            net_flow_df = net_flow_df.append(flow_dict, ignore_index=True)
        cols = [i for i in df_result.columns if "Demand" in i or "Supply" in i]
        df_result[cols] = df_result[cols].astype(float).round(2)
        save_disc_path = "input/auction/csv_disc/{}.csv".format(date)
        df_result.to_csv(save_disc_path, index=False, float_format='%.2f')
    vol_price_df = vol_price_df.dropna()
    vol_price_df.to_csv("output/auction/vol_price_auction.csv", index=False, float_format='%.2f')
    net_flow_df.to_csv("output/auction/volume_analyses.csv", index=False, float_format='%.2f')


def get_original_bid_methods():  # Used in other script
    return get_initial_info_raw, get_buy_and_sell_dfs, get_auction_df


def get_initial_info_raw(df):  # Helping method
    idx_vol_demand = \
        df[df[df.columns[0]] == "Bid curve chart data (Volume for accepted blocks buy)"].index[0]
    idx_vol_supply = \
        df[df[df.columns[0]] == "Bid curve chart data (Volume for accepted blocks sell)"].index[0]
    idx_net_flow = df[df[df.columns[0]] == "Bid curve chart data (Volume for net flows)"].index[0]
    acs_demand = df.loc[idx_vol_demand, df.columns[1]]
    acs_supply = df.loc[idx_vol_supply, df.columns[1]]
    net_flows = df.loc[idx_net_flow, df.columns[1]]
    idx_bid_start = df[df[df.columns[0]] == "Buy curve"].index[0] + 1
    return idx_vol_demand, idx_vol_supply, idx_net_flow, acs_demand, acs_supply, net_flows, idx_bid_start


def get_buy_and_sell_dfs(df_h, idx_bid_start):  # Helping method
    df_h = df_h.iloc[idx_bid_start:].reset_index(drop=True)
    index_sell = df_h[df_h["Category"] == "Sell curve"].index[0]
    df_h = df_h.rename(columns={df_h.columns[0]: 'Category', df_h.columns[1]: 'Value'})
    df_h = df_h.dropna(how='all')
    df_buy = df_h[0:index_sell]
    df_sell = df_h[index_sell + 1:]
    return df_buy, df_sell


def get_auction_df(df, add_flow, lower_lim, upper_lim, acs_vol, net_flows):  # Helping method
    df_vol = pd.DataFrame(columns=["Price", "Volume"])
    df_vol["Price"] = df[df["Category"] == "Price value"]["Value"].tolist()
    df_vol["Volume"] = df[df["Category"] == "Volume value"]["Value"].tolist()
    df_vol = df_vol[lower_lim <= df_vol["Price"]]
    df_vol = df_vol[upper_lim >= df_vol["Price"]].reset_index(drop=True)
    df_vol["Volume"] = df_vol["Volume"] + acs_vol
    if add_flow:
        df_vol["Volume"] = df_vol["Volume"] + abs(net_flows)
    return df_vol


def get_discrete_bid_df(df, lower, upper):  # Helping method
    price_range = range(lower, upper + 1)
    result = pd.DataFrame(columns=df.columns)
    result["Price"] = price_range
    for i in range(len(result)):
        closest = df.iloc[(df["Price"] - price_range[i]).abs().argsort()[:1]].reset_index(drop=True).loc[0, "Volume"]
        result.loc[i, "Volume"] = closest
    return result


def rename_folders_from_raw():
    raw_folder = "input/auction/raw_s"
    # dates = [(dt(2015, 1, 1) + timedelta(days=x)).date() for x in range(365)]
    all_files = sorted(Path(raw_folder).iterdir())
    for file in all_files:  # list all raw xls files
        date_string = str(file).split("_")[-3][:-3]
        date = dt.strptime(date_string, "%d-%m-%Y").date()
        print(date)
        # dates.remove(date)
        first_path = "\\".join(str(file).split("\\")[0:3])
        mcp_name = "_".join(str(file).split("\\")[3].split("_")[0:3])
        save_path = first_path + "\\" + mcp_name + "_" + str(date) + ".xls"
        os.rename(file, save_path)
    # print(dates)


def min_max_price():
    price = get_data("01.01.2014", "31.12.2020", ["System Price"], os.getcwd(), "h")["System Price"].tolist()
    max_price = max(price)
    min_price = min(price)
    print("Max {}, min {}".format(max_price, min_price))


def plot_mean_curves():
    folder = "input/auction/csv_disc"
    title_date = "(01.07.2014 - 02.06.2020)"
    price_range = range(-10, 211)
    df_demand = pd.DataFrame(columns=price_range).transpose()
    df_demand["Mean demand"] = np.PZERO
    df_supply = pd.DataFrame(columns=price_range).transpose()
    df_supply["Mean supply"] = np.PZERO
    no = 0
    for file in sorted(Path(folder).iterdir()):  # list all csv files
        date_string = str(file).split("\\")[3].split("_")[-1][:-4]
        print(date_string)
        data = pd.read_csv(file)
        for i in range(int((len(data.columns) - 1) / 2)):
            no += 1
            demand = data[["Price"] + [data.columns[1 + i * 2]]].set_index("Price").rename(
                columns={data.columns[1 + i * 2]: "Demand" + str(no)})
            supply = data[["Price"] + [data.columns[1 + i * 2 + 1]]].set_index("Price").rename(
                columns={data.columns[1 + i * 2 + 1]: "Supply" + str(no)})
            df_demand = df_demand.merge(demand, left_index=True, right_index=True)
            df_supply = df_supply.merge(supply, left_index=True, right_index=True)
            df_demand["Mean demand"] = (no - 1) / no * df_demand["Mean demand"] + (1 / no) * demand[
                "Demand{}".format(no)]
            df_supply["Mean supply"] = (no - 1) / no * df_supply["Mean supply"] + (1 / no) * supply[
                "Supply{}".format(no)]
            df_demand = df_demand[["Mean demand"]]
            df_supply = df_supply[["Mean supply"]]
    plt.subplots(figsize=full_fig)
    plt.plot(df_demand["Mean demand"], df_demand.index, label="Avg. demand")
    plt.title("Average Demand Curve " + title_date, pad=title_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/plots/eda_bids/mean_demand.png")
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.title("Average Supply Curve " + title_date, pad=title_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.plot(df_supply["Mean supply"], df_supply.index, label="Avg. supply", color="orange")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/plots/eda_bids/mean_supply.png")
    plt.close()
    plt.subplots(figsize=full_fig)
    plt.plot(df_demand["Mean demand"], df_demand.index, label="Avg. demand")
    plt.plot(df_supply["Mean supply"], df_supply.index, label="Avg. supply", color="orange")
    plt.title("Average Market Curves " + title_date, pad=title_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/plots/eda_bids/mean_curves.png")
    plt.close()
    df = pd.DataFrame()
    df["Price"] = range(-10, 211)
    df["Mean supply"] = df_supply["Mean supply"].reset_index(drop=True).round(2)
    df["Mean demand"] = df_demand["Mean demand"].reset_index(drop=True).round(2)
    df.to_csv("output/auction/mean_curves.csv", index=False)


def eda_disc_auction_data(overview, make_analysis_csv):
    if overview:
        all_dates = [i.strftime("%Y-%m-%d") + ".csv" for i in
                     pd.date_range(dt(2014, 7, 1), dt(2020, 6, 3) - timedelta(days=1), freq='d')]
        all_files = sorted(Path("input/auction/csv_disc").iterdir())
        all_end_files = [str(i).split("\\")[-1] for i in all_files]
        print("--------------------------------\n")
        print("There are {} csv-files in folder ranging {} dates".format(len(all_end_files), len(all_dates)))
        for d in all_dates:
            if d not in all_end_files:
                print("Missing csv file for {}".format(d))
        number_of_auctions = len(all_dates) * 24 + 1  # adding one as 2014 has +1 hour but no -1 hour
        print("Number of auctions during period: {}".format(number_of_auctions))
        prices_vol = get_data("01.07.2014", "31.12.2020", ["System Price", "Total Vol"], os.getcwd(), "h")
        max_p = prices_vol[prices_vol["System Price"] == prices_vol["System Price"].max()].iloc[0]
        print("Max price: {} h{}:\t {} (volume {})".format(max_p["Date"].date(), max_p["Hour"], max_p["System Price"],
                                                           max_p["Total Vol"]))
        min_p = prices_vol[prices_vol["System Price"] == prices_vol["System Price"].min()].iloc[0]
        print("Min price: {} h{}:\t {} (volume {})".format(min_p["Date"].date(), min_p["Hour"], min_p["System Price"],
                                                           min_p["Total Vol"]))
        max_v = prices_vol[prices_vol["Total Vol"] == prices_vol["Total Vol"].max()].iloc[0]
        print("Max vol: {} h{}:\t {} (price {})".format(max_v["Date"].date(), max_v["Hour"], max_v["Total Vol"],
                                                        max_v["System Price"]))
        min_v = prices_vol[prices_vol["Total Vol"] == prices_vol["Total Vol"].min()].iloc[0]
        print("Min vol: {} h{}:\t {} (price {})".format(min_v["Date"].date(), min_v["Hour"], min_v["Total Vol"],
                                                        min_v["System Price"]))
        print("--------------------------------\n")
    if make_analysis_csv:
        years = range(2014, 2021)
        all_files = sorted(Path("input/auction/csv_disc").iterdir())
        true_df = get_data("01.07.2014", "31.12.2020", ["System Price", "Total Vol"], os.getcwd(), "h")
        df = pd.DataFrame(columns=["Date", "Hour", "True price", "True vol", "Disc price", "Disc vol"])
        for f in all_files:
            date = dt.strptime(str(f).split("\\")[-1][:-4], "%Y-%m-%d").date()
            print(date)
            data = pd.read_csv(f)
            for i in range(int((len(data.columns) - 1) / 2)):
                demand = data[["Price"] + [data.columns[1 + i * 2]]]
                hour = demand.columns[1].split(" ")[-1]
                if "_" not in hour:
                    row = true_df[(true_df["Date"].dt.date == date) & (true_df["Hour"] == int(hour))].reset_index(
                        drop=True)
                    demand_line = LineString(np.column_stack((demand["Price"], demand[demand.columns[1]])))
                    supply = data[["Price"] + [data.columns[1 + i * 2 + 1]]]
                    supply_line = LineString(np.column_stack((supply["Price"], supply[supply.columns[1]])))
                    intersect = supply_line.intersection(demand_line)
                    new = {"Date": date, "Hour": hour, "True price": row.loc[0, "System Price"],
                           "True vol": row.loc[0, "Total Vol"],
                           "Disc price": round(intersect.x, 4), "Disc vol": round(intersect.y, 2)}
                    df = df.append(new, ignore_index=True)
                    df.to_csv("output/auction/vol_price_disc_analysis.csv", index=False)
    df = pd.read_csv("output/auction/vol_price_disc_analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Mae price"] = abs(df["True price"] - df["Disc price"])
    df["Mape price"] = 100 * df["Mae price"] / df["True price"]
    df["Mae vol"] = abs(df["True vol"] - df["Disc vol"])
    df["Mape vol"] = 100 * df["Mae vol"] / df["True vol"]
    df_wrong_2020 = df[df["Date"] >= dt(2020, 6, 3)]
    df = df[df["Date"] < dt(2020, 6, 3)]
    print("MAE price discrete bids: {:.3f}".format(df["Mae price"].mean()))
    print("MAPE price discrete bids: {:.3f}%".format(df["Mape price"].mean()))
    print("MAE volume discrete bids: {:.3f}".format(df["Mae vol"].mean()))
    print("MAPE volume discrete bids: {:.3f}\n".format(df["Mape vol"].mean()))
    df_grouped = df.groupby(by=df["Date"].dt.year).mean()[["Mae price", "Mape price", "Mae vol", "Mape vol"]]
    # print(df_grouped)
    print("MAE price discrete bids wrong 2020: {:.3f}".format(df_wrong_2020["Mae price"].mean()))
    print("MAE volume discrete bids wrong 2020: {:.3f}\n".format(df_wrong_2020["Mae vol"].mean()))


def investigate_unit_price_information_loss():
    df_2014 = pd.read_csv("output/vol_price_auction_2014.csv")
    df_2019 = pd.read_csv("output/vol_price_auction_2019.csv")
    df_2020 = pd.read_csv("output/vol_price_auction_2020.csv")
    for year, df in {2014: df_2014, 2019: df_2019, 2020: df_2020}.items():
        print("\n{} ----------------------".format(year))
        df["full_vol_dev"] = 100 * (df["Total Vol"] - df["Full vol"]) / df["Total Vol"]
        print("Full: Mean error in volume {:.3f}%".format(df["full_vol_dev"].mean()))
        print("Full: Abs mean error in volume {:.3f}%".format(abs(df["full_vol_dev"]).mean()))
        df["disc_vol_dev"] = 100 * (df["Total Vol"] - df["Disc vol"]) / df["Total Vol"]
        print("Disc: Mean error in volume {:.3f}%".format(df["disc_vol_dev"].mean()))
        print("Disc: Abs mean error in volume {:.3f}%".format(abs(df["disc_vol_dev"]).mean()))
        df["full_price_dev"] = 100 * (df["System Price"] - df["Full price"]) / df["System Price"]
        print("Full: Mean error in price {:.3f}%".format(df["full_price_dev"].mean()))
        print("Full: Abs mean error in price {:.3f}%".format(abs(df["full_price_dev"]).mean()))
        df["disc_price_dev"] = 100 * (df["System Price"] - df["Disc price"]) / df["System Price"]
        print("Disc: Mean error in price {:.3f}%".format(df["disc_price_dev"].mean()))
        print("Disc: Abs mean error in price {:.3f}%".format(abs(df["disc_price_dev"]).mean()))


def make_price_classes_from_mean_curves(number_of_demand, number_of_supply):
    demand_prices, demand_volumes = get_price_classes_from_mean("demand", number_of_demand)
    print("{} price classes demand from mean curve: \n{}".format(number_of_demand, demand_prices))
    plot_classes(demand_prices, demand_volumes, "demand")
    supply_prices, supply_volumes = get_price_classes_from_mean("supply", number_of_supply)
    print("{} price classes supply from mean curve: \n{}".format(number_of_supply, supply_prices))
    plot_classes(supply_prices, supply_volumes, "supply")


def get_price_classes_from_mean(curve, number_of_classes):  # helping method
    mean_curves = pd.read_csv("output/auction/mean_curves.csv")
    df = mean_curves[["Price", "Mean {}".format(curve)]]
    line = LineString(np.column_stack((df["Mean {}".format(curve)], df["Price"])))
    min_vol = df["Mean {}".format(curve)].min()
    max_vol = df["Mean {}".format(curve)].max()
    volume_diff = max_vol - min_vol
    interval_step = volume_diff / (number_of_classes - 1)
    vol_intervals = [max_vol - i * interval_step for i in range(1, number_of_classes - 1)]
    if curve == "demand":
        price_classes = [-10]
    elif curve == "supply":
        price_classes = [210]
    for vol in vol_intervals:
        yline = LineString([(vol, -10), (vol, 210)])
        price = line.intersection(yline).y
        price_classes.append(price)
    volumes = [max_vol - i * interval_step for i in range(number_of_classes)]
    if curve == "demand":
        price_classes.append(210)
    elif curve == "supply":
        price_classes.append(-10)
        price_classes.reverse()
        volumes.reverse()
    return [round(i, 2) for i in price_classes], [round(j, 2) for j in volumes]


def plot_classes(prices, volumes, curve):  # helping method
    mean_curves = pd.read_csv("output/auction/mean_curves.csv")
    df = mean_curves[["Price", "Mean {}".format(curve)]]
    plt.subplots(figsize=full_fig)
    out_path = "output/auction/price_classes/mean_curve_{}_{}.png".format(curve, len(prices))
    plt.title("{} Price Classes Defined by {} Equal Size Volumes".format(curve.capitalize(), len(prices)),
              pad=title_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    color_int = 0 if curve == "demand" else 1
    c_map = plt.get_cmap("tab10")
    color = c_map(color_int)
    plt.plot(df[df.columns[1]], df["Price"], color=color, label="Mean {} curve".format(curve), linewidth=4)
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    for i in range(len(volumes)):
        vol = volumes[i]
        price = prices[i]
        label = "Volume classes" if i == 0 else '_nolegend_'
        plt.vlines(vol, y_lim[0], price, label=label, color="grey", linestyles="dotted")
        label = "Price classes" if i == 0 else '_nolegend_'
        plt.hlines(price, x_lim[0], vol, label=label, color="grey", linestyles="solid")
    for line in plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.ylim(y_lim[0], y_lim[1])
    plt.xlim(x_lim[0], x_lim[1])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def get_auction_subset_for_evaluation():  # Helping method
    number_of_auctions = 51937
    res_length = 1000  # pick out thousand random auctions
    random.seed(1)
    random_list = random.sample(range(number_of_auctions), res_length)
    all_hours = [i for i in pd.date_range(dt(2014, 7, 1), dt(2020, 6, 3), freq='h')]
    chosen_hours = [all_hours[i] for i in random_list]
    return chosen_hours


def create_auction_subset():
    chosen_hours = get_auction_subset_for_evaluation()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(full_fig[0] + 4.5, full_fig[1]))
    ax1.set_title("Sample Distribution of Auctions for Price Class Evaluation - Month and Year", pad=title_pad)
    ax2.set_title("Sample Distribution of Auctions for Price Class Evaluation - Hour of Day and Year", pad=title_pad)
    df_1, max_y_1 = create_auction_get_df(range(1, 13), chosen_hours, len(chosen_hours), "month")
    columns_1 = [dt(2020, int(month), 1).strftime('%b') for month in df_1.columns]
    df_1.columns = columns_1
    plot_df_auction_dist(df_1, ax1, max_y_1, "Month")
    df_2, max_y_2 = create_auction_get_df(range(24), chosen_hours, len(chosen_hours), "hour")
    plot_df_auction_dist(df_2, ax2, max_y_2, "Hour of day")
    plt.tight_layout()
    plt.savefig("output/auction/price_classes/sample_dist_{}.png".format(len(chosen_hours)))
    print("Plot saved to output/auction/price_classes/sample_dist_{}.png".format(len(chosen_hours)))
    plt.close()


def create_auction_get_df(col_range, chosen_hours, total_auctions, time):  # helping method
    df = pd.DataFrame(columns=["Year"] + [str(i) for i in col_range])
    for year in range(2014, 2021):
        row = {"Year": year}
        for col in df.columns:
            if col != "Year":
                row[col] = 100 * len(
                    [i for i in chosen_hours if getattr(i, time) == int(col) and i.year == year]) / total_auctions
        df = df.append(row, ignore_index=True)
    df.index = df["Year"].astype(int)
    df = df[[i for i in df.columns if i != "Year"]]
    max_y = max([df[i].sum() for i in df.columns])
    return df, max_y


def plot_df_auction_dist(df, ax, max_y, label):  # helping method
    df.transpose().plot(kind="bar", stacked=True, ax=ax, color=seven_colors)
    for line in ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                          shadow=True).get_lines():
        line.set_linewidth(1)
    ax.axis(ymin=0, ymax=max_y * 1.1)
    ax.set_xlabel(label, labelpad=label_pad)
    ax.set_ylabel("Proportion of sample [%]", labelpad=label_pad)


def evaluate_mean_price_classes(create_csv):
    if os.path.exists("output/auction/price_class_evaluation.csv"):
        df = pd.read_csv("output/auction/price_class_evaluation.csv")
    else:
        df = pd.DataFrame(columns=["Amount", "Demand classes", "Supply classes", "Demand MAE", "Supply MAE"])
    if create_csv:
        for i in range(8, 37):
            demand_classes, _ = get_price_classes_from_mean("demand", i)
            supply_classes, _ = get_price_classes_from_mean("supply", i)
            chosen_hours = get_auction_subset_for_evaluation()
            mae_demand, mae_supply = evaluate_price_class(demand_classes, supply_classes, chosen_hours)
            row = {"Amount": i, "Demand classes": demand_classes, "Supply classes": supply_classes,
                   "Demand MAE": mae_demand, "Supply MAE": mae_supply}
            df = df.append(row, ignore_index=True)
            df.to_csv("output/auction/price_class_evaluation.csv", index=False)
    df["Improvement Demand"] = df["Demand MAE"].shift(1) - df["Demand MAE"]
    df["Improvement Supply"] = df["Supply MAE"].shift(1) - df["Supply MAE"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(full_fig[0] + 3, full_fig[1]))
    plot_mae_improvement(ax1, "demand", df, 16)
    plot_mae_improvement(ax2, "supply", df, 18)
    plt.tight_layout()
    plt.savefig("output/auction/price_classes/mae_mean_curves.png")


def plot_mae_improvement(ax, curve, df, opt):
    col_index = 0 if curve == "demand" else 1
    color = plt.get_cmap("tab10")(col_index)
    ax.plot(df["Amount"], df["{} MAE".format(curve.capitalize())], color=color, label="MAE".format(curve.capitalize()),
            linewidth=2)
    ax.plot(df["Amount"], df["Improvement {}".format(curve.capitalize())], color=color, linestyle="dotted",
            linewidth=3, label="Marginal MAE benefit".format(curve.capitalize()))
    y_min, y_max = ax.get_ylim()
    ax.axvline(opt, y_min, y_max, color="red", label="Opt. = {}".format(opt), linestyle="dotted")
    ax.set_ylim(y_min, y_max)
    for line in ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                          shadow=True).get_lines():
        line.set_linewidth(2)
    ax.set_title("MAE per Number of {} Price Classes".format(curve.capitalize()), pad=title_pad)
    ax.set_ylabel("MAE [MWh]", labelpad=label_pad)
    ax.set_xlabel("Number of price classes", labelpad=label_pad)


def evaluate_price_class(demand, supply, hours):  # Helping method
    print("Evaluation\t-------------------------------------------------------")
    print("Demand prices:\t{}".format(demand))
    print("Supply prices:\t{}".format(supply))
    supply_avg_error = pd.DataFrame(columns=[i for i in range(-10, 211)])
    demand_avg_error = pd.DataFrame(columns=[i for i in range(-10, 211)])
    maes_demand_full = []
    maes_supply_full = []
    for auction in hours:
        mae_demand = []
        mae_supply = []
        demand_volumes = []
        supply_volumes = []
        file_path = "input/auction/csv_disc/{}.csv".format(auction.strftime("%Y-%m-%d"))
        hour = auction.hour
        columns = ["Price", "Demand h {}".format(hour), "Supply h {}".format(hour)]
        data = pd.read_csv(file_path, usecols=columns)
        data = data.rename(columns={"Demand h {}".format(hour): "Demand", "Supply h {}".format(hour): "Supply"})
        demand_line = LineString(np.column_stack((data["Demand"], data["Price"])))
        supply_line = LineString(np.column_stack((data["Supply"], data["Price"])))
        min_vol_demand = data["Demand"].min()
        max_vol_demand = data["Demand"].max()
        min_vol_supply = data["Supply"].min()
        max_vol_supply = data["Supply"].max()
        for demand_price in demand:
            h_line = LineString([(min_vol_demand, demand_price), (max_vol_demand, demand_price)])
            demand_volumes.append(round(demand_line.intersection(h_line).x, 2))
        est_demand_line = LineString(np.column_stack((demand_volumes, demand)))
        for supply_price in supply:
            h_line = LineString([(min_vol_supply, supply_price), (max_vol_supply, supply_price)])
            supply_volumes.append(round(supply_line.intersection(h_line).x, 2))
        est_supply_line = LineString(np.column_stack((supply_volumes, supply)))
        demand_error = {}
        supply_error = {}
        for index, row in data.iterrows():
            price = row["Price"]
            hline_demand = LineString([(min_vol_demand, price), (max_vol_demand, price)])
            est_demand = est_demand_line.intersection(hline_demand).x
            true_demand = row["Demand"]
            ae_demand = abs(true_demand - est_demand)
            demand_error[price] = ae_demand
            mae_demand.append(ae_demand)
            hline_supply = LineString([(min_vol_supply, price), (max_vol_supply, price)])
            est_supply = est_supply_line.intersection(hline_supply).x
            true_supply = row["Supply"]
            ae_supply = abs(true_supply - est_supply)
            supply_error[price] = ae_supply
            mae_supply.append(ae_supply)
        demand_avg_error = demand_avg_error.append(demand_error, ignore_index=True)
        supply_avg_error = supply_avg_error.append(supply_error, ignore_index=True)
        maes_demand_full.append(sum(mae_demand) / len(mae_demand))
        maes_supply_full.append(sum(mae_supply) / len(mae_supply))
    df_demand_error = demand_avg_error.mean(axis=0).to_frame()
    df_demand_error = df_demand_error.sort_values(by=df_demand_error.columns[0])
    print(df_demand_error.tail(15))
    mae_demand_total = round(sum(maes_demand_full) / len(maes_demand_full), 2)
    df_supply_error = supply_avg_error.mean(axis=0).to_frame()
    df_supply_error = df_supply_error.sort_values(by=df_supply_error.columns[0])
    print(df_supply_error.tail(15))
    mae_supply_total = round(sum(maes_supply_full) / len(maes_supply_full), 2)
    print("MAE DEMAND: \t{}".format(mae_demand_total))
    print("MAE SUPPLY: \t{}".format(mae_supply_total))
    return mae_demand_total, mae_supply_total


def create_best_price_classes(plot, make_csv_files):
    save_path = "input/auction/time_series.csv"
    error_path = "input/auction/time_series_errors.csv"
    last_date = dt(2014, 6, 30).date()
    demand_cl, supply_cl, _, _ = get_best_price_classes(plot)
    if os.path.exists(save_path):
        price_classes_df = pd.read_csv(save_path)
        price_classes_df["Date"] = pd.to_datetime(price_classes_df["Date"], format="%Y-%m-%d")
        error_df = pd.read_csv(error_path)
        error_df["Date"] = pd.to_datetime(error_df["Date"], format="%Y-%m-%d")
        last_date = price_classes_df.tail()["Date"].tolist()[0].date()
        print("File exists. Continue from date {}\n".format(last_date + timedelta(days=1)))
    else:
        print("No existing file..\n")
        price_classes_df = pd.DataFrame(columns=["Date", "Hour"] + ["d {}".format(i) for i in demand_cl] +
                                                ["s {}".format(i) for i in supply_cl])
        error_df = pd.DataFrame(columns=["Date", "Hour", "MAE d", "MAE s"])
    if make_csv_files:
        all_files = sorted(Path("input/auction/csv_disc").iterdir())
        for day_file in all_files:
            date = dt.strptime(str(day_file).split("\\")[-1][:-4], "%Y-%m-%d").date()
            if date > last_date:
                print(date)
                df = pd.read_csv(day_file)
                number_of_hours = int((len(df.columns) - 1) / 2)
                for i in range(number_of_hours):
                    demand = df[["Price", df.columns[i * 2 + 1]]]
                    supply = df[["Price", df.columns[i * 2 + 2]]]
                    hour = demand.columns[1][9:]
                    time_df = {"Date": date, "Hour": hour}
                    if "_" not in hour:  # drop extra hour day light saving time
                        df_demand, df_demand_error = add_vol_rows(time_df.copy(), time_df.copy(), demand, demand_cl,
                                                                  "d")
                        df_supply, df_supply_error = add_vol_rows(time_df.copy(), time_df.copy(), supply, supply_cl,
                                                                  "s")
                        volume_hour_df = df_demand.join(df_supply.set_index(["Date", "Hour"]), on=["Date", "Hour"])
                        price_classes_df = price_classes_df.append(volume_hour_df, ignore_index=True)
                        error_hour_df = df_demand_error.join(df_supply_error.set_index(["Date", "Hour"]),
                                                             on=["Date", "Hour"])
                        error_df = error_df.append(error_hour_df, ignore_index=True)
                price_classes_df.to_csv(save_path, index=False)  # save time series every day
                error_df.to_csv(error_path, index=False)  # save error time series every day


def add_vol_rows(row_dict, error_dict, df, price_classes, curve):  # Helping method
    vol_min = df[df.columns[1]].min()
    vol_max = df[df.columns[1]].max()
    true_vol_line = LineString(np.column_stack((df[df.columns[1]], df["Price"])))
    est_volumes = []
    for price_class in price_classes:
        h_line = LineString([(vol_min, price_class), (vol_max, price_class)])
        est_volume = true_vol_line.intersection(h_line).x
        row_dict["{} {}".format(curve, price_class)] = est_volume
        est_volumes.append(est_volume)
    est_vol_line = LineString(np.column_stack((est_volumes, price_classes)))
    aes_volumes = []
    for index, row in df.iterrows():
        unit_price = row["Price"]
        true_vol = row[df.columns[1]]
        h_line = LineString([(vol_min, unit_price), (vol_max, unit_price)])
        est_volume = h_line.intersection(est_vol_line).x
        aes_volumes.append(abs(true_vol - est_volume))
    error_dict["MAE {}".format(curve)] = round(sum(aes_volumes) / len(aes_volumes), 2)
    volume_df = pd.DataFrame.from_dict(row_dict, orient="index").T
    volume_df["Date"] = pd.to_datetime(volume_df["Date"], format="%Y-%m-%d")
    error_df = pd.DataFrame.from_dict(error_dict, orient="index").T
    error_df["Date"] = pd.to_datetime(error_df["Date"], format="%Y-%m-%d")
    return volume_df, error_df


def get_best_price_classes(plot):  # Helping method
    mean_curves = pd.read_csv("output/auction/mean_curves.csv")
    d = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
    mae_demand = 26.73  # for all hours from 2014, 2020
    s = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105, 165, 210]
    mae_supply = 48.40  # for all hours from 2014, 2020
    if plot:
        plot_visual_price_class(d, "demand", mean_curves)
        plot_visual_price_class(s, "supply", mean_curves)
    return d, s, mae_demand, mae_supply


def plot_visual_price_class(classes, curve, df):  # Helping method
    orig_line = LineString(np.column_stack((df["Price"], df["Mean {}".format(curve)])))
    col_index = 0 if curve == "demand" else 1
    color = plt.get_cmap("tab10")(col_index)
    plt.subplots(figsize=full_fig)
    plt.plot(df["Mean {}".format(curve)], df["Price"], color=color, label="Mean {}".format(curve),
             linewidth=3)
    x_lim = plt.gca().get_xlim()
    y_lim = plt.gca().get_ylim()
    for i in range(len(classes)):
        c = classes[i]
        label = "{} price classes".format(len(classes)) if i == 0 else "_no_label_"
        h_line = LineString([(c, x_lim[0]), (c, x_lim[1])])
        x_max = h_line.intersection(orig_line).y
        plt.hlines(c, x_lim[0], x_max, color="grey", label=label)
    plt.xlim(x_lim)
    plt.ylim(y_lim[0], y_lim[1] * 1.03) if curve == "supply" else plt.ylim(y_lim)
    plt.title("Visual Price Classes for {}".format(curve.capitalize()), pad=title_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    for line in plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                           shadow=True).get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig("output/auction/price_classes/{}_best_classes_{}.png".format(len(classes), curve))
    plt.close()


def fix_summer_winter_time_bid_csv():
    all_times_df = get_data("01.07.2014", "02.06.2020", [], os.getcwd(), "h")
    all_bids_df = get_auction_data("01.07.2014", "02.06.2020", ["d", "s"], os.getcwd())
    all_bids_df = all_times_df.merge(all_bids_df, on=["Date", "Hour"], how="outer")
    all_bids_df = all_bids_df.interpolate(method='linear', axis=0)
    missing_df = all_bids_df[all_bids_df.isna().any(axis=1)]
    assert len(missing_df) == 0
    all_bids_df.to_csv("input/auction/time_series.csv", index=False, float_format='%.2f')


def test_supply_change_next_14_days():
    all_dates = [i for i in pd.date_range(dt(2014, 7, 1), dt(2020, 6, 3)-timedelta(days=14), freq='d')]
    number_of_auctions = len(all_dates)
    res_length = 300  # pick out n random dates
    random.seed(1)
    random_list = random.sample(range(number_of_auctions), res_length)
    dates_list = [all_dates[i].date() for i in random_list]
    demand, supply, _, _ = get_best_price_classes(plot=False)
    supply_classes = ["s {}".format(i) for i in supply]
    mape_df = pd.DataFrame(columns=supply_classes).T
    for i in range(1, 15):
        mape_df["MAPE {} day".format(i)] = np.PZERO
    for i in range(len(dates_list)):
        day = dates_list[i]
        print("{}\tChecking predictive power for {}".format(i, day))
        supply = get_auction_data(day, day+timedelta(days=14), "s", os.getcwd())
        for hour in range(24):
            current_supply = supply[hour:hour+1]
            for j in range(1, 15):
                index_day = j*24 + hour
                day_ahead_supply = supply[index_day: index_day + 1]
                for price in supply_classes:
                    ae = abs(current_supply.loc[hour, price] - day_ahead_supply.loc[index_day, price])
                    mape = 100 * ae / day_ahead_supply.loc[index_day, price]
                    prev_mean = mape_df.loc[price, "MAPE {} day".format(j)]
                    new_mean = prev_mean * (i / (i + 1)) + mape * (1 / (i + 1))
                    mape_df.loc[price, "MAPE {} day".format(j)] = new_mean
    print(mape_df)
    print("\n-------------------------------")
    print(mape_df.mean(axis=0))
    print("\n")
    print(mape_df.mean(axis=1))


def make_sensitivity_columns_demand():
    sens = [-8, -1, 1, 8]
    save_path = "input/auction/price_sensitivities.csv"
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        columns = ["Est price", "Est volume",  "Demand {}".format(sens[0]), "Demand {}".format(sens[1]), "Demand "
                   "{}".format(sens[2]), "Demand {}".format(sens[3])]
        df = pd.DataFrame(columns=["Date", "Hour"] + columns)
        df.to_csv("input/auction/price_sensitivities.csv", index=False)
    to_date = "02.06.2020"
    time_df = get_data("01.07.2014", to_date, [], os.getcwd(), "h")
    auctions = get_auction_data("01.07.2014", to_date, ["d", "s"], os.getcwd())
    demand_cl, supply_cl, _, _ = get_best_price_classes(plot=False)
    random.seed(1)
    plot_list = random.sample(range(len(auctions)), 30)
    for i in range(len(time_df)):
        row = {"Date": time_df.loc[i, "Date"].date(), "Hour": time_df.loc[i, "Hour"]}
        demand_volumes = [auctions.loc[i, j] for j in auctions.columns if "d" in j]
        demand_line = LineString(np.column_stack((demand_volumes, demand_cl)))
        supply_volumes = [auctions.loc[i, j] for j in auctions.columns if "s" in j]
        min_supply = min(supply_volumes)
        max_supply = max(supply_volumes)
        supply_line = LineString(np.column_stack((supply_volumes, supply_cl)))
        price = demand_line.intersection(supply_line).y
        volume = demand_line.intersection(supply_line).x
        df.loc[i, "Est price"] = round(price, 2)
        df.loc[i, "Est volume"] = round(volume, 2)
        for s in sens:
            vol = get_volume_sensitivity(price, volume, min_supply, max_supply, supply_line, s)
            row["Demand {}".format(s)] = vol
        if i % 24 == 0:
            print(row["Date"])
        if i in plot_list or price > 197 or price < 0:
            plot_sensitivity(demand_volumes, supply_volumes, demand_cl, supply_cl,volume, row, sens)
        df = df.append(row, ignore_index=True)
        save_list = [row["Date"], row["Hour"]] + [row["Demand {}".format(i)] for i in sens]
        append_list_as_row(save_path, save_list)


def append_list_as_row(file_name, list_of_elem):
    from csv import writer
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def plot_sensitivity(demand_vol, supply_vol, dem_classes, sup_classes, est_vol, row, sens):
    plt.subplots(figsize=full_fig)
    plt.title("Sensitivity Plot for {}, h {}".format(row["Date"], row["Hour"]), pad=title_pad)
    plt.xlabel("Volume [MWh]", labelpad=label_pad)
    plt.ylabel("Price [€]", labelpad=label_pad)
    col_1 = plt.get_cmap("tab10")(0)
    col_2 = plt.get_cmap("tab10")(1)
    plt.plot(demand_vol, dem_classes, color=col_1, label="Demand", linewidth=3)
    plt.plot(supply_vol, sup_classes, color=col_2, label="Supply", linewidth=3)
    y_lim = plt.gca().get_ylim()
    for i in range(len(sens)):
        if i == 0:
            lab = "Sensitivity lines ({})".format(", ".join([str(s) for s in sens]))
        else:
            lab = "_nolabel_"
        s = sens[i]
        demand = est_vol + row["Demand {}".format(s)]
        plt.vlines(demand, y_lim[0], 210, color="black", label=lab, linestyles="dotted", linewidth=2)

    for line in plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.gca().set_ylim(y_lim)
    plt.tight_layout()
    print("-- saved plot hour {} --".format(row["Hour"]))
    plt.savefig("output/auction/price_sensitivities/{}_{}.png".format(row["Date"], row["Hour"]))
    plt.close()


def get_volume_sensitivity(price, volume, min_s, max_s, supply_line, s):
    if price + s < -10:
        h_line = LineString([(min_s, -10), (max_s, -10)])
    elif price + s > 210:
        h_line = LineString([(min_s, 210), (max_s, 210)])
    else:
        h_line = LineString([(min_s, price+s), (max_s, price+s)])
    sens_vol = h_line.intersection(supply_line).x
    return round(sens_vol - volume, 2)


if __name__ == '__main__':
    print("Running bid curve script..\n")
    # auction_data()
    # min_max_price()
    # rename_folders_from_raw()
    # auction_data()
    # merge_same_dates_in_one_csv()
    # plot_mean_curves()
    # eda_disc_auction_data(overview=True, make_analysis_csv=False)
    # investigate_unit_price_information_loss()
    # make_price_classes_from_mean_curves(16, 18)
    # create_auction_subset()
    # evaluate_mean_price_classes(create_csv=False)
    # create_best_price_classes(plot=False, make_csv_files=False)
    # fix_summer_winter_time_bid_csv()
    # test_supply_change_next_14_days()
    make_sensitivity_columns_demand()

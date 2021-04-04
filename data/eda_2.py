import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

full_fig = (13, 7)
label_pad = 12
title_pad = 20


def plot_mean_curves():
    folder = "input/auction/csv_disc"
    title_date = "(01.07.2014 - 02.06.2020)"
    price_range = range(-10, 211)
    df_demand = pd.DataFrame(columns=price_range).transpose()
    df_supply = pd.DataFrame(columns=price_range).transpose()
    no = 0
    for file in sorted(Path(folder).iterdir()):  # list all csv files
        date_string = str(file).split("\\")[3].split("_")[-1][:-4]
        print(date_string)
        data = pd.read_csv(file)
        for i in range(int((len(data.columns) - 1) / 2)):
            demand = data[["Price"]+[data.columns[1+i*2]]].set_index("Price").rename(columns={data.columns[1+i*2]: "Demand" + str(no)})
            supply = data[["Price"] + [data.columns[1+i*2+1]]].set_index("Price").rename(columns={data.columns[1+i*2+1]: "Supply " + str(no)})
            df_demand = df_demand.merge(demand, left_index=True, right_index=True)
            df_supply = df_supply.merge(supply, left_index=True, right_index=True)
            no += 1
    print("\nCalculating mean demand...")
    df_demand["Mean demand"] = df_demand.mean(axis=1)
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
    print("\nCalculating mean supply...")
    df_supply["Mean supply"] = df_supply.mean(axis=1)
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


if __name__ == '__main__':
    plot_mean_curves()
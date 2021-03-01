import matplotlib.pyplot as plt
from datetime import datetime as dt
from src.preprocessing.arcsinh import arcsinh
import os
from data.data_handler import get_data


def plot_norm_weekday():
    start_date = dt(2019, 1, 1)
    end_date = dt(2019, 1, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday"], os.getcwd())
    training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    grouped_df = training_data.groupby(by="Weekday").mean()
    normalized_df = (grouped_df-grouped_df.mean())/grouped_df.std()
    label_pad = 12
    title_pad = 20
    plt.subplots(figsize=(6.5, 7))
    true_color = "steelblue"
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.bar(days, normalized_df["Trans System Price"], color=true_color)
    plt.title("Mean Normalized Price Per Weekday 2019", pad=title_pad)
    plt.ylabel("Norm. Price", labelpad=label_pad)
    plt.xlabel("Day of week", labelpad=label_pad)
    ymin = min(normalized_df["Trans System Price"])*1.1
    ymax = max(normalized_df["Trans System Price"])*1.1
    plt.ylim(ymin, ymax)
    for i, v in enumerate(normalized_df["Trans System Price"].tolist()):
        sys = round(grouped_df["System Price"].tolist()[i], 1)
        if v < 0:
            pad = -0.1
        else:
            pad = 0.05
        plt.text(i, v+pad, sys, color="steelblue", fontweight='bold', ha='center')
    plt.tight_layout()
    path = "output/plots/eda/price_per_week_day_2019.png"
    plt.savefig(path)


if __name__ == '__main__':
    plot_norm_weekday()
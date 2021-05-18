# script for generating correlation matrix on data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_handler import get_data
import os


def plot_correlation_matrix_first():
    replace_dict = {"System Price": "Price", "Total Vol": "Volume", "Total Hydro": "Hydro", "Total Hydro Dev":
                    "Hydro Dev", "Temp Norway": "Temp Norway", "Prec Norway 7": "Prec Norway 7",
                    "Prec Norway": "Prec Norway"}
    columns = ["System Price", "Total Vol", "Supply", "Demand", "Total Hydro", "Total Hydro Dev", "Temp Norway",
               "Wind Prod", "Prec Norway", "Prec Norway 7", "Snow Norway", "Oil", "Gas", "Coal", "Low Carbon", "EEX", "APX", "OMEL"]
    df = get_data("01.07.2014", "02.06.2020", columns, os.getcwd(), "d")
    for i in range(2010, len(df)):
        df.loc[i, "Snow Norway"] = np.NAN
    df = df[columns]
    col = "System Price"
    #df["1 day"] = df[col].shift(1)
    #df["2 day"] = df[col].shift(2)
    #df["3 day"] = df[col].shift(3)
    #df["1 week"] = df[col].shift(7)
    #df["2 week"] = df[col].shift(14)
    df = df.rename(columns=replace_dict)
    df = df.dropna()
    ticks = [col for col in df.columns]
    f, ax = plt.subplots(figsize=(13, 10.5))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=True, fmt=".2f", annot_kws={'size': 11})
    plt.xticks(np.arange(len(df.columns)) + 0.5, ticks, rotation=45, size=11)
    plt.yticks(np.arange(len(df.columns)) + 0.5, ticks, rotation='horizontal', size=11)
    plt.title("Correlation Matrix across all Daily Data Features", pad=20, size=15)
    plt.tight_layout()
    plt.savefig("output\\plots\\eda\\corr_matrix.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_correlation_matrix_first()

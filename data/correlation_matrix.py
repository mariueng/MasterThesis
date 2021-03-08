# script for generating correlation matrix on data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_handler import get_data
import os


def plot_correlation_matrix_first(path, size, columns):
    replace_dict = {"System Price": "Price", "Total Vol": "Volume", "Total Hydro": "Hydro", "Sine Week": "Week",
                    "Sine Month": "Month", "Sine Season": "Season","Total Hydro Dev": "Hydro Dev",
                    "T Nor": "Temperature"}
    df = get_data("01.01.2014", "31.12.2019", columns, os.getcwd(), "d")
    df = df[columns]
    #ticks = [replace_dict[col] for col in df.columns]
    ticks = [col for col in df.columns]
    f, ax = plt.subplots(figsize=(size, size-2))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=True, fmt=".2f",annot_kws={'size':12})
    plt.xticks(np.arange(len(df.columns))+0.5, ticks, rotation=45, size=12)
    plt.yticks(np.arange(len(df.columns))+0.5, ticks, rotation='horizontal', size=12)
    plt.title("Correlation Matrix v.1", size=16, pad=10)
    plt.tight_layout()
    plt.savefig("output\\plots\\eda\\corr_matrix_v1.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    path_ = "..\\data\\input\\combined\\all_data_hourly.csv"
    columns_ = ["System Price", "Total Vol", "Total Hydro Dev", "T Nor", "T Namsos", "T Troms", "T Hamar", "T Krsand", "T Bergen"]
    plot_correlation_matrix_first(path_, 10, columns_)

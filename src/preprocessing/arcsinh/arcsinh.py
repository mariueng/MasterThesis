import numpy as np
from data.data_handler import get_data
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import median_abs_deviation


# area (or inverse) hyperbolic sine transformation (Ziel, Weron 2018)
def to_arcsinh(df, col):
    a = df[col].mean()
    b = median_abs_deviation(df[col], scale=1/1.4826)
    df["Trans " + col] = (1/b) * (df[col] - a)
    df["Trans " + col] = np.arcsinh(df["Trans " + col])
    return df, a, b


def to_arcsinh_from_a_and_b(value, a, b):
    value = (1/b) * (value - a)
    value = np.arcsinh(value)
    return value


def from_arcsin_to_original(array_like, a, b):
    array_like = [b*np.sinh(i) + a for i in array_like]
    return array_like


def validate_method():
    label_pad = 12
    title_pad = 20
    img_size = (6.5, 7)
    bins = 50
    true_color = "steelblue"
    trans_color = "firebrick"
    data = get_data("01.01.2019", "31.12.2019", ["System Price"], os.getcwd())
    data_orig = data["System Price"].tolist()
    plt.hist(data_orig, bins=50, density=100, color=true_color)
    titles = ["System price [€]", "Density", "Electricity Price Distribution 2019"]
    plot_distribution(img_size, label_pad, title_pad, true_color, data_orig, bins, titles)
    trans_data, a, b = to_arcsinh(data, "System Price")
    trans_data = trans_data["Trans System Price"].tolist()
    titles = ["Transformed price", "Density", "Ashinh Transformed Price 2019"]
    plot_distribution(img_size, label_pad, title_pad, trans_color, trans_data, bins, titles)


def plot_distribution(img_size, label_pad, title_pad, color, data, bins, titles):
    plt.subplots(figsize=img_size)
    plt.hist(data, bins=bins, density=100, color=color)
    plt.xlabel(titles[0], labelpad=label_pad)
    plt.ylabel(titles[1], labelpad=label_pad)
    plt.title(titles[2], pad=title_pad)
    plt.tight_layout()
    plt.savefig(titles[2].replace(" ", "_") + ".png")
    plt.close()


if __name__ == '__main__':
    # validate_method()
    a_ = from_arcsin_to_original([1.405482], 31.55199967556544, 9.147642000000003)
    print(a_)
    b_ = to_arcsinh_from_a_and_b(49.08, 31.55199967556544, 9.147642000000003)
    print(b_)

import os
from data.data_handler import get_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import preprocessing



def get_possible_features():  # Helping method
    features = [
        "Total Vol",
        "Total Hydro",
        "Total Hydro Dev",
        "Demand",
        "Supply",
        "Wind Prod",
        "Prec Norway 7",
        "Temp Norway",
        "Oil",
        "Gas",
        "Coal",
        "Low Carbon"
    ]
    return features


def forward_selection():
    test_features = get_possible_features()
    data = get_data("01.01.2014", "31.12.2019", ["System Price"] + test_features, os.getcwd(), "d")
    y = data["System Price"].to_numpy()
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(y)
    x = data.drop(columns=["Date", "System Price"]).to_numpy()
    rfc = RandomForestRegressor()
    sfs = SequentialFeatureSelector(rfc, n_features_to_select=3)
    sfs.fit(x, y)
    print(sfs.transform(x).shape)
    results_list = sfs.get_support()
    for i in range(len(test_features)):
        print("{}:\t {}".format(test_features[i], results_list.item(i)))


if __name__ == '__main__':
    forward_selection()
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL):
    if not os.path.isdir("data/"):
         os.makedirs("data/")
    tgz_path = "data/housing.tgz"
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path="data/")
    housing_tgz.close()

def load_housing_data():
    df = pd.DataFrame.from_csv("data/housing.csv")
    df["income_cat"] = np.ceil(df["median_income"] / 1.5)
    df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
    return df

def hist_plot():
    df.hist(bins=50, figsize=(20,15))
    plt.savefig("plots/house_hist.png")

def split():
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    return train_set, test_set

def strat_split():
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set


if __name__ == '__main__':
    #fetch_housing_data()
    df = load_housing_data()
    hist_plot()
    train_set, test_set = split()
    strat_train_set, strat_test_set = strat_split()

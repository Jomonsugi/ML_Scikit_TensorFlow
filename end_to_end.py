import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer, LabelEncoder
from attributesadder import CombinedAttributesAdder, DataFrameSelector, housing_lin_reg
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression

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
    housing = pd.DataFrame.from_csv("data/housing.csv", index_col=None)
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    return housing

def hist_plot():
    housing.hist(bins=50, figsize=(20,15))
    plt.savefig("plots/house_hist.png")

def random_split():
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    return train_set, test_set

def strat_split():
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set

def lat_long_plot():
    #adding the alpha parameter identifies places of high density
    strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # and now giving it more distinction
    strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population",
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.savefig("plots/lat_long.png")

def corr_matrix():
    corr_matrix = strat_train_set.corr()
    mhv = corr_matrix["median_house_value"].sort_values(ascending=False)
    print(mhv)
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    sm = scatter_matrix(housing[attributes], figsize=(10,6))
    plt.savefig("plots/scatter_matrix.png")
    mi_sm = strat_train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    #notice the lines that emerge at 500K, 450K, etc.
    #it might be a ghood idea to remove districts creating these lines
    #so the algorithm doesn't learn to produce these quirks
    plt.savefig("plots/scatter_mhv.png")

def clean(data):
    '''
    handle pandas df with DataFrameSelector class
    imputing all missing numerical values with the median
    adding rooms per household, population per household, and bedrooms per rooms
    standardizing data
    '''
    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
    '''
    LabelBinarizer transforms text categories to integar categories and then from int categories to one-hot vectors
    '''
    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label_binarizer', LabelBinarizer()),
        ])

    #has to be a better way to get the encoded columns...
    # encoder = LabelEncoder()
    # housing_cat = housing["ocean_proximity"]
    # housing_cat_enc = encoder.fit_transform(housing_cat)
    #
    # # print(encoder.classes_)

    #piplines handling numerical and categorical combined
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    model_train_data = full_pipeline.fit_transform(housing)
    return full_pipeline, model_train_data

def lin_reg(housing_prep):
    lin_reg = LinearRegression()
    model = lin_reg.fit(housing_prep, housing_labels)
    return model

def lin_pred(data):
    prepped_data = full_pipeline.transform(data)
    predictions = model.predict(prepped_data)
    return predictions


if __name__ == '__main__':
    plt.close('all')
    #fetch_housing_data()
    housing = load_housing_data()
    # hist_plot()
    train_set, test_set = random_split()
    strat_train_set, strat_test_set = strat_split()
    #plots using strat train set
    # lat_long_plot()
    # corr_matrix()
    #for pipeline
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    #prepare training data with pipeline, exact pipeline should be used for
    #test data later

    full_pipeline, model_train_data = clean(housing)
    model = lin_reg(model_train_data)

    some_data = housing.iloc[:5]
    predictions = lin_pred(some_data)

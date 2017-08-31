from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer, LabelEncoder
from attributesadder import CombinedAttributesAdder, DataFrameSelector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression

class transform_data:
    def __init__(self, housing):
        self.housing = housing

    def strat_split(self, housing):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        return strat_train_set, strat_test_set

    def clean(self):
        strat_train_set, strat_test_set = self.strat_split(self.housing)
        housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
        housing_labels = strat_train_set["median_house_value"].copy()
        housing_num = housing.drop("ocean_proximity", axis=1)
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
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

        #piplines handling numerical and categorical combined
        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])

        model_train_data = full_pipeline.fit_transform(housing)
        return full_pipeline, model_train_data, housing_labels, strat_test_set

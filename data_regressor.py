import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

######### edit as needed
FILENAME = "filename.txt"
NUMBER_OF_FEATURES = 6
DATASET_DICT = {
    "demo": [5, 17, 282, 283, 284, 285],
    "demo_pacc_np1": [9, 13, 18, 33, 34, 37, 38, 39, 40],
    "demo_pacc_np2": [44, 45, 46, 47, 48, 49, 50, 52],
    "demo_pacc_np3": list(range(54, 61)),
    "demo_pacc_np4": list(range(61, 67)),
    "demo_pacc_np_HV_adj": [202],
    "demo_pacc_np_HV_adj_func1": list(range(82, 88)),
    "demo_pacc_np_HV_adj_func2": list(range(88, 94)),
}
NO_PACC_DATASET_DICT = {
    "demo": [5, 17, 282, 283, 284, 285],
    "demo_!pacc_np1": [13, 18, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    "demo_!pacc_np2": list(range(43, 54)),
    "demo_pacc_np3": list(range(54, 61)),
    "demo_pacc_np4": list(range(61, 67)),
    "demo_pacc_np_HV_adj": [202],
    "demo_pacc_np_HV_adj_func1": list(range(82, 88)),
    "demo_pacc_np_HV_adj_func2": list(range(88, 94)),
}
#########


def prepare_df(df):
    """
    Prepares dataframe for models by removing outliers,
    replacing uncommon ethnicity codes, and label encoding columns
    paramters: df (dataframe)
    returns: prepared dataframe
    """
    df = df[df["toff"] > -10]
    df.dropna(inplace=True)
    df.drop_duplicates()
    df_final = df.replace({"B/NA": "NatA/AN", "NA/W": "NatA/AN"})
    encode = LabelEncoder()
    df_final["SEX_encoded"] = encode.fit_transform(df_final["SEX"])
    df_final["E4_Status_encoded"] = encode.fit_transform(df_final["E4_Status"])
    df_final["Race_encoded"] = encode.fit_transform(df_final["Race"])
    df_final["Handedness_encoded"] = encode.fit_transform(df_final["Handedness"])
    return df_final


def random_forest_regressor(features, outcome, feature_list):
    """
    Runs random forest regressor model, plots accuracy
    paramters: features (dataframe of all features with data),
    outcome (dataframe of outcome with data),
    feature_list (list of features)
    returns: mape (float of accuracy), feature_imp_series(series of featues & their importances)
    """
    features_train, features_test, outcome_train, outcome_test = train_test_split(
        features, outcome, test_size=0.3, random_state=1
    )
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(features_train, outcome_train)
    outcome_pred = regressor.predict(features_test)
    mape = mean_absolute_percentage_error(outcome_test, outcome_pred)
    feature_imp_series = pd.Series(
        regressor.feature_importances_, index=feature_list
    ).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    plt.scatter(outcome_pred, outcome_test)
    plt.title("Random Forest Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return (mape, feature_imp_series)


def gradient_boosting_model(features, outcome, feature_list):
    """
    Runs gradient boosting model, plots accuracy
    paramters: features (dataframe of all features with data),
    outcome (dataframe of outcome with data),
    feature_list (list of features)
    returns: mape (float of accuracy), feature_imp_series(series of featues & their importances)
    """
    features_train, features_test, outcome_train, outcome_test = train_test_split(
        features, outcome, test_size=0.3, random_state=1
    )
    regressor = GradientBoostingRegressor(n_estimators=100)
    regressor.fit(features_train, outcome_train)
    outcome_pred = regressor.predict(features_test)
    mape = mean_absolute_percentage_error(outcome_test, outcome_pred)
    feature_imp_series = pd.Series(
        regressor.feature_importances_, index=feature_list
    ).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    plt.scatter(outcome_pred, outcome_test)
    plt.title("Gradient Boosting Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return (mape, feature_imp_series)


def prepare_plot(feature_imp_series):
    """
    prepares values to pass into matplotlib plot function, so that other is only used as a slice
    if there is more features than NUMBER_OF_FEATURES
    paramters: feature_imp_series(series of featues & their importances)
    returns: values (list of values), labels (list of labels)
    """
    if len(feature_imp_series) > NUMBER_OF_FEATURES:
        other = sum(feature_imp_series[NUMBER_OF_FEATURES:])
        values = np.append(feature_imp_series[:NUMBER_OF_FEATURES], [other])
        labels = np.append(feature_imp_series.index[:NUMBER_OF_FEATURES], ["other"])
    else:
        values = feature_imp_series[:NUMBER_OF_FEATURES]
        labels = feature_imp_series.index[:NUMBER_OF_FEATURES]
    return (values, labels, feature_imp_series[NUMBER_OF_FEATURES:].keys())


def main():
    """
    prepares dataframe, runs machine learning models on each dataset, plots pie chart
    of feature importances and line chart of MAPES
    paramters: none
    returns: none
    """
    df = pd.read_csv(FILENAME, sep="\t")
    df_prepared = prepare_df(df)
    col_array = np.array(list(df_prepared.columns.values))
    data_set_list = list(DATASET_DICT.values())
    feature_list = []
    dataset_num = 0
    stat_dict = {"dataset_num": [], "Features": [], "RFR MAPE": [], "GBM MAPE": []}
    for lst in data_set_list:
        feature_list += list(col_array[lst])
        stat_dict["dataset_num"].append(dataset_num)
        stat_dict["Features"].append(feature_list.copy())
        features = df_prepared[feature_list]
        outcome = df_prepared["toff"]
        rfr = random_forest_regressor(features, outcome, feature_list)
        gbm = gradient_boosting_model(features, outcome, feature_list)
        stat_dict["RFR MAPE"].append(rfr[0])
        stat_dict["GBM MAPE"].append(gbm[0])
        (gbm_values, gbm_labels, gbm_other) = prepare_plot(gbm[1])
        (rfr_values, rfr_labels, rfr_other) = prepare_plot(rfr[1])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
        ax1.pie(
            gbm_values,
            labels=gbm_labels,
            autopct="%1.1f%%",
            textprops={"fontsize": 8},
        )
        ax2.pie(
            rfr_values,
            labels=rfr_labels,
            autopct="%1.1f%%",
            textprops={"fontsize": 8},
        )
        ax1.set_title("Gradient Boosting Model", fontsize=12)
        ax2.set_title("Random Forests Model", fontsize=12)
        fig.suptitle(
            "Feature Importances for Dataset " + list(DATASET_DICT.keys())[dataset_num],
            fontsize=16,
        )
        txt = fig.text(
            0.01,
            0.02,
            "GBM other: "
            + str(list(gbm_other))
            + "\n"
            + "RFR other: "
            + str(list(rfr_other)),
            fontsize=7,
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 2500
        txt2 = fig.text(
            0.05,
            0.9,
            "RFR MAPE: "
            + str(stat_dict["RFR MAPE"][dataset_num])
            + "\n"
            + "GBM MAPE:"
            + str(stat_dict["GBM MAPE"][dataset_num]),
            fontsize=8,
        )
        txt2.set_bbox(dict(facecolor="white", alpha=0.5))
        plt.show()
        dataset_num += 1
    plt.figure(figsize=(9, 6))
    plt.plot(stat_dict["dataset_num"], stat_dict["RFR MAPE"], "-o", label="RFR")
    plt.plot(stat_dict["dataset_num"], stat_dict["GBM MAPE"], "-o", label="GBM")
    plt.legend()
    plt.xlabel("DataSet Group")
    plt.ylabel("MAPE")
    plt.title("DataSet Group vs MAPE", fontsize=16)
    plt.show()


main()

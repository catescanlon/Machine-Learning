import numpy as np
import pandas as pd
import sklearn as sci
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import calc_importance_matrix as cim

######### edit as needed
FILENAME = "filename.txt"
# toggle false to make it three groups of classifications instead of 2
TWO_CUTOFF = True
# toggle flase to not do PCA
PCA_BOOL = True
NUMBER_OF_FEATURES = 6
QUARTER_PERCENTILE = -14.49415
THIRD_PERCENTILE = -13.512
HALF_PERCENTILE = -10.8636
TWO_THIRD_PERCENTILE = -9.15013
THREE_QUARTER_PERCENTILE = -8.2789975
DATASET_DICT = {
    "demo": [5, 17, 335, 336, 337, 338],
    "demo_pacc_np": [9],
    "demo_pacc_np_HV_adj": [202],
    "demo_pacc_np_HV_adj_func1": list(range(82, 88)),
    "demo_pacc_np_HV_adj_func2": list(range(88, 94)),
    "demo_pacc_np_HV_adj_func2_tau": [324, 325],
    "demo_pacc_np_HV_adj_func2_tau_pib": [333],
}
DATASET_DICT_NP = {
    "demo": [5, 17, 335, 336, 337, 338],
    "demo_pacc_np1": [9, 13, 18, 33, 34, 37, 38, 39, 40],
    "demo_pacc_np2": [44, 45, 46, 47, 48, 49, 50, 52],
    "demo_pacc_np3": list(range(54, 61)),
    "demo_pacc_np4": list(range(61, 67)),
    "demo_pacc_np_HV_adj": [202],
    "demo_pacc_np_HV_adj_func1": list(range(82, 88)),
    "demo_pacc_np_HV_adj_func2": list(range(88, 94)),
    "demo_pacc_np_HV_adj_func2_tau": [324, 325],
    "demo_pacc_np_HV_adj_func2_tau_pib": [333],
}
#########


def prepare_df(df):
    """
    Prepares dataframe for models by removing outliers,
    replacing uncommon ethnicity codes, and label encoding columns
    Creates 2 or 3 groups of classification based on bool at top of code
    paramters: df (dataframe)
    returns: prepared dataframe
    """
    df.dropna(inplace=True)
    df.drop_duplicates()
    df_final = df.replace({"B/NA": "NatA/AN", "NA/W": "NatA/AN"})
    time_encoded = []
    zero_count = 0
    total_count = 0
    chance = 0
    if TWO_CUTOFF:
        for value in df["time_new"]:
            total_count += 1
            if value < THREE_QUARTER_PERCENTILE:
                time_encoded.append(0)
                zero_count += 1
            if value >= THREE_QUARTER_PERCENTILE:
                time_encoded.append(1)
            chance = zero_count / total_count
            if chance < 0.5:
                chance = 1 - chance
    if not TWO_CUTOFF:
        for value in df["time_new"]:
            if value < THIRD_PERCENTILE:
                time_encoded.append(0)
            if value >= THIRD_PERCENTILE and value < TWO_THIRD_PERCENTILE:
                time_encoded.append(1)
            if value >= TWO_THIRD_PERCENTILE:
                time_encoded.append(2)
    df_final["time_encoded"] = time_encoded
    encode = LabelEncoder()
    df_final["SEX_encoded"] = encode.fit_transform(df_final["SEX"])
    df_final["E4_Status_encoded"] = encode.fit_transform(df_final["E4_Status"])
    df_final["Race_encoded"] = encode.fit_transform(df_final["Race"])
    df_final["Handedness_encoded"] = encode.fit_transform(df_final["Handedness"])
    return (df_final, chance)


def random_forest_classifier(features, outcome, feature_list, chance):
    """
    Runs random forest classifier model & plots an importance matrix of features separated by class
    Runs PCA if bool at top is true
    paramters: features (dataframe of all features with data),
    outcome (dataframe of outcome with data),
    feature_list (list of features)
    returns: accuracy (float), feature_imp_series(series of featues & their importances)
    """
    feature_imp_series = None
    features_train, features_test, outcome_train, outcome_test = train_test_split(
        features, outcome, test_size=0.3, random_state=1
    )
    if PCA_BOOL:
        scaler = StandardScaler()
        scaler.fit(features_train)
        features_train = scaler.transform(features_train)
        features_test = scaler.transform(features_test)

        pca = PCA(0.95)
        pca.fit(features_train)
        features_train = pca.transform(features_train)
        features_test = pca.transform(features_test)

    classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=50,
        max_features=None,
        min_samples_split=10,
        min_weight_fraction_leaf=0.1,
    )
    classifier.fit(features_train, outcome_train)
    outcome_pred = classifier.predict(features_test)
    accuracy = sci.metrics.accuracy_score(outcome_test, outcome_pred)
    max_depth = list()
    for tree in classifier.estimators_:
        max_depth.append(tree.tree_.max_depth)

    if not PCA_BOOL:
        feature_imp_series = pd.Series(
            classifier.feature_importances_, index=feature_list
        ).sort_values(ascending=False)

    if PCA_BOOL:
        comp_exp_var = pca.explained_variance_ratio_
        comp_num = pca.n_components_
        columm_list = []
        for num in range(0, comp_num):
            columm_list += [num]
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=columm_list,
            index=feature_list,
        )
        print(loadings)
        print("Variance per component: ", comp_exp_var)
        print("Total variance: ", sum(comp_exp_var))
        print("Number of components: ", comp_num)
        print("Chance: ", chance)
        print("Accuracy: ", accuracy)
    print("avg max depth %0.1f" % (sum(max_depth) / len(max_depth)))

    if not PCA_BOOL:
        # calc_importance_matrix function from outside code
        imp_mat = cim.calc_importance_matrix(classifier)
        marker_lst = ["o", "v", "*", "D", "2", "s"]
        color_lst = sns.color_palette(n_colors=10)
        color_marker_lst = np.array(
            [(c, m) for m in marker_lst for c in color_lst], dtype="object"
        )
        fig, ax = plt.subplots(figsize=(13, 7))
        for j in np.arange(len(feature_list)):
            ax.scatter(
                np.arange(len([0, 1]))
                + np.random.uniform(-1, 1, size=len([0, 1])) * 0.25 / 2.0,
                imp_mat[:, j],
                color=color_marker_lst[j][0],
                marker=color_marker_lst[j][1],
                s=10,
                label=feature_list[j],
            )
            ax.hlines(
                y=0.075,
                xmin=-0.25 / 2.0,
                xmax=len([0, 1]) - 1 + 0.25 / 2.0,
                colors="k",
                linestyles="dashed",
                lw=1,
            )
        ax.set_xticks(np.arange(len([0, 1])))
        ax.set_xticklabels([0, 1], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Class")
        ax.set_ylabel("Importance")
        ax.set_ylim(bottom=0)
        plt.legend(loc="center left", title="Features", ncol=2, bbox_to_anchor=(1, 0))
        plt.tight_layout()
        plt.show()
    return (accuracy, feature_imp_series)


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


def grid_search(features_train, features_test, outcome_train, outcome_test):
    """
    Performs grid search on split testing data using a dictionary of options
    Prints: the best parameters and a classification report
    """
    gs_dictionary = {
        "n_estimators": [100, 10, 500],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 50, 100],
        "min_samples_split": [2, 5, 10, 50],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.4],
        "max_features": [None, "sqrt", "log2"],
        "n_jobs": [None, -1],
        "warm_start": [True, False],
        "max_samples": [None, 0.33, 0.5, 0.66, 0.75, 0.9],
    }
    grid = GridSearchCV(RandomForestClassifier(), gs_dictionary)
    grid.fit(features_train, outcome_train)
    print(grid.best_params_)
    grid_predictions = grid.predict(features_test)
    print(classification_report(outcome_test, grid_predictions))


def main():
    """
    prepares dataframe, runs machine learning models on each dataset, plots pie chart
    of feature importances and line chart of MAPES
    paramters: none
    returns: none
    """
    df = pd.read_csv(FILENAME, sep="\t")
    df_prepared, chance = prepare_df(df)
    col_array = np.array(list(df_prepared.columns.values))
    data_set_list = list(DATASET_DICT.values())
    feature_list = []
    dataset_num = 0
    stat_dict = {"dataset_num": [], "Features": [], "Accuracy": [], "Chance": []}
    for lst in data_set_list:
        feature_list += list(col_array[lst])
        stat_dict["dataset_num"].append(dataset_num)
        stat_dict["Chance"].append(chance)
        stat_dict["Features"].append(feature_list.copy())
        features = df_prepared[feature_list]
        outcome = df_prepared["time_encoded"]
        rfc = random_forest_classifier(features, outcome, feature_list, chance)
        stat_dict["Accuracy"].append(rfc[0])
        if not PCA_BOOL:
            (rfc_values, rfc_labels, rfc_other) = prepare_plot(rfc[1])
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.pie(
                rfc_values,
                labels=rfc_labels,
                autopct="%1.1f%%",
                textprops={"fontsize": 8},
            )
            ax.set_title("Random Forests Model", fontsize=12)
            fig.suptitle(
                "Feature Importances for Dataset "
                + list(DATASET_DICT.keys())[dataset_num],
                fontsize=16,
            )
            txt = fig.text(
                0.01,
                0.02,
                "Other: " + str(list(rfc_other)),
                fontsize=7,
                wrap=True,
            )
            txt._get_wrap_line_width = lambda: 2500
            txt2 = fig.text(
                0.05,
                0.9,
                "Accuracy: "
                + str(stat_dict["Accuracy"][dataset_num])
                + "\n"
                + "Chance: "
                + str(chance),
            )
            txt2.set_bbox(dict(facecolor="white", alpha=0.5))
            plt.show()
        dataset_num += 1
    plt.figure(figsize=(9, 6))
    plt.plot(stat_dict["dataset_num"], stat_dict["Accuracy"], "-o", label="Accuracy")
    plt.plot(stat_dict["dataset_num"], stat_dict["Chance"], "-o", label="Chance")
    plt.legend()
    plt.xlabel("DataSet Group")
    plt.ylabel("Accuracy of Random Forest Classifier")
    plt.title("DataSet Group vs Accuracy", fontsize=16)
    plt.show()


main()

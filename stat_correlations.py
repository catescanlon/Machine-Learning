import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import HLR
from scipy import stats
import pickle

FILENAME = "Data_macroSleep_PACC_CDR.txt"
CONNECTIVITY_NETWORKS = [
    "MRI_fcMRI_DMN_Orth_On_675_20",
    "MRI_fcMRI_Motor_Orth_On_675_20",
    "MRI_fcMRI_PriVis_Orth_On_675_20",
    "DAN",
    "MRI_fcMRI_r_CRTL_Orth_On_675_20",
    "MRI_fcMRI_l_CRTL_Orth_On_675_20",
    "MRI_fcMRI_ExVis_Orth_On_675_20",
]
SLEEP_MACRO = ["PCT_N1", "Percent_N2", "PCT_N3", "PCT_R"]
PACC = ["NP_PACC_PACC5"]
DEMOGRAPHICS = [
    "MRI_Age",
    "Sex_binary",
    "APOE_binary",
    "MRI_fcMRI_MeanMovement_QA",
]
SLEEP_MICRO_SLOW = [
    "DENS_11_FRONTAL",
    "FRQ_11_FRONTAL",
    "CHIRP_11_FRONTAL",
    "ISA_S_11_FRONTAL",
]
SLEEP_MICRO_FAST = [
    "DENS_15_CENTRAL",
    "FRQ_15_CENTRAL",
    "CHIRP_15_CENTRAL",
    "ISA_S_15_CENTRAL",
]
DATA = CONNECTIVITY_NETWORKS + SLEEP_MICRO_SLOW + SLEEP_MICRO_FAST


def investigate(df):
    """
    Saves histograms of each of the variables & a saves csv of descriptives table
    Parameters: df (dataframe)
    Returns: None
    """
    conn_array = df[CONNECTIVITY_NETWORKS].to_numpy()
    sleep_array = df[SLEEP_MACRO].to_numpy()
    labels_s = ["N1", "N2", "N3", "Rem"]
    labels_c = ["DMN", "Motor", "PriVis", "DAN", "rFPCN", "lFPCN", "ExVis"]
    colors = [
        "darkblue",
        "blue",
        "mediumpurple",
        "thistle",
        "violet",
        "pink",
        "crimson",
    ]

    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2)
    ax0.hist(df["MRI_Age"])
    ax0.set_title("Age")
    ax1.hist(df["MRI_fcMRI_MeanMovement_QA"])
    ax1.set_title("Mean Movement")
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.savefig("age_head_motion_hist.png")

    fig, [[ax2, ax3], [ax4, ax5]] = plt.subplots(nrows=2, ncols=2)
    ax2.hist(df["DENS_15_CENTRAL"])
    ax2.set_title("DENS_15_CENTRAL")
    ax3.hist(df["FRQ_15_CENTRAL"])
    ax3.set_title("FRQ_15_CENTRAL")
    ax4.hist(df["CHIRP_15_CENTRAL"])
    ax4.set_title("CHIRP_15_CENTRAL")
    ax5.hist(df["ISA_S_15_CENTRAL"])
    ax5.set_title("ISA_S_15_CENTRAL")
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.savefig("fast_micro_sleep_hist.png")

    fig, [[ax6, ax7], [ax8, ax9]] = plt.subplots(nrows=2, ncols=2)
    ax6.hist(df["DENS_11_FRONTAL"])
    ax6.set_title("DENS_11_FRONTAL")
    ax7.hist(df["FRQ_11_FRONTAL"])
    ax7.set_title("FRQ_11_FRONTAL")
    ax8.hist(df["CHIRP_11_FRONTAL"])
    ax8.set_title("CHIRP_11_FRONTAL")
    ax9.hist(df["ISA_S_11_FRONTAL"])
    ax9.set_title("ISA_S_11_FRONTAL")
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.savefig("slow_micro_sleep_hist.png")

    plt.figure(figsize=(7, 5))
    plt.hist(
        sleep_array,
        bins=8,
        density=True,
        histtype="bar",
        color=colors[: len(SLEEP_MACRO)],
        label=labels_s,
    )
    plt.legend(prop={"size": 10})
    plt.title("sleep percentages")
    plt.savefig("sleep_percent_hist.png")

    plt.figure(figsize=(7, 5))
    plt.hist(
        conn_array, density=True, bins=8, histtype="bar", color=colors, label=labels_c
    )
    plt.legend(prop={"size": 10})
    plt.title("connectivity networks")
    plt.savefig("conn_networks_hist.png")
    descriptives = df[DATA].describe()
    descriptives.to_csv("descriptives.csv")


def plot_cor_matrix(corr, mask):
    """
    Saves heatmap graph of r values at a given pvalue, designated by mask of p < 0.05
    parameters: corr (dataframe), mask (all corr values with a p < 0.05)
    returns: None
    """
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        linewidths=2,
        linecolor="black",
        cbar_kws={"orientation": "vertical"},
        xticklabels=[
            "DENS_SLOW",
            "FRQ_SLOW",
            "CHIRP_SLOW",
            "ISA_S_SLOW",
            "DENS_FAST",
            "FRQ_FAST",
            "CHIRP_FAST",
            "ISA_S_FAST",
        ],
        yticklabels=[
            "DMN",
            "Mot",
            "PriVis",
            "DAN",
            "rFPCN",
            "lFPCN",
            "ExVis",
        ],
    )
    plt.savefig("correlations_matrix.png")


def calculate_pvalues(df):
    """
    Calculates p values from a give dataframe of data
    parameters: df (dataframe)
    returns: pvalues (dataframe)
    """
    df_cols = pd.DataFrame(columns=df.columns)
    p_values = df_cols.transpose().join(df_cols)
    corr_values = p_values.copy()
    for r in df:
        for c in df:
            corr_values[r][c] = float(stats.spearmanr(df[r], df[c])[0])
            p_values[r][c] = stats.spearmanr(df[r], df[c])[1]
    return corr_values, p_values


def basic_correlations(df):
    """
    runs pearson's correlations on sleep vs connectivity data, calculates p values &
    saves violin plots for N2 and DAN
    parameters: df (dataframe)
    returns: None
    """
    # FOR CONDENSED CORRELATIONS MATRIX FIGURE (change x&y labels)
    conn = CONNECTIVITY_NETWORKS
    sleep = SLEEP_MICRO_SLOW + SLEEP_MICRO_FAST
    df_conn = df[conn]
    df_conn_sleep_corr = pd.DataFrame()
    for column in sleep:
        df_sleep = df[column]
        correlations = df_conn.corrwith(df_sleep, axis=0, method="spearman")
        df_conn_sleep_corr[column] = correlations

    corr_values = calculate_pvalues(df[DATA])[0]
    float_corr_values = df[DATA].corr(method="spearman")
    p_values = calculate_pvalues(df[DATA])[1]
    corr_values.to_csv("r_correlations_spearman_condensed.csv")
    p_values.to_csv("p_values_spearman_condensed.csv")
    mask = np.invert(np.tril(p_values < 0.05))
    plot_cor_matrix(df_conn_sleep_corr, mask)
    sns.violinplot(y=df["Percent_N2"], color="green", inner="points")
    plt.ylabel(ylabel="Percentage of time spent in N2")
    plt.savefig("N2_violin.png")
    sns.violinplot(y=df["DAN"], inner="points")
    plt.ylabel(ylabel="Dorsal Attention Network Functional Connectivity")
    plt.savefig("DAN_violin.png")


def multiple_regressions(df, sleep_predictor, conn_outcome):
    """
    runs multiple regressions on signficant findings from basic correlations
    parameters: df (dataframe) sleep_predictor (string) sleep_name (strsing)
        conn_outcome (string) conn_name (string)
    returns: None
    """
    predictors = [
        df[["MRI_Age", "MRI_fcMRI_MeanMovement_QA", "Sex_binary"]],
        df[["MRI_Age", "MRI_fcMRI_MeanMovement_QA", "Sex_binary", sleep_predictor]],
    ]
    predictors_names = [
        ["age", "head motion", "sex"],
        ["age", "head motion", "sex", sleep_predictor],
    ]
    outcome = df[[conn_outcome]]
    model = HLR.HLR_model(
        diagnostics=True, showfig=True, save_folder="results", verbose=True
    )
    model_results, reg_models = model.run(
        X=predictors, X_names=predictors_names, y=outcome
    )
    model.save_results()
    new_predictors = sm.add_constant(predictors[1])
    model = sm.OLS(outcome, new_predictors)
    result = model.fit()


def main():
    """
    pulls df file & runs functions
    parameters: none
    returns: none
    """
    df = pd.read_csv(FILENAME, sep="\t")
    investigate(df)
    basic_correlations(df)
    multiple_regressions(df, "first variable", "second variable")


main()

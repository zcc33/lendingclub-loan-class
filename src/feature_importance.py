import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt
from clean_data import clean_data


if __name__ == "__main__":
    data = pd.read_pickle("../data/data.pkl")
    X, y = clean_data(data)

    logistic = joblib.load("../models/logistic.sav")
    randomforest = joblib.load("../models/randomforest.sav")
    xgboost = joblib.load("../models/xgboost.sav")

    top10_coefs_l = logistic.coef_[0][np.argsort(abs(logistic.coef_[0]))][::-1][:15]
    top10_names_l = X.columns[np.argsort(abs(logistic.coef_[0]))][::-1][:15]

    top10_coefs_r = randomforest.feature_importances_[np.argsort(randomforest.feature_importances_)][::-1][:15]
    top10_names_r = X.columns[np.argsort(randomforest.feature_importances_)][::-1][:15]

    top10_coefs_x = xgboost.feature_importances_[np.argsort(xgboost.feature_importances_)][::-1][:15]
    top10_names_x = X.columns[np.argsort(xgboost.feature_importances_)][::-1][:15]

    fig, axs = plt.subplots(3,1, figsize=(10,15), dpi=200, constrained_layout=True)

    sns.barplot(x=top10_coefs_l, y=top10_names_l, ax = axs[0])
    axs[0].set_title("Logistic Regression Top 15 Features", fontsize = 14)
    axs[0].set_xlabel("Coefficient Value")

    sns.barplot(x=top10_coefs_r, y=top10_names_r, ax = axs[1])
    axs[1].set_title("Random Forest Top 15 Features", fontsize = 14)
    axs[1].set_xlabel("Gini Importance")

    sns.barplot(x=top10_coefs_x, y=top10_names_x, ax = axs[2])
    axs[2].set_title("XGBoost Top 15 Features", fontsize = 14)
    axs[2].set_xlabel("Gini Importance")

    fig.savefig("../img/feature_importances.jpg")



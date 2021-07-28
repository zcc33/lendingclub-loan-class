#need to convert from notebook
import pandas as pd
import numpy as np
from clean_data import clean_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import matplotlib.pyplot as plt
import joblib


def plot_pr_curves(recall_d, precision_d, recall_l, precision_l, recall_x, precision_x):
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    ax.plot(recall_d, precision_d, 'o', label="Dummy classifier")
    ax.plot(recall_l, precision_l, label="Logistic regression")
    ax.plot(recall_r, precision_r, label="Random forest")
    ax.plot(recall_x, precision_x, label="XGBoost")
    ax.set_title("Precision-recall curves on test set")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    fig.savefig("../img/pr_curve.jpg")


if __name__ == "__main__":
    data = pd.read_pickle("../data/data.pkl")
    X, y = clean_data(data)

    metrics = pd.DataFrame({"PR AUC": np.zeros(4), "ROC AUC": np.zeros(4), "Brier loss": np.zeros(4)}, index = ["Dummy Classifier", "Logistic Regression", "Random Forest", "XGBoost"])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    log_params = {}
    rf_params = {}
    xgb_params = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    logistic = LogisticRegression(**log_params)
    logistic.fit(X_train_scaled, y_train)

    randomforest = RandomForestClassifier(**rf_params)
    randomforest.fit(X_train, y_train)

    xgboost = xgb.XGBClassifier(**xgb_params)


    probs = dummy.predict_proba(X_test)[:,1]
    precision_d, recall_d, _ = precision_recall_curve(y_test, probs)
    metrics["Dummy Classifier"]["PR AUC"] = auc(recall_d, precision_d)
    metrics["Dummy Classifier"]["ROC AUC"] = roc_auc_score(y_test, probs)
    metrics["Dummy Classifier"]["Brier loss"] = brier_score_loss(y_test, probs)

    probs = logistic.predict_proba(X_test)[:,1]
    precision_l, recall_l, _ = precision_recall_curve(y_test, probs)
    metrics["Logistic Regression"]["PR AUC"] = auc(recall_l, precision_l)
    metrics["Logistic Regression"]["ROC AUC"] = roc_auc_score(y_test, probs)
    metrics["Logistic Regression"]["Brier loss"] = brier_score_loss(y_test, probs)

    probs = randomforest.predict_proba(X_test)[:,1]
    precision_r, recall_r, _ = precision_recall_curve(y_test, probs)
    metrics["Random Forest"]["PR AUC"] = auc(recall_r, precision_r)
    metrics["Random Forest"]["ROC AUC"] = roc_auc_score(y_test, probs)
    metrics["Random Forest"]["Brier loss"] = brier_score_loss(y_test, probs)

    probs = xgboost.predict_proba(X_test)[:,1]
    precision_x, recall_x, _ = precision_recall_curve(y_test, probs)
    metrics["XGBoost"]["PR AUC"] = auc(recall_x, precision_x)
    metrics["XGBoost"]["ROC AUC"] = roc_auc_score(y_test, probs)
    metrics["XGBoost"]["Brier loss"] = brier_score_loss(y_test, probs)

    metrics.to_csv("../models/metrics.csv")

    plot_pr_curves(recall_d, precision_d, recall_l, precision_l, recall_x, precision_x)

    joblib.dump(logistic, "../models/logistic.sav")
    joblib.dump(randomforest, "../models/randomforest.sav")
    joblib.dump(xgboost, "../models/xgboost.sav")

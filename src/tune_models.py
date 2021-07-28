from typing import DefaultDict
import pandas as pd
import numpy as np
from clean_data import clean_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def tune_log(X_train_scaled, y_train):
    param1 = {'penalty': ['l2'], 'C':[0.1, 1, 10],  "solver": ["newton-cg", "lbfgs"], "max_iter": [1000]}
    logistic = LogisticRegression()
    clf = GridSearchCV(logistic, [param1], scoring = "neg_brier_score", verbose = 3)
    clf.fit(X_train_scaled, y_train)
    return(clf.best_params_)

def tune_rf(X_train, y_train):
    param = {'n_estimators': [20, 100, 200], 'max_depth':[None, 6, 15]}
    randomforest = RandomForestClassifier()
    clf = GridSearchCV(randomforest, param, scoring = "neg_brier_score", verbose = 3)
    clf.fit(X_train, y_train)
    return(clf.best_params_)

def tune_xgb(X_train, y_train):
    param = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 'learning_rate':[0.1, 0.3], "use_label_encoder": [False], "verbosity": [0]}
    print("Performing grid search cross validation")
    xgb_clf = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_clf, param, scoring = "neg_brier_score", verbose = 3)
    clf.fit(X_train, y_train)
    return(clf.best_params_)

if __name__ == "__main__":
    print("Loading dataset")
    data = pd.read_pickle("../data/data.pkl")
    X, y = clean_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    print("Performing grid search cv for logistic regression")
    log_params = tune_log(X_train_scaled, y_train)

    print("Performing grid search cv for random forest")
    rf_params = tune_rf(X_train, y_train)

    print("Performing grid search cv for xgboost")
    xgb_params = tune_xgb(X_train, y_train)
    
    print("Finished tuning. Outputting best parameters to file")

    with open("../models/params.txt", "w") as f:
        print("Logistic regression parameters:", file=f)
        print(log_params, file=f)
        print("Random forest parameters:", file=f)
        print(rf_params, file=f)
        print("XGBoost parameters:", file=f)
        print(xgb_params, file=f)

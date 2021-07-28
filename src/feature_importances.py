import pandas as pd
import numpy as np
import joblib



if __name__ == "__main__":
    logistic = joblib.load("../models/logistic.sav")
    randomforest = joblib.load("../models/randomforest.sav")
    xgboost = joblib.load("../models/xgboost.sav")

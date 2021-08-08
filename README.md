# LendingClub Loan Classification

## Table of Contents

1. [Background](#background)
2. [Data Description and Processing](#dataset)
3. [Data Processing](#processing)
4. [EDA Plots](#eda)
5. [Objectives](#objectives)
7. [Class Imbalance and Scoring](#class)
8. [Model Results](#results)
9. [Conclusions and Future Work](#future)


## Background <a name ="background"> </a>

[LendingClub](https://www.lendingclub.com/) is a fin-tech company started in 2006, whose focus is allowing consumers to access financial products using the internet. For most of its history, LendingClub's main product was facilitating unsecured peer-to-peer loans ranging from $1000 to $40,000. At its 2015 peak, LendingClub was the largest peer-to-peer loan platform in the world. At that point, $15.98 billion in loans had been originated through its history. In the fall of 2020, LendingClub ended its peer-to-peer lending program for undisclosed reasons to focus on commercial banking.

As of 2015, according to [Wikipedia](https://en.wikipedia.org/wiki/LendingClub#Loan_performance_statistics), the average LendingClub borrower had the following characteristics:

* FICO score of 699
* Debt-to-income ratio of 17.7%
* Credit history of 16.2 years
* Annual personal income of $73,945
* Sought out a loan of $14,553 for debt consolidation or paying off credit cards

On the investor side, LendingClub peer investors:

* Funded $11 billion in total loans, including $2 billion in Q2 2015
* Funded loans with an average nominal interest rate of 14% and default rate of 3.39% 
* Earned average net annualized returns of 8.93%.

### Project Overview 
The puprose of 


## Description of Data <a name ="dataset"> </a>

We used the most comprehensive LendingClub dataset available. It’s described as all loans from 2007 to 2018, but actually only contains 2012 to 2018.

LendingClub no longer offers data or information about loans on its website.

The data set contains over 2 million loans, over 150 variables.

Since variables are only available for accepted loans, we can only build models on loans that were accepted by LendingClub. This is a very different task than if we could build models on all loan applications. 

Target variable: Loan Status {“Current”, “Fully Paid”, “Charged Off”}

Seemingly useful predictors: FICO score, grade, interest rate, annual income, employee length, and many others

Need to prevent data leakage! ---> Filter out the variables collected after loan origination

### Distribution of non-current Loans
![](img/paid_off.jpg)


## Data Processing <a name ="processing"> </a>
Challenge 1: Too large to load

-Leveraged dask to load the full dataset and then drop columns.

-Performed memory management with dtypes to optimize pandas speed.

-Challenge 2: Categorical & ordinal data

-Converted ordinal values to numeric. For example, grade and length of employment.

Some categorical variables needed 50+ dummies (like state).

Challenge 3: Missing values

-Filtered out columns with greater than 10% missing values.

-Dropped rows with any missing values (checked to see this didn’t alter target variable much).

### Final Dataset
Shape: (152,069, 104)

152,069 loans from 2012 to 2013:
    15.4% charged-off (label 1)
    84.6% fully paid (label 0)

~40 predictor variables, with dummy categories bringing columns to 104



## EDA Plots <a name ="eda"> </a>


![](img/Log%20Annual%20Income.jpg)
![](img/FICO%20score.jpg)
![](img/Borrower%20State.jpg)
![](img/Loan%20Grades.jpg)



## Objectives <a name ="objectives"> </a>
Potential targets:

**Predict charge-off rates                  (classification task)**

Predict whether loan was profitable     (classification task)

Predict return (%) from loan             (regression task)



Candidate models (for classification):

-Logistic regression

-Random forest

-XGBoost (gradient boosted trees)




## Class Imbalance and Scoring <a name ="class"> </a>
15% vs 85% imbalance

However:
There are much worse cases out there (1% vs 99%)
There is a sufficient number of the minority class (tens of thousands)
The class imbalance also exists in the population

We could: undersample majority class, make synthetic data from minority class (SMOTE), use “class_weight = balanced” in model loss functions

Models will want to predict everything is in the 85% group (Fully Paid). This simply depends on the threshold used.

### Scoring metric
If we use threshold-agnostic metrics to evaluate our models, class imbalance shouldn’t be a problem. E.g. ROC AUC, brier score

Accuracy, recall, and precision all depend on the threshold used.



ROC AUC is threshold-agnostic and gives more nuanced information about the model. Since the data is imbalanced, better to use Precision-Recall AUC.



Brier loss is also threshold-agnostic. It uses the direct probabilities given by the model. It’s the mean squared error between target outcome and predicted probability.

Brier loss ranges from 0 (good) to 1 (bad). 

A random 50/50 model would have a Brier loss of 0.25.


## Model Results <a name ="model"> </a>

|  | Configuration | Train Scores | Test Scores
| --- | ---| --- | ---|
| Dummy classifier (majority) | Gives perfect probability of <br> being majority class | n/a | PR AUC: 0.577 <br> Brier loss: 0.154 <br> ROC AUC: 0.5 <br>
| Logistic Regression | Standardized data <br> Default regularization <br> Max_iter = 1000 | PR AUC: 0.290 <br> Brier loss: 0.121 <br> ROC AUC: 0.7 | PR AUC: 0.291 <br> Brier loss: 0.121 <br> ROC AUC: 0.7
| Random Forest | max_depth = 6 <br> n_estimators = 20 | PR AUC: 0.294 <br> Brier loss: 0.123 <br> ROC AUC: 0.698 | PR AUC: 0.279 <br> Brier loss: 0.123 <br> ROC AUC: 0.689
| XGBoost | max_depth = 5 <br> n_estimators = 10 <br> eta (learning rate) = 0.1 | PR AUC: 0.289 <br> Brier loss: 0.138 <br> ROC AUC: 0.696 | PR AUC: 0.28 <br> Brier loss: 0.138 <br> ROC AUC: 0.689


![](img/pr_curve.jpg)


## Conclusions and Future Work <a name ="future"> </a>

Surprisingly poor performance despite strong variables (grade, FICO score, interest rate).

Just as an example, to get a recall of 0.8, we would have precision of only 0.2.

All models (logistic, random forest, xgboost) performed similarly.

Possibly because we are training on loans that were already accepted by LendingClub, so hard to discriminate them further based on variables that LendingClub already used.

Future work could be to train on other targets, like profitability. Also quantify how much better our model does than random in terms of profitability for an investor.
## References

1. https://www.kaggle.com/wordsforthewise/lending-club
2. https://en.wikipedia.org/wiki/LendingClub
3. https://resources.lendingclub.com/LCDataDictionary.xlsx
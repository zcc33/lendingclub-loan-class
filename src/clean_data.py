import pandas as pd

def clean_data(data):
    mapping = {"A": 7, "B":6, "C":5, "D":4, "E":3, "F":2, "G": 1}
    data["grade"] = data["grade"].map(mapping)

    term_dummies = pd.get_dummies(data['term'], drop_first = True, prefix="term")
    home_ownership_dummies = pd.get_dummies(data['home_ownership'], drop_first = True, prefix="home_ownership")
    verification_status_dummies = pd.get_dummies(data['verification_status'], drop_first = True, prefix="verification_status")
    purpose_dummies = pd.get_dummies(data['purpose'], drop_first = True, prefix="purpose")
    addr_state_dummies = pd.get_dummies(data['addr_state'], drop_first = True, prefix="addr_state")
    initial_list_status_dummies = pd.get_dummies(data['initial_list_status'], drop_first = True, prefix="initial_list_status")

    data = pd.concat([data, term_dummies, home_ownership_dummies, verification_status_dummies, purpose_dummies, addr_state_dummies, initial_list_status_dummies], axis=1)

    data = data.drop(['term', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status'], axis=1)

    mapping = {"Charged Off": 1, "Fully Paid": 0}
    y = data["loan_status"].map(mapping).astype("float32")
    X = data.drop(['loan_status', 'funded_amnt', 'issue_d', 'profitable', 'return_pct', 'total_pymnt'], axis=1)

    for i in X.columns:
        if X[i].dtype.name=="category":
            X[i] = X[i].astype("float32")

    return X, y


if __name__ == "__main__":
    data = pd.read_pickle("../data/data.pkl")
    X, y = clean_data(data)
    print(X.head())
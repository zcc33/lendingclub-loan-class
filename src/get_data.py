import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

def make_paid_figure(data):
    year = range(2012, 2019)
    num_loans = []
    percent_paidoff = []
    for i in year:
        loans = (data["issue_d"].dt.year == i).sum()
        paid = (data[data["issue_d"].dt.year==i].loan_status == "Fully Paid").sum()
        num_loans.append(loans)
        percent_paidoff.append((paid/loans)*100)

    temp = pd.DataFrame({"year": year, "num_loans": num_loans, "percent_paidoff": percent_paidoff})

    sns.set_theme()
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    sns.set_style("ticks")
    sns.set_context('notebook')

    fig, axs = plt.subplots(1,2, figsize=(15,6))
    sns.barplot(data=temp, x="year", y="num_loans", ax = axs[0])
    axs[0].set_title("Total non-Current Loans", fontsize = 16)
    axs[0].set_ylabel("Loans")
    axs[0].set_xlabel("Year Issued")

    sns.lineplot(data=temp, x="year", y="percent_paidoff", linewidth= 3, ax = axs[1])
    axs[1].set_title("Percentage of non-Current Loans that are Fully Paid", fontsize = 16)
    axs[1].set_ylabel("Fully Paid (%)")
    axs[1].set_xlabel("Year Issued")
    axs[1].set(ylim=(65, 90))
    plt.tight_layout()

    fig.savefig("../img/paid_off.jpg")

def dask_load():
    #load using dask
    df = dd.read_csv("../data/accepted_2007_to_2018Q4.csv", assume_missing = True, dtype={'desc': 'object', 'id': 'object', 'sec_app_earliest_cr_line': 'object', 'debt_settlement_flag_date': 'object',
    'hardship_end_date': 'object', 'hardship_loan_status': 'object', 'hardship_reason': 'object', 'hardship_start_date': 'object', 'hardship_status': 'object', 'hardship_type': 'object', 'payment_plan_start_date': 'object',
    'settlement_date': 'object', 'settlement_status': 'object'})

    pct_missing_values = df.isna().sum(axis=0).compute()/len(df)
    thresh = 0.1

    #columns with a large percentage of missing values
    missing_cols = list(pct_missing_values[pct_missing_values>thresh].index)

    #manually define leaky variables and unneeded columns
    leaky_cols = ['funded_amnt_inv', 'inq_last_6mths', 'out_prncp', 'out_prncp_inv', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'tot_coll_amt', 'tot_cur_bal', 'mths_since_recent_bc',
    'acc_open_past_24mths', 'avg_cur_bal', 'num_accts_ever_120_pd', 'tot_hi_cred_lim', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'collections_12_mths_ex_med', 'debt_settlement_flag', 
    'chargeoff_within_12_mths', 'delinq_amnt', 'inq_last_6mths', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd']

    unneeded_cols = ['id', 'installment', 'sub_grade', 'emp_title', 'url', 'title', 'zip_code', 'earliest_cr_line']

    drop_cols = missing_cols +leaky_cols + unneeded_cols

    #we only want loans that are not current
    df = df[df.loan_status != "Current"]

    return df.drop(drop_cols, axis=1).compute()

if __name__ == "__main__":
    #load using dask and then convert into pandas dataframe
    data = dask_load()
    print("dask loading done")

    #drop missing values - this only changes good loan percentage from 77.89% to 78.20%, which is not that big
    data.dropna(inplace=True)

    #memory management
    data = data.astype({'loan_amnt': 'float32', 'funded_amnt': 'float32', 'term': 'category', 'int_rate': 'float32', 'grade': 'category', 'emp_length': 'category', 'home_ownership': 'category', 'verification_status': 'category',\
    'loan_status': 'category', 'pymnt_plan': 'category', 'purpose': 'category', 'addr_state': 'category', 'dti': 'float32', 'delinq_2yrs': 'int32', 'initial_list_status': 'category', 'policy_code': 'category', 'mo_sin_rcnt_rev_tl_op': 'int32',\
    'mo_sin_rcnt_tl': 'int32', 'mort_acc': 'int32', 'num_sats': 'int32', 'pub_rec_bankruptcies': 'int32', 'tax_liens': 'int32', 'application_type': 'category', 'hardship_flag': 'category', 'disbursement_method': 'category',\
    'issue_d': 'datetime64[ns]', 'open_acc': 'int32', 'pub_rec': 'int32'})

    #make chart showing paid-off rates by year in the dataset
    make_paid_figure(data)
    print("figure made")
    
    data = data[(data["issue_d"].dt.year == 2013) | (data["issue_d"].dt.year == 2012)]
    data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Charged Off')]

    for col in data.columns:
        if data[col].dtype.name=='category':
            data[col] = data[col].cat.remove_unused_categories()

    mapping = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years":10}
    data["emp_length"] = data["emp_length"].map(mapping)

    data.drop(columns=["disbursement_method", "hardship_flag", "pymnt_plan", "application_type", "policy_code"], inplace=True)

    data["fico"] = (data["fico_range_low"] + data["fico_range_high"])/2
    data.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)

    data["profitable"] = data["total_pymnt"] > data["funded_amnt"]
    data["return_pct"] = ((data["total_pymnt"] - data["funded_amnt"])/data["funded_amnt"])*100

    data.to_pickle("../data/data.pkl")
    print("saved as pickle file")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cont(col, feature_name):
    sns.set_theme()
    sns.set(rc={"figure.dpi":200, 'savefig.dpi':200})
    sns.set_style("ticks")
    sns.set_context('notebook')

    fig, axs = plt.subplots(1,2, figsize = (15,6))
    sns.histplot(col, ax = axs[0])
    axs[0].set_title(feature_name + " histogram", fontsize = 16)
    axs[0].set_ylabel("Counts")
    axs[0].set_xlabel(feature_name)

    sns.boxplot(x=col, y=data["loan_status"], width=0.3, ax=axs[1])
    axs[1].set_title(feature_name + " by loan status", fontsize = 16)
    axs[1].set_ylabel("")
    axs[1].set_xlabel(feature_name)

    plt.tight_layout()
    fig.savefig("../img/"+feature_name+".jpg")


def plot_cat(col_name, feature_name, rotation_amt = 0):
    sns.set_theme()
    sns.set(rc={"figure.dpi":200, 'savefig.dpi':200})
    sns.set_style("ticks")
    sns.set_context('notebook')

    cats = np.sort(data[col_name].unique())
 
    fig, axs = plt.subplots(1,2, figsize = (15,6))
    g = sns.countplot(x=col_name, data =data, order = cats, ax = axs[0])
    axs[0].set_title(feature_name + " histogram", fontsize = 16)
    axs[0].set_ylabel("Counts")
    axs[0].set_xlabel(feature_name)
    g.set_xticklabels(labels = cats, rotation=rotation_amt)

    #rates = data.groupby(col_name)["loan_status"].value_counts(normalize=True).loc[:,"Charged Off"]
    rates = []
    for i in cats:
        temp = data[data[col_name]==i]["loan_status"].value_counts(normalize=True)
        if "Charged Off" in temp.index:
            rates.append(temp.loc["Charged Off"])
        else:
            rates.append(0)
            
    h = sns.barplot(x=cats, y=rates, ax=axs[1])
    axs[1].set_title("Charge-off rates for varying " + feature_name, fontsize = 16)
    axs[1].set_ylabel("Charge-off rate")
    axs[1].set_xlabel(feature_name)
    h.set_xticklabels(labels = cats, rotation=rotation_amt)

    plt.tight_layout()

    fig.savefig("../img/"+feature_name+".jpg")

if __name__ == "__main__":
    data = pd.read_pickle("../data/data.pkl")
    plot_cont(np.log(data["annual_inc"]), "Log Annual Income")
    plot_cont(data["fico"], "FICO score")
    plot_cat("addr_state", "Borrower State", 90)
    plot_cat("grade", "Loan Grades", 0)
    



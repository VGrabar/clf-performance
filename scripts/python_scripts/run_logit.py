import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if __name__ == "__main__":

    train_data_path = "data/rosbank/trainvalid.csv"
    test_data_folder = "data/rosbank/test"
    test_data_list = glob.glob(test_data_folder + "/*.csv")

    dataset = pd.read_csv(train_data_path, delimiter=",")
    # categorical variables
    dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
    cat_vars = ["mcc_last", "mcc_mode"]
    for var in cat_vars:
        cat_list = "var" + "_" + var
        cat_list = pd.get_dummies(dataset[var], prefix=var)
        data_new = dataset.join(cat_list)
        dataset = data_new
    dataset.drop(cat_vars, axis=1, inplace=True)
    X = dataset.loc[:, dataset.columns != "label"]
    y = dataset.loc[:, dataset.columns == "label"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=23)

    # logreg
    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    print("Accuracy of logistic regression classifier on test set: {:.2f}".format(logreg.score(X, y)))
    metrics = {}
    for test_data_path in test_data_list:
        period_name = test_data_path.split("/")[-1]
        period_name = period_name.split(".")[0]
        period_name = period_name.split("_")[-1]
        print(f"period name: {period_name}")
        metrics[period_name] = {}
        dataset = pd.read_csv(test_data_path, delimiter=",")
        dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
        # categorical variables
        cat_vars = ["mcc_last", "mcc_mode"]
        # for var in cat_vars:
        #     cat_list = "var" + "_" + var
        #     cat_list = pd.get_dummies(dataset[var], prefix=var)
        #     data_new = dataset.join(cat_list)
        #     dataset = data_new
        data_new = dataset.join(cat_list)
        dataset = data_new
        dataset.drop(cat_vars, axis=1, inplace=True)
        X_test = dataset.loc[:, dataset.columns != "label"]
        y_true = dataset.loc[:, dataset.columns == "label"]
        y_pred = logreg.predict(X_test)
        y_probs = logreg.predict_proba(X)
        bce_loss = metrics.log_loss(y_true, y_probs)

        scores = precision_recall_fscore_support(y_true.to_list(), y_pred, pos_label=1, average="binary")

        metrics[period_name]["precision"] = scores[0]
        metrics[period_name]["recall"] = scores[1]
        metrics[period_name]["fscore"] = scores[2]
        metrics[period_name]["bce"] = bce_loss

    with open("data/rosbank/plot_logit.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # # xgboost
    # model = XGBClassifier()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]

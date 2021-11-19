import glob
import json
import os
import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def main(dataset_name: str, clf_type: str):

    train_data_path = os.path.join("data", dataset_name, "test.csv")
    test_data_folder = os.path.join("data", dataset_name, "test")
    test_data_list = glob.glob(test_data_folder + "/*.csv")

    dataset = pd.read_csv(train_data_path, delimiter=",", converters={"mcc_ratios": pd.eval})
    # normalize
    unnorm_cols = ["amounts_std", "amounts_med", "amounts_avg"]
    for feature_name in unnorm_cols:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        dataset[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
    # split list column
    mcc_cols = ["mcc_" + str(i) for i in range(0, 500)]
    split_df = pd.DataFrame(dataset["mcc_ratios"].tolist(), columns=mcc_cols)
    dataset.drop(["mcc_ratios", "Unnamed: 0", "client_id"], axis=1, inplace=True)
    dataset = pd.concat([dataset, split_df], axis=1)

    X = dataset.loc[:, dataset.columns != "label"]
    y = dataset.loc[:, dataset.columns == "label"]

    # logreg
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    print("Accuracy of logistic regression classifier on train set: {:.2f}".format(logreg.score(X, y)))
    # xgboost
    xgbmodel = XGBClassifier()
    xgbmodel.fit(X, y)
    y_pred = xgbmodel.predict(X)
    print("Accuracy of xgboost classifier on train set: {:.2f}".format(accuracy_score(y, y_pred)))
    for model in ["logit", "xgboost"]:
        metrics = {}
        for test_data_path in test_data_list:
            period_name = test_data_path.split("/")[-1]
            period_name = period_name.split(".")[0]
            period_name = period_name.split("_")[-1]
            print(f"period name: {period_name}")
            metrics[period_name] = {}
            metrics[period_name]["roc_auc"] = []
            dataset = pd.read_csv(test_data_path, delimiter=",", converters={"mcc_ratios": pd.eval})
            # normalize
            unnorm_cols = ["amounts_std", "amounts_med", "amounts_avg"]
            for feature_name in unnorm_cols:
                max_value = dataset[feature_name].max()
                min_value = dataset[feature_name].min()
                dataset[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
            # split list column
            mcc_cols = ["mcc_" + str(i) for i in range(0, 500)]
            split_df = pd.DataFrame(dataset["mcc_ratios"].tolist(), columns=mcc_cols)
            dataset.drop(["mcc_ratios", "Unnamed: 0", "client_id"], axis=1, inplace=True)
            dataset = pd.concat([dataset, split_df], axis=1)
            X_test = dataset.loc[:, dataset.columns != "label"]
            y_true = dataset.loc[:, dataset.columns == "label"]
            if model == "logit":
                y_pred = logreg.predict(X_test)
                y_probs = logreg.predict_proba(X_test)
                if clf_type == "binary":
                    y_probs = y_probs[:, 1]
            elif model == "xgboost":
                y_pred = xgbmodel.predict(X_test)
                y_probs = xgbmodel.predict_proba(X_test)
                if clf_type == "binary":
                    y_probs = y_probs[:, 1]

            print("Accuracy of " + model + " classifier on test set: {:.2f}".format(accuracy_score(y_true, y_pred)))
            bce_loss = log_loss(y_true, y_probs)
            metrics[period_name]["bce"] = bce_loss

            if clf_type == "binary":
                metrics[period_name]["roc_auc"].append(roc_auc_score(y_true, y_probs, average="macro"))
            else:
                metrics[period_name]["roc_auc"].append(
                    roc_auc_score(y_true, y_probs, multi_class="ovo", average="weighted")
                )

        with open(os.path.join("data", dataset_name, "plot_" + model + ".json"), "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)

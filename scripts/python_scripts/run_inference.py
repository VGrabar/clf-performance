import glob
import json
import os

import numpy as np
import pandas as pd
import typer
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold

from advsber.utils.data import load_jsonlines


def get_predictor(archive_path: str) -> Predictor:
    archive = load_archive(archive_path, cuda_device=-1)
    predictor = Predictor.from_archive(archive=archive, predictor_name="transactions")
    return predictor


def main(
    data_folder: str,
    model_path: str,
    save_path: str,
):
    data_list = glob.glob(os.path.join(data_folder, "*.jsonl"))
    data_list.sort()
    metrics = {}

    for i, data_path in enumerate(data_list):
        output = load_jsonlines(data_path)
        output = pd.DataFrame(output)
        if i == 0:
            data_prev = []
            preds = []

        period_name = data_path.split("/")[-1]
        period_name = period_name.split(".")[0]
        # period_name = period_name.split("_")[-1]
        print(f"period name:{period_name}")
        metrics[period_name] = {}
        if model_path is not None:
            predictor = get_predictor(model_path)
            data = [
                {"transactions": row["transactions"], "amounts": row["amounts"]} for index, row in output.iterrows()
            ]
            curr_data = [d for d in data if (d not in data_prev)]
            curr_preds = predictor.predict_batch_json(curr_data)
            print("curr: ", len(curr_preds))
            data_prev = data
            preds.extend(curr_preds)
            print("total: ", len(preds))

        cross_entropy = []
        pred_labels = []
        pred_probs = []

        for pred_row in preds:
            lbl = int(pred_row["label"])
            cross_entropy.append(-1 * pred_row["logits"][lbl])
            pred_labels.append(lbl)
            pred_probs.append(pred_row["probs"][lbl])

        gt_labels = np.array(output["label"])
        cross_entropy = np.array(cross_entropy)
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)
        # splitting by folds
        metrics[period_name]["roc_auc"] = []
        metrics[period_name]["average_precision"] = []
        metrics[period_name]["precision"] = []
        metrics[period_name]["recall"] = []
        metrics[period_name]["fscore"] = []
        metrics[period_name]["bce"] = []
        # y_score = clf.predict_proba(X)[:, 1]
        print("labels distr:")
        print(np.bincount(gt_labels))

        kfold = 5
        random_range = np.random.permutation(len(preds))
        ind_folds = np.array_split(random_range, kfold)

        for ind in ind_folds:
            print(len(ind))
            curr_labels = gt_labels[ind]
            curr_pred_labels = pred_labels[ind]
            curr_pred_probs = pred_probs[ind]
            scores = precision_recall_fscore_support(curr_labels, curr_pred_labels, pos_label=1, average="binary")

            metrics[period_name]["roc_auc"].append(roc_auc_score(curr_labels, curr_pred_probs, average="macro"))
            metrics[period_name]["average_precision"].append(
                average_precision_score(curr_labels, curr_pred_probs, average="macro", pos_label=1)
            )
            metrics[period_name]["precision"].append(scores[0])
            metrics[period_name]["recall"].append(scores[1])
            metrics[period_name]["fscore"].append(scores[2])
            bce = cross_entropy[ind].sum() / len(ind)
            metrics[period_name]["bce"].append(bce)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)

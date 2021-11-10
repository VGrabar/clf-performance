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
from more_itertools import sliced

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
            chunk_size = len(output)
        preds = []
        cross_entropy = []
        pred_labels = []
        pred_probs = []
        gt_labels = output["label"].tolist()

        period_name = data_path.split("/")[-1]
        period_name = period_name.split(".")[0]
        print(f"period name:{period_name}")
        metrics[period_name] = {}
        index_slices = sliced(range(len(output)), chunk_size//8)
        if model_path is not None:
            predictor = get_predictor(model_path)
            k = 0
            for index_slice in index_slices:
                k += len(index_slice)
                print(f"curr:{k} out of {len(output)}")
                curr_data = [
                    {"transactions": row["transactions"], "amounts": row["amounts"]} for index, row in output.iloc[index_slice].iterrows()]
                curr_preds = predictor.predict_batch_json(curr_data)
                preds.extend(curr_preds)


        for pred_row in preds:
            lbl = int(pred_row["label"])
            c_e_loss = sum(-x*y for x, y in list(zip(pred_row["logits"], pred_row["probs"]))) 
            cross_entropy.append(c_e_loss)
            pred_labels.append(lbl)
            pred_probs.append(pred_row["probs"][lbl])

        print("gt:",len(gt_labels))
        print("pred:",len(pred_labels))
        # splitting by folds
        metrics[period_name]["roc_auc"] = []
        metrics[period_name]["average_precision"] = []
        metrics[period_name]["precision"] = []
        metrics[period_name]["recall"] = []
        metrics[period_name]["fscore"] = []
        metrics[period_name]["bce"] = []
        print("labels distr:")
        print(np.bincount(gt_labels))

        kfold = 5
        random_range = np.random.permutation(len(preds))
        ind_folds = np.array_split(random_range, kfold)

        for ind in ind_folds:
            curr_labels = np.array(gt_labels)[ind]
            curr_pred_labels = np.array(pred_labels)[ind]
            curr_pred_probs = np.array(pred_probs)[ind]
            scores = precision_recall_fscore_support(curr_labels, curr_pred_labels, pos_label=1, average="binary")
            #scores = precision_recall_fscore_support(curr_labels, curr_pred_labels, average="weighted")

            metrics[period_name]["roc_auc"].append(roc_auc_score(curr_labels, curr_pred_probs, average="macro"))
            #metrics[period_name]["roc_auc"].append(roc_auc_score(curr_labels, curr_pred_probs, multi_class="ovo",average="weighted"))
            metrics[period_name]["average_precision"].append(
                average_precision_score(curr_labels, curr_pred_probs, average="macro", pos_label=1)
            )
            metrics[period_name]["average_precision"].append([0])
            metrics[period_name]["precision"].append(scores[0])
            metrics[period_name]["recall"].append(scores[1])
            metrics[period_name]["fscore"].append(scores[2])
            bce = np.array(cross_entropy)[ind].sum() / len(ind)
            metrics[period_name]["bce"].append(bce)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)

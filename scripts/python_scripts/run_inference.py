import json
import glob
import os
import typer
import pandas as pd
import numpy as np
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
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

        period_name = data_path.split("/")[-1]
        period_name = period_name.split(".")[0]
        period_name = period_name.split("_")[-1]
        print(f"period name:{period_name}")
        metrics[period_name] = {}
        preds = []
        probs = []
        if model_path is not None:
            predictor = get_predictor(model_path)
            index_slices = sliced(range(len(output)), chunk_size)
            for index_slice in index_slices:
                print("chunk:", len(index_slice))
                chunk = output.iloc[index_slice]
                data = [
                    {"transactions": row["transactions"], "amounts": row["amounts"]} for index, row in chunk.iterrows()
                ]
                curr_preds = predictor.predict_batch_json(data)
                print()
                preds.extend(curr_preds)
                print(len(preds))

        cross_entropy = []
        pred_labels = []
        pred_probs = []

        for pred_row in preds:
            lbl = int(pred_row["label"])
            pred_labels.append(lbl)
            cross_entropy.append(-1 * pred_row["logits"][lbl])
            pred_probs.append(pred_row["probs"][lbl])

        output["label"] = np.array(output["label"])
        output["loss"] = np.array(cross_entropy)
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
        kfold = 5
        len_fold = len(output) // kfold
        ind_folds = [list(range(i, i + len_fold)) for i in range(0, len(output), len_fold)]
        for ind in ind_folds:
            print(ind)
            print(len(ind))
            curr_labels = output["label"][ind]
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
            bce = output["loss"][ind].sum() / len(ind)
            metrics[period_name]["bce"].append(bce)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # for i, pred in enumerate(preds):
    #     label = pred["label"]
    #     prob = pred["probs"][
    #         predictor._model.vocab.get_token_index(str(output["data"][i]["label"]), namespace="labels")
    #     ]
    #     output["adversarial_data"][i]["label"] = int(label)
    #     output["adversarial_probability"][i] = prob

    # y_true = [output["data"][i]["label"] for i in range(len(output))]
    # y_adv = [output["adversarial_data"][i]["label"] for i in range(len(output))]
    # nad = normalized_accuracy_drop(wers=output["wer"], y_true=y_true, y_adv=y_adv)
    # typer.echo(f"NAD = {nad:.2f}")

    # misclf_error = misclassification_error(y_true=y_true, y_adv=y_adv)
    # typer.echo(f"Misclassification Error = {misclf_error:.2f}")

    # prob_drop = probability_drop(true_prob=output["probability"], adv_prob=output["adversarial_probability"])
    # typer.echo(f"Probability drop = {prob_drop:.2f}")

    # mean_wer = float(np.mean(output["wer"]))
    # typer.echo(f"Mean WER = {mean_wer:.2f}")

    # added_amounts = []
    # for _, row in output.iterrows():
    #     added_amounts.append(sum(row["adversarial_data"]["amounts"]) - sum(row["data"]["amounts"]))

    # anad = amount_normalized_accuracy_drop(added_amounts, y_true=y_true, y_adv=y_adv)
    # typer.echo(f"aNAD-1000 = {anad:.2f}")
    # try:
    #     diversity = diversity_rate(output)
    #     diversity = round(diversity, 3)
    # except ValueError:
    #     diversity = None
    # typer.echo(f"Diversity_rate = {diversity}")

    # if lm_path is not None:
    #     perplexity = calculate_perplexity(
    #         [adv_example["transactions"] for adv_example in output["adversarial_data"]], get_predictor(lm_path)
    #     )
    #     typer.echo(f"perplexity = {perplexity}")
    # else:
    #     perplexity = None

    # if save_to is not None:
    #     metrics = {
    #         "NAD": round(nad, 3),
    #         "ME": round(misclf_error, 3),
    #         "PD": round(prob_drop, 3),
    #         "Mean_WER": round(mean_wer, 3),
    #         "aNAD-1000": round(anad, 3),
    #         "diversity_rate": diversity,
    #         "perplexity": perplexity,
    #     }
    #     with open(save_to, "w") as f:
    #         json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)

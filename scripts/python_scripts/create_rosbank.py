import pandas as pd
import numpy as np
from datetime import datetime
import json
import jsonlines
import typer
import argparse

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_slice_subsample(sub_data, cnt_min, cnt_max, split_count):
    sub_datas = []
    cnt_min = cnt_min if len(sub_data) > cnt_max else int(cnt_min * len(sub_data) / cnt_max)
    cnt_max = cnt_max if len(sub_data) > cnt_max else len(sub_data) - 1
    split_count = split_count if len(sub_data) > cnt_max else int(len(sub_data) / cnt_max * split_count)
    for i in range(0, split_count):
        if cnt_min < cnt_max:
            T_i = np.random.randint(cnt_min, cnt_max)
            s = np.random.randint(0, len(sub_data) - T_i - 1)
            S_i = sub_data[s : s + T_i - 1]
            sub_datas.append(S_i)
    return sub_datas


def create_set(name, data, target, period: bool):
    len_ = len(np.unique(target.client_id))
    dict_data = {}
    with jsonlines.open(name, "w") as writer:
        for client_id in tqdm(np.unique(target.client_id)):
            sub_data = data[data["client_id"] == client_id]
            sub_data_target = target[target["client_id"] == client_id]
            sub_datas = split_slice_subsample(sub_data, 25, 150, 30)
            for loc_data in sub_datas:
                if len(loc_data.small_group) > 3:
                    loc_dict = {
                        "transactions": list(loc_data.small_group),
                        "amounts": list(loc_data.amount_rur),
                        "label": int(sub_data_target.bins),
                        "client_id": int(client_id),
                    }
                    if period:
                        loc_dict["period"] = list(loc_data.transaction_period)
                    writer.write(loc_dict)

    return


def split_data(dir_, data, target_data, period: bool):
    """
    Train-val split and saving as jsonlines
    """
    target_data_train, target_data_valid = train_test_split(target_data, test_size=0.2, random_state=10, shuffle=True)
    print("Creating train set...")
    create_set(dir_ + "/" + "train.jsonl", data, target_data_train, period)
    print("Creating valid set...")
    create_set(str(dir_) + "/" + "valid.jsonl", data, target_data_valid, period)
    return


def main(dataset_name: str, percentage: str):

    # dataset_name = "rosbank"
    transactions = pd.read_csv("data/" + dataset_name + "/original/train.csv")
    full_len = len(transactions)
    # filter out observations
    transactions = transactions.sort_values(by=["TRDATETIME"])
    # splitting train and test by transaction time
    train_transactions = transactions[: int(full_len * int(percentage) / 100)]
    test_transactions = transactions[int(full_len * int(percentage) / 100) :]

    train_target_data = train_transactions[["cl_id", "target_flag"]]
    train_target_data = train_target_data.drop_duplicates()
    train_target_data.reset_index(drop=True, inplace=True)
    test_target_data = test_transactions[["cl_id", "target_flag"]]
    test_target_data = test_target_data.drop_duplicates()
    test_target_data.reset_index(drop=True, inplace=True)

    train_data = train_transactions.rename(
        columns={"cl_id": "client_id", "MCC": "small_group", "amount": "amount_rur"}
    )
    train_target_data = train_target_data.rename(columns={"cl_id": "client_id", "target_flag": "bins"})
    test_data = test_transactions.rename(
        columns={"PERIOD": "transaction_period", "cl_id": "client_id", "MCC": "small_group", "amount": "amount_rur"}
    )
    test_target_data = test_target_data.rename(columns={"cl_id": "client_id", "target_flag": "bins"})

    # change transaction to numbers
    for dataset in [train_data, test_data]:
        keys = np.unique(dataset.small_group)
        new_values = np.arange(0, len(keys), dtype=int)
        dictionary = dict(zip(keys, new_values))
        new_column = [dictionary[key] for key in list(dataset.small_group)]
        dataset.small_group = new_column
    # change transaction to numbers
    # keys = np.unique(test_data.small_group)
    # new_values = np.arange(0, len(keys), dtype=int)
    # dictionary = dict(zip(keys, new_values))
    # new_column = [dictionary[key] for key in list(data.small_group)]
    # test_data.small_group = new_column

    train_target_data = train_target_data.dropna(subset=["bins"])
    test_target_data = test_target_data.dropna(subset=["bins"])

    print("Creating test set...")
    create_set("data/" + dataset_name + "/test.jsonl", test_data, test_target_data, period=True)
    print("")
    split_data("data/" + dataset_name, train_data, train_target_data, period=False)

    return


if __name__ == "__main__":

    typer.run(main)

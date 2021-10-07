import pandas as pd
import numpy as np
import json
import jsonlines
import typer

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_slice_subsample(sub_data, cnt_min, cnt_max, split_count):
    sub_datas = []
    for i in range(0, split_count):
        T_i = np.random.randint(cnt_min, cnt_max)
        s = np.random.randint(0, len(sub_data) - T_i - 1)
        S_i = sub_data[s : s + T_i - 1]
        sub_datas.append(S_i)

    return sub_datas


def create_set(name, data, target, period: bool = False):
    len_ = len(np.unique(target.client_id))
    dict_data = {}
    with jsonlines.open(name, "w") as writer:
        for client_id in tqdm(np.unique(target.client_id)):
            sub_data = data[data["client_id"] == client_id]
            sub_data_target = target[target["client_id"] == client_id]
            sub_datas = split_slice_subsample(sub_data, 25, 150, 30)
            for loc_data in sub_datas:
                if len(loc_data.small_group):
                    loc_dict = {
                        "transactions": list(loc_data.small_group),
                        "amounts": list(loc_data.amount_rur),
                        "label": int(sub_data_target.bins),
                        "client_id": int(client_id),
                    }
                    if period:
                        loc_dict["period"] = list(loc_data.trans_date)
                    writer.write(loc_dict)

    return


def split_data(dir_, data, target_data, period: bool = False):
    target_data_train, target_data_valid = train_test_split(target_data, test_size=0.2, random_state=10, shuffle=True)
    print("Create train set...")
    create_set(dir_ + "/" + "train.jsonl", data, target_data_train)
    print("Create valid set...")
    create_set(str(dir_) + "/" + "valid.jsonl", data, target_data_valid)
    return


def main(percentage: str):

    transactions = pd.read_csv("data/age/original/transactions_train.csv")
    target_data = pd.read_csv("data/age/original/train_target.csv")
    data = pd.merge(transactions, target_data, on="client_id")

    # leave out test set
    full_len = len(data)
    # splitting train and test by transaction time
    data = data.sort_values(by=["trans_date"])
    train_data = data[: int(full_len * int(percentage) / 100)]
    test_data = data[int(full_len * int(percentage) / 100) :]

    train_target_data = train_data[["client_id", "bins"]]
    train_target_data = train_target_data.drop_duplicates()
    train_target_data.reset_index(drop=True, inplace=True)
    test_target_data = test_data[["client_id", "bins"]]
    test_target_data = test_target_data.drop_duplicates()
    test_target_data.reset_index(drop=True, inplace=True)

    print("Creating test set...")
    create_set("data/age/test.jsonl", test_data, test_target_data, period=True)
    print("")
    split_data("data/age", train_data, train_target_data)

    return


if __name__ == "__main__":
    typer.run(main)

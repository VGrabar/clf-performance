import pandas as pd
import numpy as np
import json
import jsonlines
import typer

from tqdm import tqdm

from sklearn.model_selection import train_test_split


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
                        loc_dict["period"] = list(loc_data.transaction_month)
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
    dataset_name = "gender_tinkoff"
    transactions = pd.read_csv("./data/" + dataset_name + "/original/transactions.csv")
    target_data = pd.read_csv("./data/" + dataset_name + "/original/customer_train.csv")
    data = pd.merge(transactions, target_data, on="customer_id")

    data = data.sort_values(by=["transaction_month", "transaction_day"])
    data = data.rename(
        columns={
            "customer_id": "client_id",
            "merchant_mcc": "small_group",
            "transaction_amt": "amount_rur",
            "gender_cd": "bins",
        }
    )
    # change transaction to numbers
    keys = np.unique(data.small_group)
    new_values = np.arange(0, len(keys), dtype=int)
    dictionary = dict(zip(keys, new_values))
    new_column = [dictionary[key] for key in list(data.small_group)]
    data.small_group = new_column
    # recode gender to bins
    data = data.dropna(subset=["bins"])
    dict_gender = {"M": 0, "F": 1}
    bins_new = [dict_gender[key] for key in list(data.bins)]
    data.bins = bins_new
    # leave out test set
    full_len = len(data)
    # splitting train and test by transaction time
    train_data = data[: int(full_len * int(percentage) / 100)]
    test_data = data[int(full_len * int(percentage) / 100) :]

    train_target_data = train_data[["client_id", "bins"]]
    train_target_data = train_target_data.drop_duplicates()
    train_target_data.reset_index(drop=True, inplace=True)
    test_target_data = test_data[["client_id", "bins"]]
    test_target_data = test_target_data.drop_duplicates()
    test_target_data.reset_index(drop=True, inplace=True)

    #train_target_data = train_target_data.dropna(subset=["bins"])
    #test_target_data = train_target_data.dropna(subset=["bins"])

    print("Creating test set...")
    create_set("./data/" + dataset_name + "/test.jsonl", test_data, test_target_data, period=True)
    print("")
    split_data("./data/" + dataset_name, train_data, train_target_data)

    return


if __name__ == "__main__":
    typer.run(main)

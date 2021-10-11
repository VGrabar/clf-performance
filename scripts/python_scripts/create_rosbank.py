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


def create_set(name, data, target):
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
                    writer.write(loc_dict)

    return


def create_test_sets(folder_name, data, target):
    """
    Splits total data by periods
    """
    periods = np.unique(data.PERIOD)
    # sort periods by month-year
    periods.sort(key=lambda x: x.split("/")[2] + x.split("/")[1])
    print(periods)
    history = []
    for month in periods:
        print(month)
        day_month_year = month.split("/")
        month_name = day_month_year[2] + day_month_year[1]
        history.append(month)
        period_data = data[data["PERIOD"].isin(history)]

        dict_data = {}
        with jsonlines.open(folder_name + "/test_" + month_name + ".json", "w") as writer:
            for client_id in tqdm(np.unique(target.client_id)):
                sub_data = period_data[period_data["client_id"] == client_id]
                sub_data_target = target[target["client_id"] == client_id]
                if len(sub_data) > 3:
                    sub_datas = split_slice_subsample(sub_data, 25, 150, 30)
                    for loc_data in sub_datas:
                        if len(loc_data.small_group) > 3:
                            loc_dict = {
                                "transactions": list(loc_data.small_group),
                                "amounts": list(loc_data.amount_rur),
                                "label": int(sub_data_target.bins),
                                "client_id": int(client_id),
                            }
                            writer.write(loc_dict)

    return


def split_data(dir_, data, target_data):
    """
    Train-val split and saving as jsonlines
    """
    target_data_train, target_data_valid = train_test_split(target_data, test_size=0.2, random_state=10, shuffle=True)
    print("Creating train set...")
    create_set(dir_ + "/" + "train.jsonl", data, target_data_train)
    print("Creating valid set...")
    create_set(str(dir_) + "/" + "valid.jsonl", data, target_data_valid)
    return


def main(percentage: str):

    transactions = pd.read_csv("data/rosbank/original/train.csv")
    transactions = transactions.rename(
        columns={"cl_id": "client_id", "MCC": "small_group", "amount": "amount_rur", "target_flag": "bins"}
    )
    full_len = len(transactions)
    print(transactions["bins"].describe())
    # filter out observations
    transactions["PERIOD_DATETIME"] = pd.to_datetime(transactions["PERIOD"])
    transactions = transactions.sort_values(by=["PERIOD_DATETIME"])
    # change transaction to numbers
    keys = np.unique(transactions.small_group)
    new_values = np.arange(0, len(keys), dtype=int)
    dictionary = dict(zip(keys, new_values))
    new_column = [dictionary[key] for key in list(transactions.small_group)]
    transactions.small_group = new_column

    # splitting train and test by transaction time
    train_transactions = transactions[: int(full_len * int(percentage) / 100)]
    print(f"threshold period:{train_transactions.iloc[-1].PERIOD}")

    train_target_data = train_transactions[["client_id", "bins"]]
    train_target_data = train_target_data.drop_duplicates()
    train_target_data.reset_index(drop=True, inplace=True)
    train_target_data = train_target_data.dropna(subset=["bins"])

    plot_target_data = transactions[["client_id", "bins"]]
    plot_target_data = plot_target_data.drop_duplicates()
    plot_target_data.reset_index(drop=True, inplace=True)
    plot_target_data = plot_target_data.dropna(subset=["bins"])

    print("Creating test sets...")
    create_test_sets("data/rosbank/test", transactions, plot_target_data)
    print("Creating train-validations sets...")
    split_data("data/rosbank", train_transactions, train_target_data)

    return


if __name__ == "__main__":

    typer.run(main)

import argparse
import json
from datetime import datetime

import jsonlines
import os
import numpy as np
import pandas as pd
import typer
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


def write_data(jsonl_name, csv_name, period_data, target):
    """
    Writes sequences to jsonl with data and features to csv
    """
    with jsonlines.open(jsonl_name, "w") as writer:
        # features
        cl_ids = []
        bins_list = []
        min_amount = []
        max_amount = []
        med_amount = []
        avg_amount = []
        mode_mcc = []
        last_mcc = []
        for client_id in tqdm(np.unique(target.client_id)):
            sub_data = period_data[period_data["client_id"] == client_id]
            sub_data_target = target[target["client_id"] == client_id]

            if len(sub_data) > 3:
                # features
                cl_ids.append(int(client_id))
                bins_list.append(int(sub_data_target.bins))
                med_amount.append(int(sub_data.amount_rur.median()))
                avg_amount.append(float(sub_data.amount_rur.mean()))
                min_amount.append(int(sub_data.amount_rur.min()))
                max_amount.append(int(sub_data.amount_rur.max()))
                mode_mcc.append(int(sub_data.small_group.mode()[0]))
                last_mcc.append(int(sub_data.small_group.min()))
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

    df_data = {
        "amounts_max": max_amount,
        "amounts_min": min_amount,
        "amounts_med": med_amount,
        "amounts_avg": avg_amount,
        "mcc_mode": mode_mcc,
        "mcc_last": last_mcc,
        "label": bins_list,
        "client_id": cl_ids,
    }
    df = pd.DataFrame.from_dict(df_data, orient="columns")
    df.to_csv(csv_name)

    return


def create_all_sets(folder_name, data, target, threshold_period):
    """
    Splits total data by periods, accumulates it and save to separate files
    """
    periods = np.unique(data.PERIOD)
    # sort periods by month-year
    periods = sorted(periods, key=lambda x: x.split("/")[2] + x.split("/")[1])
    history = []
    for month in periods:
        print(month)
        day_month_year = month.split("/")
        month_name = day_month_year[2] + day_month_year[1]
        # cumulative history
        history.append(month)
        period_data = data[data["PERIOD"].isin(history)]
        test_jsonl_name = os.path.join(folder_name, "test", month_name + ".jsonl")
        test_csv_name = os.path.join(folder_name, "test", month_name + ".csv")
        write_data(test_jsonl_name, test_csv_name, period_data, target)

        if month == threshold_period:
            print(f"threshold period for train: {threshold_period}")
            print("Creating train-valid sets")
            target_data_train, target_data_valid = train_test_split(
                target, test_size=0.2, random_state=10, shuffle=True
            )
            train_jsonl_name = os.path.join(folder_name, "train.jsonl")
            train_csv_name = os.path.join(folder_name, "train.csv")
            valid_jsonl_name = os.path.join(folder_name, "valid.jsonl")
            valid_csv_name = os.path.join(folder_name, "valid.csv")
            test_jsonl_name = os.path.join(folder_name, "test.jsonl")
            test_csv_name = os.path.join(folder_name, "test.csv")
            # train
            write_data(train_jsonl_name, train_csv_name, period_data, target_data_train)
            # validation
            write_data(valid_jsonl_name, valid_csv_name, period_data, target_data_valid)
            # test = train and validation
            write_data(test_jsonl_name, test_csv_name, period_data, target)

    return


def main(percentage: str):

    transactions = pd.read_csv("data/rosbank/original/train.csv")
    transactions = transactions.rename(
        columns={"cl_id": "client_id", "MCC": "small_group", "amount": "amount_rur", "target_flag": "bins"}
    )
    full_len = len(transactions)

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
    threshold_period = train_transactions.iloc[-1].PERIOD

    target_data = transactions[["client_id", "bins"]]
    target_data = target_data.drop_duplicates()
    target_data.reset_index(drop=True, inplace=True)
    target_data = target_data.dropna(subset=["bins"])

    print("Creating test sets...")
    create_all_sets("data/rosbank", transactions, target_data, threshold_period)

    return


if __name__ == "__main__":

    typer.run(main)

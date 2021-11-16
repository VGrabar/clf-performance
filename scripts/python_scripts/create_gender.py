import json
import os

import jsonlines
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
        std_amount = []
        med_amount = []
        avg_amount = []
        perc_mcc = []
        for client_id in tqdm(np.unique(target.client_id)):
            sub_data = period_data[period_data["client_id"] == client_id]
            sub_data_target = target[target["client_id"] == client_id]

            if len(sub_data) > 3:
                # features
                cl_ids.append(int(client_id))
                bins_list.append(int(sub_data_target.bins))
                med_amount.append(int(sub_data.amount_rur.median()))
                avg_amount.append(float(sub_data.amount_rur.mean()))
                std_amount.append(int(sub_data.amount_rur.std()))
                individual_ratio = [0] * 500
                ind_sum = 0
                for mcc in sub_data.small_group:
                    ind_sum += 1
                    individual_ratio[int(mcc)] += 1
                individual_ratio = [i / ind_sum for i in individual_ratio]
                perc_mcc.append(individual_ratio)
                # sub_datas = split_slice_subsample(sub_data, 25, 150, 30)
                # for loc_data in sub_datas:
                #     if len(loc_data.small_group) > 3:
                #         loc_dict = {
                #             "transactions": list(loc_data.small_group),
                #             "amounts": list(loc_data.amount_rur),
                #             "label": int(sub_data_target.bins),
                #             "client_id": int(client_id),
                #         }
                #         writer.write(loc_dict)

    df_data = {
        "amounts_std": std_amount,
        "amounts_med": med_amount,
        "amounts_avg": avg_amount,
        "mcc_ratios": perc_mcc,
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
    # sort periods by day
    periods = sorted(periods, key=lambda x: int(x))
    history = []
    for per in periods:
        print(per)
        period_name = str(per)
        # cumulative history
        history.append(per)
        period_data = data[data["PERIOD"] == per]
        # period_data = data[data["PERIOD"].isin(history)]
        test_jsonl_name = os.path.join(folder_name, "test", period_name + ".jsonl")
        test_csv_name = os.path.join(folder_name, "test", period_name + ".csv")
        write_data(test_jsonl_name, test_csv_name, period_data, target)

        if per == threshold_period:
            print(f"threshold period for train: {threshold_period}")
            print("Creating train-valid sets")
            target_data_train, target_data_valid = train_test_split(
                target, test_size=0.2, random_state=10, shuffle=True, stratify=target["bins"]
            )
            print(target_data_train.describe())
            print(target_data_valid.describe())
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

    transactions = pd.read_csv("./data/gender/original/transactions.csv")
    target_data = pd.read_csv("./data/gender/original/gender_train.csv")
    print(transactions.head())
    transactions.term_id = transactions.term_id.fillna("UNK")
    transactions["trans"] = "mcc" + transactions["mcc_code"].astype(str)
    data = pd.merge(transactions, target_data, on="customer_id")
    data = data.rename(
        columns={
            "customer_id": "client_id",
            "trans": "small_group",
            "amount": "amount_rur",
            "gender": "bins",
            "tr_datetime": "PERIOD",
        }
    )
    full_len = len(data)
    # splitting train and test by transaction time
    data["PERIOD"] = data["PERIOD"].apply(lambda x: int(str(x).split(" ")[0]) // 30)
    print(np.unique(data["PERIOD"]))
    data = data.sort_values(by=["PERIOD"])

    train_data = data[: int(full_len * int(percentage) / 100)]
    threshold_period = train_data.iloc[-1].PERIOD

    # change transaction to numbers
    keys = np.unique(data.small_group)
    new_values = np.arange(0, len(keys), dtype=int)
    dictionary = dict(zip(keys, new_values))
    new_column = [dictionary[key] for key in list(data.small_group)]
    data.small_group = new_column

    target_data = data[["client_id", "bins"]]
    target_data = target_data.drop_duplicates()
    target_data.reset_index(drop=True, inplace=True)
    target_data = target_data.dropna(subset=["bins"])

    print("Creating test sets...")
    create_all_sets("data/gender", data, target_data, threshold_period)

    return


if __name__ == "__main__":
    typer.run(main)

import sys

sys.path.append("../../")
import numpy as np
import pandas as pd

from src.ad2s import AD2SDetector
from sklearn.metrics import roc_auc_score
from utils.util import cfg
from prettytable import PrettyTable


def cal(df):

    clf = AD2SDetector()
    clf_ab_chain = AD2SDetector(flag_additive_chain=False)
    clf_ab_iso = AD2SDetector(flag_iso_partition=False)
    clf_ab_chain_iso = AD2SDetector(
        flag_additive_chain=False, flag_iso_partition=False
    )

    predict = []
    predict_chain = []
    predict_iso = []
    predict_chain_iso = []

    for item in df["data"]:
        score = clf.predict(item)
        score_chain = clf_ab_chain.predict(item)
        score_iso = clf_ab_iso.predict(item)
        score_chain_iso = clf_ab_chain_iso.predict(item)

        predict.append(score)
        predict_chain.append(score_chain)
        predict_iso.append(score_iso)
        predict_chain_iso.append(score_chain_iso)

    df["predict"] = predict
    df["predict_chain"] = predict_chain
    df["predict_iso"] = predict_iso
    df["predict_chain_iso"] = predict_chain_iso

    truth = df["label"][df["data"] != 0.0]

    auc = roc_auc_score(truth, df["predict"][df["data"] != 0.0])
    auc_chain = roc_auc_score(truth, df["predict_chain"][df["data"] != 0.0])
    auc_iso = roc_auc_score(truth, df["predict_iso"][df["data"] != 0.0])
    auc_chain_iso = roc_auc_score(
        truth, df["predict_chain_iso"][df["data"] != 0.0]
    )
    # print(
    # f"AD2S auc: {auc_chain:10.3f}, {auc_iso:10.3f}, {auc_chain_iso:10.3f}"
    # )
    return round(auc_chain, 3), round(auc_iso, 3), round(auc_chain_iso, 3)


if __name__ == "__main__":

    table = PrettyTable()
    table.field_names = ["Fixed window", "Nonzero count", "Both"]

    for synthetic_ds in [1, 2, 3, 4]:
        cfg.data.synthetic_ds = synthetic_ds
        df = pd.read_csv(cfg.data.save_path)
        auc_chain, auc_iso, auc_chain_iso = cal(df)
        table.add_row([auc_chain, auc_iso, auc_chain_iso])

    print(table)

    with open(cfg.experiments.ablation.save_path, "w") as f:
        f.write(table.get_csv_string())

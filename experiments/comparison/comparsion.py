import sys

sys.path.append("../../")
import numpy as np
import pandas as pd
from baselines import (
    KNNDetector,
    RrcfDetector,
    RShashDetector,
    SvelteDetector,
    NBC,
    TranAD,
)
from src.ad2s import AD2SDetector
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import hydra

# from utils.util import cfg

# For TranAD model
import torch
from baselines.dlutils import (
    train,
    save_model,
    eval,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
from tdigest import TDigest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_knn(df,cfg):
    clf = KNNDetector()
    agg = cfg.experiments.comparison.downsample_granularity
    model_scores = []
    agg_data = 0

    for idx, data in enumerate(df["data"]):
        agg_data += data
        if idx != 0 and (idx + 1) % agg == 0:
            score = clf.fit_score(np.array([agg_data]))
            if score is None:
                flatten_score = [0] * agg
            else:
                flatten_score = [score] * agg
            model_scores.extend(flatten_score)
            agg_data = 0

    if len(model_scores) != len(df["data"]):
        model_scores.extend([0] * (len(df["data"]) - len(model_scores)))

    df["predict"] = model_scores
    truth = df["label"][df["data"] != 0.0]
    predict = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict)
    return round(auc, 3)


def cal_rrcf(df,cfg):
    clf = RrcfDetector()
    agg = cfg.experiments.comparison.downsample_granularity
    model_scores = []
    agg_data = 0

    for idx, data in enumerate(df["data"]):
        agg_data += data
        if idx != 0 and (idx + 1) % agg == 0:
            score = clf.fit_score(np.array([agg_data]))
            if score is None:
                flatten_score = [0] * agg
            else:
                flatten_score = [score] * agg
            model_scores.extend(flatten_score)
            agg_data = 0

    if len(model_scores) != len(df["data"]):
        model_scores.extend([0] * (len(df["data"]) - len(model_scores)))

    df["predict"] = model_scores
    truth = df["label"][df["data"] != 0.0]
    predict = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict)
    return round(auc, 3)


def cal_rshash(df,cfg):
    clf = RShashDetector()
    agg = cfg.experiments.comparison.downsample_granularity
    model_scores = []
    agg_data = 0

    for idx, data in enumerate(df["data"]):
        agg_data += data
        if idx != 0 and (idx + 1) % agg == 0:
            score = clf.fit_score(np.array([agg_data]))
            if score is None:
                flatten_score = [0] * agg
            else:
                flatten_score = [score] * agg
            model_scores.extend(flatten_score)
            agg_data = 0

    if len(model_scores) != len(df["data"]):
        model_scores.extend([0] * (len(df["data"]) - len(model_scores)))

    df["predict"] = model_scores
    truth = df["label"][df["data"] != 0.0]
    predict = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict)
    return round(auc, 3)


def cal_tdigest(df,cfg):
    clf = TDigest()
    agg = cfg.experiments.comparison.downsample_granularity
    model_scores = []
    agg_data = 0

    for idx, data in enumerate(df["data"]):
        agg_data += data
        if idx != 0 and (idx + 1) % agg == 0:
            clf.update(agg_data)
            score = clf.percentile(90)
            if score is None:
                flatten_score = [0] * agg
            else:
                flatten_score = [score] * agg
            model_scores.extend(flatten_score)
            agg_data = 0

    if len(model_scores) != len(df["data"]):
        model_scores.extend([0] * (len(df["data"]) - len(model_scores)))

    df["predict"] = model_scores
    truth = df["label"][df["data"] != 0.0]
    predict = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict)
    return round(auc, 3)


def cal_svelte(df,cfg):
    clf = SvelteDetector()

    model_scores = []
    for data in df["data"]:
        score = clf.score(data)
        model_scores.append(score)

    df["predict"] = model_scores
    truth = df["label"][df["data"] != 0.0]
    predict_nonzero = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict_nonzero)
    return round(auc, 3)


def cal_nbc(df,cfg):
    # Prepare: data intervals as inputs
    X_train, X_test, y_train, y_test = train_test_split(
        df["data"], df["label"], test_size=0.7, random_state=0, shuffle=False
    )
    X_gap_train = []
    X_gap_test = []
    y_gap_train = []
    y_gap_test = []

    cnt_zero = 0
    for x, y in zip(X_train, y_train):
        if x == 0:
            cnt_zero += 1
        elif x >= 1:
            X_gap_train.append(cnt_zero)
            y_gap_train.append(y)
            cnt_zero = 0

    cnt_zero = 0
    for x, y in zip(X_test, y_test):
        if x == 0:
            cnt_zero += 1
        elif x >= 1:
            X_gap_test.append(cnt_zero)
            y_gap_test.append(y)
            cnt_zero = 0

    model = NBC()
    model.fit(X_gap_train, y_gap_train)
    model_scores = []
    for object in X_gap_test:
        score = model.score(object)
        model_scores.append(score)

    auc = roc_auc_score(y_gap_test, np.array(model_scores).ravel())
    return round(auc, 3)


def cal_tranad(df,cfg):
    # Prepare: data intervals as inputs
    X_train, X_test, y_train, y_test = train_test_split(
        df["data"], df["label"], test_size=0.7, random_state=0, shuffle=False
    )
    X_gap_train = []
    X_gap_test = []
    y_gap_train = []
    y_gap_test = []

    cnt_zero = 0
    for x, y in zip(X_train, y_train):
        if x == 0:
            cnt_zero += 1
        elif x >= 1:
            X_gap_train.append(cnt_zero)
            y_gap_train.append(y)
            cnt_zero = 0

    cnt_zero = 0
    for x, y in zip(X_test, y_test):
        if x == 0:
            cnt_zero += 1
        elif x >= 1:
            X_gap_test.append(cnt_zero)
            y_gap_test.append(y)
            cnt_zero = 0

    seq_length = 5
    batch_size = 16
    lrs_step_size = 5000

    def frame_series(X, y=None):
        nb_obs = len(X)
        nb_featurs = 1
        features = []

        for i in range(0, nb_obs - seq_length + 1):
            features.append(
                torch.DoubleTensor(X[i : i + seq_length]).unsqueeze(0)
            )

        features_var = torch.cat(features)

        return TensorDataset(features_var, features_var)

    def run_train(
        model, train_iter, test_iter, optimizer, criterion, scheduler
    ):

        num_epochs = 100
        global_step = 0
        train_loss = 0.0

        for e in range(0, num_epochs):
            for i, batch in enumerate(train_iter):

                optimizer.zero_grad()

                loss = train(model=model, criterion=criterion, batch=batch)
                train_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()

                global_step += 1

        # save_model(model, optimizer, scheduler)
        return

    def run_predict(model, X, criterion):
        with torch.no_grad():
            loss, x_hat, target = eval(
                model=model, criterion=criterion, batch=(X, X)
            )
            loss = loss.item()

        return loss, x_hat, target

    X_gap_train = np.array(X_gap_train).reshape(-1, 1)
    X_gap_test = np.array(X_gap_test).reshape(-1, 1)
    X_gap_train_tensor = frame_series(X_gap_train)
    X_gap_test_tensor = frame_series(X_gap_test)

    train_iter = DataLoader(
        X_gap_train_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )
    test_iter = DataLoader(
        X_gap_test_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )

    model = TranAD(feats=1, n_window=5).double().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lrs_step_size, 0.6)

    criterion = nn.MSELoss()
    run_train(
        model=model,
        train_iter=train_iter,
        test_iter=test_iter,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )

    model.eval()
    rec_res = []
    for i, batch in enumerate(test_iter):
        loss, x_hat, target = run_predict(
            model=model, X=batch[0], criterion=criterion
        )
        rec = abs(target - x_hat)
        if i == 0:
            for idx, x in enumerate(rec):
                if idx == 0:
                    rec_res = x.tolist()
                else:
                    rec_res.append(x[-1].tolist())
        else:
            for idx, x in enumerate(rec):
                rec_res.append(x[-1].tolist())

    auc = roc_auc_score(y_gap_test, np.array(rec_res).ravel())
    return round(auc, 3)


def cal_ad2s(df,cfg):
    clf = AD2SDetector()
    predict = []
    for data in df["data"]:
        score = clf.predict(data)
        predict.append(score)

    df["predict"] = predict
    truth = df["label"][df["data"] != 0.0]
    predict_nonzero = df["predict"][df["data"] != 0.0]
    auc = roc_auc_score(truth, predict_nonzero)
    return round(auc, 3)


@hydra.main(config_path="../../", config_name="config", version_base="1.3")
def main(cfg):
    # Load data
    df = pd.read_csv(cfg.data.save_path)

    if cfg.experiments.comparison.model == "KNNDetector":
        auc = cal_knn(df,cfg)
    elif cfg.experiments.comparison.model == "RrcfDetector":
        auc = cal_rrcf(df,cfg)
    elif cfg.experiments.comparison.model == "RShashDetector":
        auc = cal_rshash(df,cfg)
    elif cfg.experiments.comparison.model == "TDigest":
        auc = cal_tdigest(df,cfg)
    elif cfg.experiments.comparison.model == "SvelteDetector":
        auc = cal_svelte(df,cfg)
    elif cfg.experiments.comparison.model == "NBC":
        auc = cal_nbc(df,cfg)
    elif cfg.experiments.comparison.model == "TranAD":
        auc = cal_tranad(df,cfg)
    elif cfg.experiments.comparison.model == "AD2S":
        auc = cal_ad2s(df,cfg)

    auc = round(
        max(auc, 1 - auc), 3
    )  # Different methods may define anomaly as 0 or 1
    print(f"{cfg.experiments.comparison.model} auc: {auc}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import hydra


def clean_data(data:np.ndarray, gap:float) -> np.ndarray:
    """ Optional function to clean the data

    Args:
        data (np.ndarray): generate data for observations
        gap (float): Minimum gap between two observations

    Returns:
        np.ndarray: Cleaned data for observations
    """

    last_idx = 0
    cleaned_data = []
    for idx, item in enumerate(data):
        if item == 1:
            if idx - last_idx > gap:
                cleaned_data.append(item)
                last_idx = idx
            else:
                cleaned_data.append(0)
        else:
            cleaned_data.append(item)

    return cleaned_data


def generate_ds(
    l_normal: list = [10000],
    p_normal: list = [0.002],
    l_anomaly: list = [50],
    p_anomaly: list = [0.6],
    seed=0,
    method="binomial",
) -> pd.DataFrame:
    """Generate a dataset with normal and anomaly data

    Args:
        l_normal (list, optional): Length of normal data. Defaults to [10000].
        p_normal (list, optional): Intensities of normal data. Defaults to [0.002].
        l_anomaly (list, optional): Length of anomalies. Defaults to [50].
        p_anomaly (list, optional): Intensities of anomalies . Defaults to [0.6].
        seed (int, optional): Random seed . Defaults to 0.
        method (str, optional): Generate method. Defaults to "binomial".

    Returns:
        pd.DataFrame: Dataframe of the generated dataset
    """
    rng = np.random.default_rng(seed)

    vals = []

    if method == "binomial":
        for l_n, p_n in zip(l_normal, p_normal):
            new_data = rng.binomial(1, p_n, l_n)
            # new_data = clean_data(new_data, 1 / p_n)
            vals.extend(new_data)
    elif method == "uniform":
        for l_n, p_n in zip(l_normal, p_normal):
            gap = int(1 / p_n)
            tmp = [1] + [0] * (gap - 1)
            vals.extend(tmp * int(l_n / gap))

        if len(vals) < sum(l_normal):
            vals.extend([0] * (sum(l_normal) - len(vals)))

    elif method == "random":
        p_normal = rng.uniform(p_normal[0], p_normal[1], p_normal[2])
        for l_n, p_n in zip(l_normal, p_normal):
            vals.extend(rng.binomial(1, p_n, l_n))

    df = pd.DataFrame(vals, columns=["data"])
    df["label"] = False

    for l_a, p_a in zip(l_anomaly, p_anomaly):
        vals_anomaly = rng.binomial(1, p_a, l_a)
        pos = rng.integers(sum(l_normal) - l_a)
        df.loc[pos : pos + l_a - 1, "data"] = vals_anomaly
        df.loc[pos : pos + l_a - 1, "label"] = [True] * l_a
    df.loc[df["data"] == 0, "label"] = False

    df.iloc[:50, df.columns.get_loc("data")] = 0
    df.iloc[:50, df.columns.get_loc("label")] = False

    return df


def plot_ds(df:pd.DataFrame):
    """Plot the generated ds

    Args:
        df (pd.DataFrame): Dataframe of the observations

    Returns:
        plot: Plot figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["data"],
            mode="lines+markers",
            selectedpoints=np.where(df["label"] == True)[0],
            selected=dict(marker=dict(color="red", size=5)),
            unselected=dict(marker=dict(size=0)),
        )
    )
    return fig


@hydra.main(config_path="../", config_name="config",version_base="1.3")
def main(cfg):
    if os.path.exists(cfg.data.save_path):
        raise FileExistsError("Data already exists. Please delete it first.")
    else:
        if cfg.data.synthetic_ds == 1:
            df = generate_ds(
                l_normal=[100_000],
                p_normal=[0.002],
                l_anomaly=[100] * 10,
                p_anomaly=[0.2] * 10,
                seed=0,
            )
        elif cfg.data.synthetic_ds == 2:
            df = generate_ds(
                l_normal=[100_000],
                p_normal=[0.01],
                l_anomaly=[500] * 10,
                p_anomaly=[0.05] * 10,
                seed=0,
            )
        elif cfg.data.synthetic_ds == 3:
            df = generate_ds(
                l_normal=[10_000] * 10,
                p_normal=[0.001, 0.01] * 5,
                l_anomaly=[500] * 10,
                p_anomaly=[0.1] * 10,
            )
        elif cfg.data.synthetic_ds == 4:
            df = generate_ds(
                l_normal=[25994 * 2] * 5,
                p_normal=[0.0001, 0.001] * 5,
                l_anomaly=[800, 200] * 10,
                p_anomaly=[0.03, 0.2] * 10,
                seed=0,
            )
            df.loc[
                df[df["data"] == 1].sample(frac=0.005, random_state=0).index,
                "data",
            ] = 2  # This line is for dataset4 only
        df.to_csv(cfg.data.save_path, index=False)


if __name__ == "__main__":
    main()
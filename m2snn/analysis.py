import numpy as np
from itertools import product

from m2snn.snn import run_snn, masked_inference
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n
import matplotlib.pyplot as plt

from seaborn import violinplot, swarmplot
import itertools
import seaborn as sns

import torch


def compute_additivity(act_profiles):
    assert not ((act_profiles is None)), "Provide activity profiles !"

    dir_pairs = np.array(list(product([-1, 0, 1], repeat=2)))

    if not isinstance(act_profiles, list):

        adds = np.array(
            [
                (act_profiles[:, (dir_pairs == [d, d]).all(-1)]).squeeze()
                / ((act_profiles[:, dir_pairs.sum(-1) == d]).sum(-1))
                for d in [-1, 1]
            ]
        )
        # return adds
        try:
            max_value = adds[np.isfinite(adds)].max() * 1.1
        except ValueError:
            max_value = 0

        adds = np.stack(adds)
        return (
            np.nan_to_num(adds, nan=0, posinf=max_value).max(0),
            max_value,
        )

    else:
        return [compute_additivity(a) for a in act_profiles]


def additivity_from_spks(
    network,
    y_local,
    spks,
    plot=False,
    window_length=None,
):
    palette = itertools.cycle(sns.color_palette("colorblind"))

    try:
        spks = spks.cpu().detach().numpy()
    except AttributeError:
        pass

    try:
        y_local = y_local.cpu().detach().numpy()
    except AttributeError:
        pass

    dir_pairs = np.array(list(product([-1, 0, 1], repeat=2)))

    if window_length is not None:
        y_local = y_local[:, window_length - 1 :: window_length]
        spks = spks[:, window_length - 1 :: window_length]

    dir_idxs = {tuple(d): (y_local == d).all(-1) for d in dir_pairs}
    # print(spks.shape, dir_idxs[1, 1].shape)
    act_profiles = np.array(
        [(spks[idx].mean(0))[network.nb_arc == 2] for idx in dir_idxs.values()]
    )

    act_profiles = np.nan_to_num(act_profiles, nan=0)
    spiking_units = ~(act_profiles == 0).all(0)
    spiking_act_profiles = act_profiles[:, spiking_units].T

    if spiking_units.sum() != 0:
        add_metric, max_value = compute_additivity(act_profiles=act_profiles.T)

        pure_units = add_metric == max_value
        add_idxs = [
            ~spiking_units,
            spiking_units * pure_units,
            spiking_units * (~pure_units),
        ]
        add_labels = ["non spiking", "pure coincident", "spiking"]
        if plot:
            fig, axs = plt.subplots(
                1,
                2,
                figsize=(10, 3),
                constrained_layout=True,
            )
            [
                swarmplot(
                    x=np.ones_like(add_metric[idx]),
                    y=add_metric[idx],
                    label=l,
                    ax=axs[0],
                    color=next(palette),
                )
                for idx, l in zip(add_idxs, add_labels)
            ]
            violinplot(x=np.ones_like(add_metric), y=add_metric, ax=axs[1])

            additivity_img = get_img_from_fig(fig)
        else:
            additivity_img = None
    else:
        pure_units = spiking_units
        additivity_img = add_metric = None

    return {
        # "act_profiles_img": act_profiles_img,
        # "additivity_img": additivity_img,
        "act_profiles": act_profiles,
        "add_metric": add_metric,
        "pure_units": pure_units,
        "spiking_units": spiking_units,
        # "dir_idxs": dir_idxs,
    }


def compute_topk_ablations(
    network,
    metrics,
    n_ablations,
    batchs,
    metrics_name,
    loader=None,
    max_samples=1,
    position=0,
    top_bottom=True,
    device=torch.device("cuda"),
):
    all_accs = {k: [] for k in ["metric", "n_ablated", "position", "accuracy"]}
    metrics_name = (
        [f"metric_{i}" for i, _ in enumerate(metrics)]
        if metrics_name is None
        else metrics_name
    )
    for metric, name in zip(metrics, metrics_name):
        for n in n_ablations:
            for i, key in zip([-1, 1] if top_bottom else [1], ["top", "bottom"]):
                excluded = np.argsort(metric)[::i][:n]
                all_accs["metric"].append(name)
                all_accs["n_ablated"].append(n)
                all_accs["position"].append(key)

                # print(excluded)
                all_accs["accuracy"].append(
                    masked_inference(
                        network,
                        excluded,
                        batchs=batchs,
                        loader=loader,
                        max_samples=max_samples,
                        device=device,
                    )
                )

    return all_accs

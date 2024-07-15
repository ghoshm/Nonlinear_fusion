from unittest import loader
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from IPython.display import display, clear_output
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n
import copy
import matplotlib.pyplot as plt

import warnings

from m2snn.snn import build_network, run_snn, ZeroOneClipper
from m2snn.analysis import additivity_from_spks

warnings.filterwarnings("ignore")


def is_notebook():
    try:
        get_ipython()
        notebook = True
    except NameError:
        notebook = False
    return notebook


def train(
    network,
    optimizer,
    loaders,
    config,
    use_tqdm=True,
    train=True,
    nb_epochs=None,
    device=torch.device("cuda"),
    profiler=None,
    stream=None,
    scheduler=None,
):
    if str(device) == "cuda":
        if stream:
            torch.cuda.set_stream(stream)

    detection_task = len(loaders["Train"].dataset.data["y_data"].unique()) > 2

    clipper = ZeroOneClipper()

    tqdm_f = tqdm_n if is_notebook() else tqdm

    nb_arc = network.nb_arc
    adjust_loss = config.get("adjust_loss", False)
    gather_spks = config.get("gather_spks", True)
    gather_all_epochs = config.get("gather_all_epochs", False)

    if adjust_loss and detection_task:
        weights = torch.tensor([1, adjust_loss, 1]).float()
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    log_softmax_fn = nn.LogSoftmax(dim=1)

    loss_fns = (log_softmax_fn, loss_fn)

    loss_hist = []
    accs_hist = []
    reward_hist = []

    test_accs_hist = []
    test_loss_hist = []
    test_reward_hist = []

    all_test_accs = []
    all_decisions = (
        {"Test": [], "Stim-Space": []}
        if config.get("stim-space", False)
        else {"Test": []}
    )
    all_spks = (
        {"Test": [], "Stim-Space": []}
        if config.get("stim-space", False)
        else {"Test": []}
    )

    firing_rates = {
        i: {"avg": [], "std": [], "all": [], "non_spiking": []}
        for i in np.unique(nb_arc)
    }

    additivity = {}

    bs = len(loaders["Train"].dataset) // len(loaders["Train"])

    best_loss = 1e10
    nb_epochs = config["nb_epochs"] if nb_epochs is None else nb_epochs

    descs = np.full((2), "", dtype=object)

    max_steps = config.get("max_steps", None)

    if max_steps is None:
        max_steps = loaders["Train"].dataset.data["y_local"].shape[1]

    pbar = range(nb_epochs + 1)

    if use_tqdm:
        position = use_tqdm if isinstance(use_tqdm, int) else 0
        pbar = tqdm_f(pbar, position=position, leave=None, desc="Epochs : ")

    for e in pbar:
        if train and e > 0:
            # Training
            local_loss = []
            local_acc = []
            local_reward = []

            for batch_idx, (x_local, y_local) in enumerate(loaders["Train"]):
                x_local, y_local = x_local[:, :max_steps].to(device), y_local.to(device)

                if detection_task:
                    y_local += 1

                output, spks, _ = run_snn(network, x_local)
                out_sum = torch.sum(output, 1)  # sum over time

                # Add small random values if equal
                eq = torch.where(
                    (out_sum[:, 0] == out_sum[:, 1])
                )  # [2 outputs] find equal sums
                out_sum[eq] += torch.rand((out_sum[eq].shape)).to(device) / 1000
                _, am = torch.max(out_sum, 1)  # argmax over output units
                acc = np.mean(
                    (y_local == am).detach().cpu().numpy()
                )  # compare to labels

                log_p_y = log_softmax_fn(out_sum)

                # Combine supervised loss
                loss_val = loss_fn(log_p_y, y_local)

                # Spiking regularizer
                reg_loss = 0
                spks_pn = torch.sum(torch.sum(spks, dim=0), dim=0)  # spikes per neuron

                if config.get("reg_factor", 0):
                    reg_loss += config["reg_factor"] * torch.sum(
                        torch.stack(
                            [
                                (spks_pn**2)[nb_arc == i].mean()
                                for i in np.unique(nb_arc)
                            ]
                        )
                    )  # L2 per area

                loss_final = loss_val + reg_loss

                optimizer.zero_grad()
                loss_final.backward()

                optimizer.step()
                network.apply(clipper)
                if profiler:
                    profiler.step()

                local_loss.append(loss_val.item())
                local_acc.append(acc)

                descs[0] = str(
                    "Train Epoch: {} ({}/{}) | Loss: {:.3f}, Accuracy: {}%".format(
                        e,
                        batch_idx * bs,
                        len(loaders["Train"].dataset),
                        loss_val.item(),
                        (
                            (np.round(100 * a) for a in acc)
                            if type(acc) is list
                            else np.round(100 * acc)
                        ),
                    )
                )

                if use_tqdm:
                    pbar.set_description((descs.sum()))

            mean_loss, mean_acc, mean_reward = (
                np.mean(local_loss),
                np.mean(local_acc),
                np.mean(local_reward),
            )

            loss_hist.append(mean_loss)
            accs_hist.append(mean_acc)
            reward_hist.append(mean_reward)

        if scheduler is not None and e > 0:
            scheduler.step()

        # ------ Testing ------

        test_dict = compute_testing_accs(
            network,
            loaders["Test"],
            config,
            loss_fns,
            gather_spks,
            device=device,
        )

        if test_dict["mean_loss"] < best_loss:
            best_network = copy.deepcopy(network)

        test_loss_hist.append(test_dict["mean_loss"])
        test_accs_hist.append(test_dict["mean_acc"])
        test_reward_hist.append(test_dict["mean_reward"])

        for i in np.unique(nb_arc):
            f_rates = test_dict["mean_firing_rate"][nb_arc == i]
            firing_rates[i]["all"].append(f_rates)
            firing_rates[i]["avg"].append(f_rates[f_rates != 0].mean())
            firing_rates[i]["std"].append(f_rates[f_rates != 0].std())
            firing_rates[i]["non_spiking"].append((f_rates == 0).mean())

        descs[1] = str(
            " | Test Loss: {:.3f}, Accuracy: {}%".format(
                test_dict["mean_loss"].item(),
                (
                    (np.round(100 * a) for a in test_dict["mean_acc"])
                    if type(test_dict["mean_acc"]) is list
                    else np.round(100 * test_dict["mean_acc"])
                ),
            )
        )

        if use_tqdm:
            pbar.set_description((descs.sum()))

        # ------ Testing on Stimulus Space ------

        if config.get("stim-space", False):
            stim_dict = compute_testing_accs(
                network,
                loaders["Stim-Space"],
                config,
                loss_fns,
                gather_spks,
                device=device,
            )

        else:
            stim_dict = {"all_spks": None, "decisions": None, "all_accs": None}

        for origin, result_dict in zip(
            ["Test", "Stim-Space"] if config.get("stim-space", False) else ["Test"],
            [test_dict, stim_dict],
        ):
            if e <= 1:
                all_spks[origin].append(result_dict["all_spks"].cpu().data.numpy())
                all_decisions[origin].append(result_dict["decisions"])
            elif gather_all_epochs:
                all_spks[origin].append(result_dict["all_spks"].cpu().data.numpy())
                all_decisions[origin].append(result_dict["decisions"])
            else:
                all_spks[origin][-1] = result_dict["all_spks"].cpu().data.numpy()
                all_decisions[origin][-1] = result_dict["decisions"]

            additivity.setdefault(origin, {})
            spks, y_local = (
                all_spks[origin][-1],
                loaders[origin].dataset.data["y_local"],
            )

            # additivity_metric_dict = additivity_from_spks(
            #     network,
            #     y_local,
            #     spks,
            # )
            # for n, v in additivity_metric_dict.items():
            #     additivity[origin].setdefault(n, [])
            #     additivity[origin][n].append(v)

        all_test_accs.append(test_dict["all_accs"])

        if config.get("stopping_acc", None):
            if test_dict["mean_acc"] > config["stopping_acc"] and (
                config["min_epochs"] is None or e > config["min_epochs"]
            ):
                return {
                    "train_loss": np.stack(loss_hist),
                    "train_accs": np.stack(accs_hist),
                    "train_rewards": np.stack(reward_hist),
                    "test_accs": np.stack(test_accs_hist),
                    "test_rewards": np.stack(test_reward_hist),
                    "best_network": best_network,
                    "last_network": network,
                    "all_accs": np.stack(all_test_accs),
                    "all_decisions": {k: np.stack(v) for k, v in all_decisions.items()},
                    "all_spks": {k: np.stack(v) for k, v in all_spks.items()},
                    "firing_rates": {
                        i: {k: np.stack(v) for k, v in f_rates.items()}
                        for i, f_rates in firing_rates.items()
                    },
                    "additivity": additivity,
                }

    return {
        "train_loss": np.stack(loss_hist),
        "train_accs": np.stack(accs_hist),
        "train_rewards": np.stack(reward_hist),
        "test_accs": np.stack(test_accs_hist),
        "test_rewards": np.stack(test_reward_hist),
        "best_network": best_network,
        "last_network": network,
        "all_accs": np.stack(all_test_accs),
        "all_decisions": {k: np.stack(v) for k, v in all_decisions.items()},
        "all_spks": {k: np.stack(v) for k, v in all_spks.items()},
        "firing_rates": {
            i: {k: np.stack(v) for k, v in f_rates.items()}
            for i, f_rates in firing_rates.items()
        },
        "additivity": additivity,
    }


def compute_testing_accs(
    network,
    loader,
    config,
    loss_fns,
    gather_spks=False,
    device=torch.device("cuda"),
):
    log_softmax_fn, loss_fn = loss_fns

    # Testing

    local_loss = []
    local_accs = []
    local_dec = []
    all_spks = []
    local_reward = []
    avg_firing_rates = []

    # device = config["device"]

    detection_task = len(loader.dataset.data["y_data"].unique()) > 2
    max_steps = config.get("max_steps", None)
    if max_steps is None:
        max_steps = loader.dataset.data["y_local"].shape[1]
    with torch.no_grad():
        for batch_idx, (x_local, y_local) in enumerate(loader):
            x_local, y_local = x_local[:, :max_steps].to(device), y_local.to(device)

            if detection_task:
                y_local += 1

            output, spks, _ = run_snn(network, x_local)
            out_sum = output.cumsum(1)  # sum over time

            if gather_spks:
                all_spks.append(spks.cpu().data)

            # Add small random values if equal
            eq = torch.where(
                (out_sum[..., 0] == out_sum[..., 1])
            )  # [2 outputs] find equal sums
            out_sum[eq] += torch.rand((out_sum[eq].shape)).to(device) / 1000
            _, am = torch.max(out_sum, -1)  # argmax over output units
            accs = (
                (y_local.unsqueeze(1).expand_as(am) == am).detach().cpu().numpy()
            )  # compare to labels
            decison = am.detach().cpu().numpy()

            # careful !! decision and acc are per timestep
            log_p_y = log_softmax_fn(out_sum[:, -1])

            loss_val = loss_fn(log_p_y, y_local)

            local_loss.append(loss_val.item())
            local_accs.append(accs)
            local_dec.append(decison)

            avg_firing_rates.append(spks.mean(0).sum(0).cpu().data.numpy())

        mean_loss, mean_acc = np.mean(local_loss), np.mean(
            [a[:, -1].mean() for a in local_accs]
        )
        mean_reward = np.mean(local_reward)
        mean_firing_rate = np.mean(avg_firing_rates, 0)
        all_accs = np.concatenate(local_accs)
        decisions = np.concatenate(local_dec)
        if gather_spks:
            all_spks = torch.cat(all_spks)

    return {
        "mean_loss": mean_loss,
        "mean_acc": mean_acc,
        "all_accs": all_accs,
        "decisions": decisions,
        "all_spks": all_spks,
        "mean_reward": mean_reward,
        "mean_firing_rate": mean_firing_rate,  # n_neurons
    }


def mkdir_or_save_torch(to_save, save_name, save_path):
    path = Path(save_path)
    path.mkdir(exist_ok=True)
    torch.save(to_save, save_path + save_name)

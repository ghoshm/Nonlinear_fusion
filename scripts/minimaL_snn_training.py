# Import the task(s) you want to use.
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from m2snn.multisensory import (
    DetectionTask,
    ClassicalTask,
    ComodulationTask,
    BalancedComodulationTask,
)
from m2snn.spikes import SpikingMultimodal
from m2snn.snn import build_network
from m2snn.training import train

if __name__ == "__main__":

    # Set task parameters - note that these differ per task.

    # task = DetectionTask(pm=2 / 3, pe=0.057, pc=0.95, pn=1 / 3, pi=0.01)
    # task = ClassicalTask(s=0.1)
    task = BalancedComodulationTask(s=0.5)

    # Generate spikes from the trials.
    bs = 128
    n_trials = [20000, 3000]
    nb_steps = 30

    datasets = [
        SpikingMultimodal(
            task,
            data_size=n,
            nb_inputs=14 * 14,
            nb_windows=nb_steps,
            window_length=3,
        )
        for n in n_trials
    ]

    dataloaders = {
        k: torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=k == "Train")
        for k, dataset in zip(["Train", "Test"], datasets)
    }

    # Define the model.

    model_config = {
        "nb_inputs": 14 * 14,
        "nb_hiddens": [30, 15],
        "nb_outputs": 3,
        "weight_scale": 0.15,
        "device": "cpu",
        "feedforward": True,
        "tau_m": 5e-3,
        "tau_s": 3e-3,
        "het_ts": True,
    }

    networks = {
        "multi": build_network(**{"arch_type": "multi", **model_config}),
        "uni": build_network(**{"arch_type": "uni", **model_config}),
        "double_uni": build_network(**{"arch_type": "double_uni", **model_config}),
    }

    # Train the models.
    training_config = {
        "nb_epochs": 30,
        "batch_size": bs,
        "lr": 1e-3,
        "gamma": 0.95,
        "reg_factor": 0,  # 1e-6,
        "stopping_acc": 0.95,
        "min_epochs": 3,
    }

    optimizers = {
        k: torch.optim.Adam(v.parameters(), lr=training_config["lr"])
        for k, v in networks.items()
    }

    results = {}

    pbar = tqdm(range(len(networks)), desc="Architecture", position=0, leave=True)
    for optimizer, (net_type, network), _ in zip(
        optimizers.values(), networks.items(), pbar
    ):

        if training_config["gamma"]:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=training_config["gamma"]
            )
        else:
            scheduler = None

        train_results = train(
            network,
            optimizer,
            dataloaders,
            training_config,
            use_tqdm=1,
            device="cpu",
            scheduler=scheduler,
        )
        results[net_type] = train_results

    # Save the results.
    torch.save(results, "snn_results/results")

    # Plot the results.
    df_results = {"test_accs": [], "arch": [], "epochs": []}
    for arch, res in results.items():
        df_results["test_accs"].extend(res["test_accs"].tolist())
        df_results["arch"].extend([arch] * len(res["test_accs"]))
        df_results["epochs"].extend(list(range(len(res["test_accs"]))))

    df_results = pd.DataFrame(df_results)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_results, x="epochs", y="test_accs", hue="arch", ax=ax, palette="viridis"
    )
    plt.savefig("snn_results/results.png")
    plt.show()

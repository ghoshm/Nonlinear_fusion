from typing import Any
import numpy as np
import torch

from torch.utils.data import Dataset


def mv_grt(rng, y_data, nb_inputs=14 * 14, density=0.88, coh=1, flkr=0.04):

    if rng is None:
        rng = np.random.default_rng()
    data_size, nb_steps = y_data.shape[:2]

    y_data = y_data.T

    coh = np.full(data_size, coh)

    # Pre-allocate

    i_inputs = (np.sqrt(nb_inputs).astype(int), np.sqrt(nb_inputs).astype(int))
    stim_len = i_inputs[0]  # bar length

    stim_n = 1  # number of bars
    x_data = np.zeros((data_size, nb_steps, nb_inputs))  # samples x time x inputs

    # Generate data

    x_sample = torch.zeros((data_size, nb_steps, *i_inputs))

    # Starting pos of bars, for every sample
    x_1 = torch.randint(0, i_inputs[1] - ((stim_n - 1) * 2), [data_size])

    # postions per timesteps
    x_1 = (torch.tensor(y_data).cumsum(0) + x_1).T % i_inputs[0]

    # create full bars
    x_sample.scatter_(
        -1, x_1.unsqueeze(-1).expand(x_sample.shape[:-1]).unsqueeze(-1), 1
    )

    # coherence
    x_mask = rng.binomial(
        1, np.maximum(coh, flkr), (nb_steps, i_inputs[0], i_inputs[1], data_size)
    ).transpose(-1, 0, 1, 2)

    x_sample *= x_mask

    # density
    x_mask = rng.random(x_sample.shape) < density
    x_sample *= x_mask
    x_data = x_sample.flatten(start_dim=-2)

    return x_data, y_data


def lat_s(rng, y_data, nb_inputs=14 * 14, density=0.12, flkr=0.01):

    if rng is None:
        rng = np.random.default_rng()

    data_size, nb_steps = y_data.shape[:2]

    i_inputs = np.sqrt(nb_inputs).astype(int)

    # Pre-allocate
    x_data = np.zeros((data_size, nb_steps, nb_inputs))  # samples x time x inputs
    h_inputs = i_inputs // 2

    # Generate data
    # data_size x nb_steps x channels
    p = np.full((data_size, nb_steps, 2), flkr)

    mask_0, mask_R, mask_L = [(y_data == d) for d in [0, 1, -1]]

    p[mask_L, 0] = density
    p[mask_R, 1] = density

    x_tmp = [[]] * 2
    for ch in [0, 1]:
        x_tmp[ch] = np.zeros((data_size, nb_steps, i_inputs, h_inputs))

        spike_probs = p[..., ch]
        spike_probs = (
            np.expand_dims(spike_probs, (-1, -2))
            .repeat(i_inputs, -2)
            .repeat(h_inputs, -1)
        )

        x_tmp[ch] = rng.binomial(1, spike_probs, x_tmp[ch].shape)

    x_data = np.concatenate(x_tmp, -1).reshape((data_size, nb_steps, -1))

    return x_data, y_data.T


def generate_spikes(
    task,
    data_size=10,
    nb_windows=30,
    window_length=3,
    nb_inputs=14 * 14,
    unimodal=False,
    seed=42,
):
    """
    Generate spike data for a given task.

    Args:
        task: The task object that generates abstract trials.
        data_size (int): The number of trials to generate.
        nb_windows (int): The number of windows per trial.
        window_length (int): The length (in ts) of each window.
        nb_inputs (int): The number of input neurons.
        unimodal (bool): Whether to generate unimodal spikes.
        seed (int): The random seed for spike generation.

    Returns:
        dict: A dictionary containing the generated spike data.
            - "x_data" (torch.Tensor): The input spike data.
            - "y_data" (torch.Tensor): The target spike data.
            - "y_local" (torch.Tensor): The local spike data.
    """

    # Get Directions

    assert task is not None, "Task must be provided"

    trials = task.generate_trials(data_size, nb_windows)
    y_data, y_audio, y_video = trials.M, trials.A, trials.V

    if window_length:
        # print(f"Window length: {window_length}, Shape : {y_audio.shape}")
        y_audio = np.repeat(y_audio, window_length, axis=1)
        y_video = np.repeat(y_video, window_length, axis=1)
        # print(f"Window length: {window_length}, Shape : {y_audio.shape}")

    # Get Spikes
    rng = np.random.default_rng(seed=seed)

    if not unimodal:
        x_video, y_video = mv_grt(rng, y_video, nb_inputs)
    else:
        x_video, y_video = lat_s(rng, y_video, nb_inputs)

    x_audio, y_audio = lat_s(rng, y_audio, nb_inputs)

    if len(np.unique(y_data)) == 2:
        y_data[y_data == -1] = 0

    # Merge
    x_data = np.concatenate((x_video, x_audio), -1)

    y_data_local = np.stack((y_video, y_audio), axis=-1)

    y_data_local = y_data_local.transpose(1, 0, 2)

    data_dict = {
        "x_data": torch.from_numpy(x_data).float(),
        "y_data": torch.from_numpy(y_data),
        "y_local": torch.from_numpy(y_data_local),
    }

    return data_dict


class SpikingMultimodal(Dataset):
    def __init__(
        self,
        task,
        data_size=10,
        nb_windows=30,
        window_length=3,
        nb_inputs=14 * 14,
        unimodal=True,
        seed=42,
    ) -> None:
        super().__init__()

        self.generate_data(
            task,
            data_size,
            nb_windows,
            window_length,
            nb_inputs,
            unimodal,
            seed,
        )

        # try:
        #     plotting.generate_gifs(self.data["x_data"], nb_windows=nb_windows)
        # except (ValueError, NameError) as e:
        #     pass

    def generate_data(
        self,
        task,
        data_size,
        nb_windows,
        window_length,
        nb_inputs,
        unimodal,
        seed,
    ):

        self.data_size, self.nb_windows = data_size, nb_windows
        self.unimodal = unimodal
        self.data = generate_spikes(
            task,
            data_size=data_size,
            nb_windows=nb_windows,
            nb_inputs=nb_inputs,
            unimodal=unimodal,
            window_length=window_length,
            seed=seed,
        )

    def __getitem__(self, index: Any):
        return self.data["x_data"][index], self.data["y_data"][index]

    def __len__(self):
        return len(self.data["y_data"])

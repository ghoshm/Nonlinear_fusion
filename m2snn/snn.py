import torch
import numpy as np
import torch.nn as nn
import shutil, os

from m2snn.spikes import *
from m2snn.surrogate import spike_fn
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy


class SNN(nn.Sequential):
    def __init__(self, *modules):
        super(SNN, self).__init__(*modules)

    def forward(self, x):
        recs = []
        for module in self._modules.values():
            x = module(x)
            recs.append(x)
            if isinstance(x, tuple):
                x = x[0]
        return recs


class RSNNLayer(nn.Module):
    def __init__(
        self,
        nb_inputs,
        nb_hidden,
        w_mask=None,
        v_mask=None,
        h_mask=None,
        ie_ratio=None,
        dropout=0.0,
        time_step=1e-3,
        tau_m=10e-3,
        tau_s=5e-3,
        train_ts=False,
        het_ts=False,
        gamma_shape=3,
        th=1.0,
        weight_scale=1.0,
        simple_LIF=False,
        device="cpu",
        dtype=torch.float32,
        rng=None,
    ):
        super().__init__()

        if rng is None:
            rng = np.random.default_rng()

        # Dimensions
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden

        # Hyperparameters
        self.time_step = time_step

        self.train_ts, self.het_ts = train_ts, het_ts

        if het_ts:
            gamma_alpha = np.exp(
                -time_step
                / rng.gamma(
                    gamma_shape,
                    scale=tau_s / gamma_shape,
                    size=(1, nb_hidden),
                )
            )
            gamma_beta = np.exp(
                -time_step
                / rng.gamma(
                    gamma_shape,
                    scale=tau_m / gamma_shape,
                    size=(1, nb_hidden),
                )
            )

            self.alpha_param = nn.Parameter(
                torch.from_numpy(gamma_alpha).to(dtype),
                requires_grad=train_ts,
            )

            self.beta_param = nn.Parameter(
                torch.from_numpy(gamma_beta).to(dtype),
                requires_grad=train_ts,
            )
            # nn.init.constant_(self.alpha, alpha)
            # nn.init.constant_(self.beta, beta)
        else:
            self.alpha_param = nn.Parameter(
                torch.tensor(np.exp(-time_step / tau_s)), requires_grad=train_ts
            )
            self.beta_param = nn.Parameter(
                torch.tensor(np.exp(-time_step / tau_m)), requires_grad=train_ts
            )

        self.th = th

        self.simple_LIF = simple_LIF

        # Other
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout)
        self.weight_scale = weight_scale
        self.dtype = dtype
        self.device = device

        # Masks
        if w_mask is None:
            w_mask = torch.ones(nb_inputs, nb_hidden, dtype=self.dtype)

        if h_mask is None:
            h_mask = torch.ones(nb_hidden, dtype=self.dtype)

        self.register_buffer("w_mask", w_mask)
        self.register_buffer("h_mask", h_mask)

        # Initialise weights
        k = weight_scale * (1.0 / torch.mean(torch.sum(self.w_mask, dim=0)))
        w_params = torch.FloatTensor(nb_inputs, nb_hidden).uniform_(
            -np.sqrt(k), np.sqrt(k)
        )
        self.w_params = nn.Parameter(w_params)

        # Recurrent Connections
        if v_mask is None:  # Feedforward
            v_mask = torch.zeros(nb_hidden, nb_hidden, dtype=self.dtype)
            self.register_buffer("v_params", v_mask)
            self.register_buffer("v_mask", v_mask)
        else:
            self.register_buffer("v_mask", v_mask)
            k = weight_scale * (1.0 / torch.mean(torch.sum(v_mask, dim=0)))
            v_params = torch.FloatTensor(nb_hidden, nb_hidden).uniform_(
                -np.sqrt(k), np.sqrt(k)
            )
            self.v_params = nn.Parameter(v_params)

        self.use_dales_law = ie_ratio is not None
        self.force_positive = self.use_dales_law and ie_ratio == 0.0
        self.ie_ratio = ie_ratio
        if self.use_dales_law:
            self.init_dales_law()

    def init_dales_law(self, inhib_units=[None, None]):
        self.inhib_units = [
            (
                rng.choice(m.shape[0], int(m.shape[0] * self.ie_ratio), replace=False)
                if i is None
                else i
            )
            for m, i in zip([self.w_mask, self.v_mask], inhib_units)
        ]
        # print(self.inhib_units)
        # self.inhib_units = np.arange(ie_ratio * nb_inputs)
        # print(self.inhib_units)
        dales_masks = [torch.ones_like(m) for m in [self.w_mask, self.v_mask]]
        for l, (d_m, inh_units) in enumerate(zip(dales_masks, self.inhib_units)):
            d_m[inh_units, :] = -1
            self.register_buffer(f"dales_mask_{l}", d_m)

    def set_h_mask(self, mask):
        self.register_buffer("h_mask", mask)

    @property
    def w(self):
        return (
            self.w_params * self.w_mask
            if not self.use_dales_law
            else torch.abs(self.w_params * self.w_mask) * self.dales_mask_0
        )

    @property
    def v(self):
        return (
            self.v_params * self.v_mask
            if not self.use_dales_law
            else torch.abs(self.v_params * self.v_mask) * self.dales_mask_1
        )

    @property
    def alpha(self):
        if self.het_ts:
            return self.alpha_param
        else:
            return (
                torch.ones((1, self.nb_hidden)).to(self.alpha_param.device)
                * self.alpha_param
            )

    @property
    def beta(self):
        if self.het_ts:
            return self.beta_param
        else:
            return (
                torch.ones((1, self.nb_hidden)).to(self.beta_param.device)
                * self.beta_param
            )

    def forward(self, inputs, mask=None):
        batch_size, nb_steps = inputs.shape[:2]

        syn = torch.zeros(
            (batch_size, self.nb_hidden), device=inputs.device, dtype=self.dtype
        )
        mem = torch.zeros(
            (batch_size, self.nb_hidden), device=inputs.device, dtype=self.dtype
        )
        out = torch.zeros(
            (batch_size, self.nb_hidden), device=inputs.device, dtype=self.dtype
        )

        mem_rec = []
        spk_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w))

        for t in range(nb_steps):
            h1 = h1_from_input[:, t, :] + torch.einsum("ab,bc->ac", (out, self.v))

            if self.use_dropout:
                h1 = self.dropout(h1)  # apply dropout mask
            h1 *= self.h_mask  # apply activity mask
            out = spike_fn(mem - self.th)
            rst = out.detach()

            if self.simple_LIF:
                new_mem = (self.beta * mem + h1) * (self.th - rst)
                new_syn = syn

            else:
                new_syn = self.alpha * syn + h1
                new_mem = self.beta * mem + (1.0 - self.beta.detach()) * syn * (
                    self.th - rst
                )

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return spk_rec, mem_rec


class ReadoutLayer(nn.Module):
    def __init__(
        self,
        nb_inputs,
        nb_outputs,
        w_mask=None,
        time_step=1e-3,
        tau_m=20e-3,
        tau_s=5e-3,
        weight_scale=1.0,
        ie_ratio=None,
        device="cpu",
        dtype=torch.float32,
        simple_LIF=False,
        spiking=False,
    ):
        super().__init__()

        # Dimensions
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        # Hyperparameters
        self.time_step = time_step
        self.alpha = np.exp(-time_step / tau_s)
        self.beta = np.exp(-time_step / tau_m)

        # Other
        self.weight_scale = weight_scale
        self.dtype = dtype
        self.device = device
        self.spiking = spiking
        self.th = 1.0

        # Mask
        if w_mask is None:
            w_mask = torch.ones(
                nb_inputs, nb_outputs, dtype=self.dtype, device=self.device
            )

        self.register_buffer("w_mask", w_mask)

        # Initialise weights
        k = weight_scale * (1.0 / torch.mean(torch.sum(self.w_mask, dim=0))).cpu()
        ww = (
            torch.FloatTensor(nb_inputs, nb_outputs)
            .uniform_(-np.sqrt(k), np.sqrt(k))
            .to(self.device)
        )
        self.ww = nn.Parameter(ww)
        self.v = None
        self.simple_LIF = simple_LIF

        self.use_dales_law = ie_ratio is not None
        self.force_positive = self.use_dales_law and ie_ratio == 0.0
        self.ie_ratio = ie_ratio
        if self.use_dales_law:
            self.init_dales_law()

    def init_dales_law(self, inhib_units=[None]):
        self.inhib_units = [
            (
                rng.choice(m.shape[0], int(m.shape[0] * self.ie_ratio), replace=False)
                if i is None
                else i
            )
            for m, i in zip([self.w_mask], inhib_units)
        ]
        # print(self.inhib_units)
        # self.inhib_units = np.arange(ie_ratio * nb_inputs)
        # print(self.inhib_units)
        dales_masks = [torch.ones_like(m) for m in [self.w_mask]]
        for l, (d_m, inh_units) in enumerate(zip(dales_masks, self.inhib_units)):
            d_m[inh_units, :] = -1
            self.register_buffer(f"dales_mask_{l}", d_m)

    @property
    def w(self):
        return (
            self.ww * self.w_mask
            if not self.use_dales_law
            else torch.abs(self.ww * self.w_mask) * self.dales_mask_0
        )

    def forward(self, inputs):
        batch_size, nb_steps = inputs.shape[:2]

        # Readout layer
        flt = torch.zeros(
            (batch_size, self.nb_outputs), device=inputs.device, dtype=self.dtype
        )
        out = torch.zeros(
            (batch_size, self.nb_outputs), device=inputs.device, dtype=self.dtype
        )
        out_rec = []

        h = torch.einsum("abc,cd->abd", (inputs, self.w))
        rst = out.detach()

        for t in range(nb_steps):
            if self.simple_LIF:
                new_out = self.beta * out + h[:, t, :]
                new_flt = flt

            else:
                new_flt = self.alpha * flt + h[:, t, :]
                new_out = self.beta * out + (1.0 - self.beta) * flt

            if self.spiking:
                new_out *= self.th - rst
                out = spike_fn(new_out - self.th)

            else:
                out = new_out

            flt = new_flt

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        return out_rec


def run_snn(network, inputs):
    all_recordings = network(inputs)

    spk_rec = []
    mem_rec = []

    for layer in range(len(network) - 1):
        spk_rec.append(all_recordings[layer][0])
        mem_rec.append(all_recordings[layer][1])

    spk_rec = torch.cat(spk_rec, axis=2)
    mem_rec = torch.cat(mem_rec, axis=2)
    out_rec = all_recordings[-1]

    return out_rec, spk_rec, mem_rec


def batched_run_snn(network, batch, device, detection_task=False):
    x_local, y_local = [b.to(device) for b in batch]

    if detection_task:
        y_local += 1

    output, spks, _ = run_snn(network, x_local)
    out_sum = torch.sum(output, 1)  # sum over time

    # Add small random values if equal
    eq = torch.where((out_sum[:, 0] == out_sum[:, 1]))  # [2 outputs] find equal sums
    out_sum[eq] += torch.rand((out_sum[eq].shape)).to(device) / 1000
    _, am = torch.max(out_sum, 1)  # argmax over output units
    acc = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels

    return acc, spks


def masked_inference(
    network,
    excluded_neurons,
    batchs=None,
    loader=None,
    device=torch.device("cuda"),
    max_samples=None,
):
    if max_samples is None:
        max_samples = len(loader)
    i = 0
    accs = []
    mask = torch.tensor(
        [i not in excluded_neurons for i in range(network[1].nb_hidden)]
    )
    network[1].set_h_mask(mask)
    detection_task = network[-1].w.shape[1] == 3
    network.to(device)

    with torch.no_grad():
        if batchs is not None:
            for batch in batchs:
                acc, _ = batched_run_snn(network, batch, device, detection_task)
                accs.append(acc)
        else:
            assert loader is not None
            for batch in loader:
                if i >= max_samples:
                    break
                acc, _ = batched_run_snn(network, batch, device, detection_task)
                accs.append(acc)
                i += 1

    return np.mean(accs)


def get_n_w(n_in, n_hids, n_out):
    n_w = n_in * n_hids[0] + n_out * n_hids[-1]
    for n1, n2 in zip(n_hids[:-1], n_hids[1:]):
        n_w += n1 * n2
    return n_w


def get_w_mask(nb_inputs, nb_hiddens):
    nb_hn1 = int(nb_hiddens[0] / 2)
    nb_arc = np.zeros(sum(nb_hiddens))
    nb_arc[nb_hn1 : nb_hiddens[0]] = 1
    nb_arc[nb_hiddens[0] :] = 2

    # w0 mask
    w0_mask = torch.zeros(2 * nb_inputs, nb_hiddens[0])
    w0_mask[:nb_inputs, :nb_hn1] = 1
    w0_mask[nb_inputs:, nb_hn1:] = 1
    w0_mask = w0_mask

    return nb_hn1, nb_arc, w0_mask


def build_network(
    nb_inputs=14 * 14,
    nb_hiddens=[60, 30],
    nb_outputs=2,
    weight_scale=1,
    arch_type=1,
    device="cpu",
    dropout=0.0,
    ie_ratio=None,
    feedforward=False,
    tau_m=10e-3,
    tau_s=5e-3,
    train_ts=False,
    het_ts=False,
    gamma_shape=3,
    simple_LIF=False,
    spiking_readout=False,
    plot_masks=False,
    seed=None,
):
    rng = np.random.default_rng(seed)

    if isinstance(nb_hiddens, str):
        nb_hiddens = [int(n) for n in nb_hiddens.strip("[]").split(", ")]

    if isinstance("arch_type", str):
        arch_types = {"uni": 0, "multi": 1, "double_uni": 2}
        assert arch_type in arch_types.keys(), "Invalid architecture type"
        arch_type = arch_types[arch_type]

    if arch_type == 0:
        n_w = get_n_w(nb_inputs, nb_hiddens, nb_outputs)
        n_hid = int(n_w / (nb_inputs + nb_outputs))
        nb_hiddens = [n_hid]

        nb_hn1, nb_arc, w0_mask = get_w_mask(nb_inputs, nb_hiddens)

        layer_1_config = {
            "nb_inputs": 2 * nb_inputs,
            "nb_hidden": n_hid,
            "w_mask": w0_mask,
            "v_mask": None,
            "weight_scale": weight_scale,
            "ie_ratio": ie_ratio,
            "dropout": dropout,
            "device": device,
            "tau_s": tau_s,
            "tau_m": tau_m,
            "train_ts": train_ts,
            "het_ts": het_ts,
            "gamma_shape": gamma_shape,
            "simple_LIF": simple_LIF,
            "rng": rng,
        }

        readout_config = {
            "nb_inputs": n_hid,
            "nb_outputs": nb_outputs,
            "weight_scale": weight_scale,
            "ie_ratio": ie_ratio,
            "device": device,
            "spiking": spiking_readout,
            "simple_LIF": simple_LIF,
        }

        network = SNN(RSNNLayer(**layer_1_config), ReadoutLayer(**readout_config))

        masks = [w0_mask]

    else:
        if arch_type == 1:
            # w_O mask
            nb_hn1, nb_arc, w0_mask = get_w_mask(nb_inputs, nb_hiddens)
            # v1 mask
            if feedforward:
                v1_mask = None
            else:
                v1_mask = 1.0 * ~torch.eye(nb_hiddens[1], dtype=bool)
                v1_mask = v1_mask  # .to(device)

            w1_mask = None

        elif arch_type == 2:
            # w_O mask
            nb_hn1, nb_arc, w0_mask = get_w_mask(nb_inputs, nb_hiddens)
            # w1 mask
            w1_mask = torch.zeros(nb_hiddens[0], nb_hiddens[1])
            w1_mask[:nb_hn1, : int(nb_hn1 / 2)] = 1
            w1_mask[nb_hn1:, int(nb_hn1 / 2) :] = 1
            w1_mask = w1_mask  # .to(device)

            # v1 mask
            if feedforward:
                v1_mask = None
            else:
                v1_mask = torch.zeros((nb_hiddens[1], nb_hiddens[1]))
                h = int(nb_hiddens[1] / 2)
                v1_mask[:h, :h] = 1
                v1_mask[h:, h:] = 1
                v1_mask = v1_mask * ~torch.eye(nb_hiddens[1], dtype=bool)
                v1_mask = v1_mask  # .to(device)

        masks = [w0_mask, w1_mask, v1_mask]

        layer_1_config = {
            "nb_inputs": 2 * nb_inputs,
            "nb_hidden": nb_hiddens[0],
            "w_mask": w0_mask,
            "v_mask": None,
            "weight_scale": weight_scale,
            "ie_ratio": ie_ratio,
            "dropout": dropout,
            "device": device,
            "tau_s": tau_s,
            "tau_m": tau_m,
            "train_ts": train_ts,
            "het_ts": het_ts,
            "gamma_shape": gamma_shape,
            "simple_LIF": simple_LIF,
            "rng": rng,
        }

        layer_2_config = deepcopy(layer_1_config)
        layer_2_config.update(
            {
                "nb_inputs": nb_hiddens[0],
                "nb_hidden": nb_hiddens[1],
                "w_mask": w1_mask,
                "v_mask": v1_mask,
            }
        )

        readout_config = {
            "nb_inputs": nb_hiddens[1],
            "nb_outputs": nb_outputs,
            "weight_scale": weight_scale,
            "ie_ratio": ie_ratio,
            "device": device,
            "spiking": spiking_readout,
            "simple_LIF": simple_LIF,
        }

        network = SNN(
            RSNNLayer(**layer_1_config),
            RSNNLayer(**layer_2_config),
            ReadoutLayer(**readout_config),
        )

        if ie_ratio is not None:
            network[1].init_dales_law([network[0].inhib_units[1], None])
            network[-1].init_dales_law([network[1].inhib_units[1], None])

    if plot_masks:
        fig, axs = plt.subplots(1, len(masks), figsize=(5, 2))
        try:
            axs[0]
        except TypeError:
            axs = [axs]
        for ax, mask in zip(axs, masks):
            if mask is not None:
                ax.imshow(mask.cpu().data.numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    nb_arc = np.zeros(sum(nb_hiddens))
    nb_arc[nb_hn1 : nb_hiddens[0]] = 1
    nb_arc[nb_hiddens[0] :] = 2

    if arch_type == 2:
        nb_arc[int(nb_hiddens[0] + (nb_hiddens[1] / 2)) :] = 3

    network.nb_arc = nb_arc
    network.dims = [
        nb_inputs,
        *nb_hiddens,
    ]

    network.arch = arch_type

    return network.to(device)


def get_effective_weights(network):
    return [n.w for n in network.children()], [
        n.v for n in network.children() if n.v is not None
    ]


def ensure_positive_weights(network):
    for p in network.w:
        assert (p >= 0).all(), "Negative Weights !"
    for p in network.v:
        assert (p >= 0).all(), "Negative Weights !"


class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "alpha_param"):
            if isinstance(module.alpha_param, torch.Tensor):
                # module.alpha.data = torch.clamp(module.alpha.data, 1/(np.e**(1/4)), 0.995)  # Clip tau to [4*time_step, ]
                module.alpha.data = torch.clamp(module.alpha_param.data, 1 / np.e, 0.99)
        if hasattr(module, "beta_param"):
            if isinstance(module.beta_param, torch.Tensor):
                module.beta_param.data = torch.clamp(
                    module.beta_param.data, 1 / np.e, 0.99
                )
        if hasattr(module, "th"):
            if isinstance(module.th, torch.Tensor):
                module.th.data = torch.clamp(module.th.data, 0.5, 1.5)


if __name__ == "__main__":
    data_size = 100
    nb_steps = 120
    nb_inputs = 14 * 14

    p_dict = {
        "CC": 0.1,
        "CN": 0.45,
        "CI": 0,
        "IN": 0.4,
        "II": 0.05,
    }

    batch_size = 32

    network = build_network(weight_scale=1)
    datasets, loaders = get_spiking_datasets(
        batch_size, p_dict, [data_size, data_size // 10], nb_steps
    )

    # Check spiking

    x_local, _ = get_mini_batch(loaders[0])
    _, spks, _ = run_snn(network, x_local)
    spks = spks.cpu().detach().numpy()

    print("Mean spikes per sample: " + str(np.mean(np.sum(np.sum(spks, 1), 1))))

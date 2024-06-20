from numpy.random import choice, rand, randn
import numpy as np
import lea  # probability calculations, see https://pypi.org/project/lea/
from collections import defaultdict
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import copy
import contourpy as cp


def plog(p):
    if p == 0:
        return -1e100  # don't set to -inf because it makes some stuff nan
    else:
        return np.log(p)


def detection_params_search(p_ranges, nb_trials, nb_steps):
    classifier_type = MAPClassifier

    # Sample task parameters
    while True:
        p = {
            k: rand() * (upper - lower) + lower
            for k, (lower, upper) in p_ranges.items()
        }

        if p["pc"] <= 0.5 * p["pn"]:
            continue
        if p["pi"] >= p["pc"]:
            continue
        if p["pi"] >= 0.5 * p["pn"]:
            continue
        if p["pi"] + p["pc"] <= p["pn"]:
            continue
        if p["pc"] + p["pi"] > 1.0:
            continue
        break

    # Generate trials
    task = DetectionTask(**p)
    trials = task.generate_trials(nb_trials, nb_steps)

    # Calculate accuracy
    accs_tmp = []
    for pairs in [False, True]:
        classifier = classifier_type(task, pairs=pairs)
        res = classifier.test(trials)
        accs_tmp.append(res.accuracy)

    # Filter for accuracy
    _, a = np.unique(trials.M, return_counts=True)  # majority class classifier
    a = a.max() / a.sum()
    w = 1 - a
    c = (1 + a) / 2
    if (max(accs_tmp) > (c - w / 2 * 0.75)) & (min(accs_tmp) < (c + w / 2 * 0.75)):
        return accs_tmp, np.array(list(p.values()))
    else:
        return [0.0, 0.0], np.array(list(p.values())) * 0.0


@dataclass
class Task:
    @property
    def random_variables(self):
        return NotImplemented

    def generate_trials(self, repeats, time_steps):
        # random variables
        rv = self.random_variables
        M = rv["M"]
        A = rv["A"]
        V = rv["V"]
        # cache calculated joint distribution
        joint_dists = {}
        for m in [-1, 0, 1]:
            if lea.P(M == m) == 0:
                continue
            joint_dists[m] = lea.joint(A, V).given(M == m).calc()
        # generate true target values
        arr_M = np.array(M.random(repeats))
        steps = np.array(
            [joint_dists[m].random(time_steps) for m in arr_M]
        )  # steps has shape (repeats, time_steps, 2)
        if time_steps == 0:
            # print(steps.shape)
            return Trials(
                repeats=repeats,
                time_steps=time_steps,
                task=self,
                M=arr_M,
                A=steps[:, None],
                V=steps[:, None],
            )
        else:
            return Trials(
                repeats=repeats,
                time_steps=time_steps,
                task=self,
                M=arr_M,
                A=steps[:, :, 0],
                V=steps[:, :, 1],
            )

    @property
    def baseline(self):
        if not hasattr(self, "_baseline"):
            M = self.random_variables["M"]
            self._baseline = max([lea.P(M == m) for m in [-1, 0, 1]])
        return self._baseline

    def baseline_reward(self, reward):
        M = self.random_variables["M"]
        probs = np.array([lea.P(M == m) for m in [-1, 0, 1]])
        expected_rewards = np.einsum("m,mg->g", probs, reward)
        return np.max(expected_rewards)


@dataclass
class DetectionTask(Task):
    pm: float
    pe: float
    pn: float
    pc: float
    pi: float

    @property
    def random_variables(self):
        if hasattr(self, "_random_vars"):
            return self._random_vars
        target = lea.pmf({-1: self.pm * 0.5, 1: self.pm * 0.5, 0: 1 - self.pm})
        emit_if_target = lea.event(self.pe)
        emit_if_no_target = lea.event(0.0)
        emit = target.switch(
            {-1: emit_if_target, 1: emit_if_target, 0: emit_if_no_target}
        )
        signal_dist = {
            (-1, True): lea.pmf({-1: self.pc, +1: self.pi, 0: 1 - self.pc - self.pi}),
            (+1, True): lea.pmf({+1: self.pc, -1: self.pi, 0: 1 - self.pc - self.pi}),
            (0, True): lea.pmf({-1: 0, +1: 0, 0: 1.0}),  # cannot happen
            (-1, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
            (0, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
            (+1, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
        }
        signal = lea.joint(target, emit).switch(signal_dist)
        signal_A, signal_V = signal.clone(n=2, shared=(target, emit))
        self._random_vars = {"M": target, "E": emit, "A": signal_A, "V": signal_V}
        return self._random_vars


# class DetectionTaskFamily:
#     def __init__(self, n, table_size=20, table_min=0.0, table_max=0.3, pairs=False):
#         self.S_table = S_table = np.linspace(table_min, table_max, table_size)
#         self.acc_table = np.array([ideal_accuracy(self._raw_task(s), n, pairs=pairs) for s in S_table])

#     def _raw_task(self, s):
#         return DetectionTask(pm=2/3, pe=s, pn=0.3, pc=.95, pi=0.01)

#     def inverse_acc_interp(self, acc):
#         return np.interp(acc, self.acc_table, self.S_table)

#     def task(self, s):
#         T = self._raw_task(self.inverse_acc_interp(1/3+2/3*s))
#         T.s = s
#         return T


class DetectionTaskFamily:
    def __init__(
        self,
        n,
        pm=2 / 3,
        pn=1 / 3,
        pi=0.01,
        pe_range=np.linspace(0, 0.3, 10),
        pc_range=np.linspace(1 / 6, 0.95, 10),
        desired_accuracy=0.8,
    ):
        self.n = n
        self.pm = pm
        self.pn = pn
        self.pi = pi
        self.pe_range = pe_range
        self.pc_range = pc_range
        # Generate a table of accuracies
        accuracy_pairs = lambda pe, pc: ideal_accuracy(self._raw_task(pe, pc), n)
        # accuracy_singletons = lambda pe, pc: ideal_accuracy(self._raw_task(pe, pc), n, pairs=False)
        self.PE, self.PC = PE, PC = np.meshgrid(pe_range, pc_range)
        self.A_pairs = A_pairs = np.vectorize(accuracy_pairs)(PE, PC)
        # self.A_singletons = A_singletons = np.vectorize(accuracy_singletons)(PE, PC)
        # Generate the contour line corresponding to performance level 0.8
        cont_gen = cp.contour_generator(x=PE, y=PC, z=A_pairs)
        (contour_line,) = cont_gen.lines(desired_accuracy)
        contour_pe, contour_pc = contour_line[
            ::-1, :
        ].T  # reverse order because interp needs increasing
        # Create a family by linearly interpolating this with the pc axis the controlling one
        # contour_pe_min = contour_pe.min()
        # contour_pe_max = contour_pe.max()
        # print(contour_pe_min, contour_pe_max)
        # self.pe_func = lambda s: contour_pe_min+(contour_pe_max-contour_pe_min)*s
        # self.pc_func = lambda s: np.interp(self.pe_func(s), contour_pe, contour_pc)
        contour_pc_min = contour_pc.min()
        contour_pc_max = contour_pc.max()
        # print(f'pe range: {contour_pe.min()}-{contour_pe.max()}')
        # print(f'pc range: {contour_pc.min()}-{contour_pc.max()}')
        self.pc_func = lambda s: contour_pc_min + (contour_pc_max - contour_pc_min) * s
        self.pe_func = lambda s: np.interp(self.pc_func(s), contour_pc, contour_pe)

    def task(self, s):
        T = self._raw_task(self.pe_func(s), self.pc_func(s))
        T.s = s
        return T

    def _raw_task(self, pe, pc):
        return DetectionTask(pm=self.pm, pe=pe, pn=self.pn, pc=pc, pi=self.pi)


@dataclass
class ClassicalTask(Task):
    s: float

    @property
    def pc(self):
        return (1 + 2 * self.s) / 3

    @property
    def pi(self):
        return (1 - self.s) / 3

    pn = pi

    @property
    def random_variables(self):
        if not hasattr(self, "_random_vars"):
            M = lea.pmf({-1: 0.5, 1: 0.5})
            S = M.switch(
                {
                    -1: lea.pmf({-1: self.pc, +1: self.pi, 0: self.pn}),
                    +1: lea.pmf({+1: self.pc, -1: self.pi, 0: self.pn}),
                }
            )
            A, V = S.clone(n=2, shared=(M,))
            self._random_vars = dict(M=M, A=A, V=V)
        return self._random_vars


@dataclass
class ComodulationTask(Task):
    pcc: float
    pii: float
    pci: float
    pcn: float
    pin: float

    @property
    def pnn(self):
        return 1 - self.pcc - self.pii - self.pci - self.pcn - self.pin

    @property
    def random_variables(self):
        if not hasattr(self, "_random_vars"):
            M = lea.pmf({-1: 0.5, 1: 0.5})
            positive_dist = {
                (1, 1): self.pcc,
                (-1, -1): self.pii,
                (1, -1): self.pci,
                (-1, 1): self.pci,
                (1, 0): self.pcn,
                (0, 1): self.pcn,
                (-1, 0): self.pin,
                (0, -1): self.pin,
                (0, 0): self.pnn,
            }
            AV = M.switch(
                {
                    -1: lea.pmf(
                        dict(((-a, -v), p) for (a, v), p in positive_dist.items())
                    ),
                    +1: lea.pmf(positive_dist),
                }
            ).as_joint("A", "V")
            A = AV.A
            V = AV.V
            self._random_vars = dict(M=M, A=A, V=V, AV=AV)
        return self._random_vars


@dataclass
class BalancedComodulationTask(ComodulationTask):
    def __init__(self, s):
        self.s = s
        pcc = 1 / 3 * s + 1 / 9 * (1 - s)
        pii = 1 / 9 * (1 - s)
        pcn = (1 + pii - 3 * pcc) / 4
        pin = (1 + pcc - 3 * pii) / 4
        super().__init__(pcc=pcc, pii=pii, pci=0, pcn=pcn, pin=pin)

    def __str__(self):
        return f"BalancedComodulationTask(s={self.s})"


@dataclass
class Trials:
    repeats: int
    time_steps: int
    M: np.ndarray
    A: np.ndarray
    V: np.ndarray
    task: Task

    def counts(self, pairs=True):
        A = self.A
        V = self.V
        if self.time_steps == 0:
            return np.zeros((self.repeats, 6 + 3 * pairs))
        if pairs:
            AV = (A + 1) + 3 * (V + 1)  # shape (repeats, time_steps)
            C = np.apply_along_axis(np.bincount, 1, AV, minlength=9)  # (repeats, 9)
        else:
            CA = np.apply_along_axis(np.bincount, 1, A + 1, minlength=3)  # (repeats, 3)
            CV = np.apply_along_axis(np.bincount, 1, V + 1, minlength=3)  # (repeats, 3)
            C = np.concatenate((CA, CV), axis=1)
        return C

    _singleton_labels = ["A=-1", "A=0", "A=1", "V=-1", "V=0", "V=1"]
    _coincidence_labels = [f"({a}, {v})" for v in [-1, 0, 1] for a in [-1, 0, 1]]

    def count_labels(self, pairs=True):
        if pairs:
            return self._coincidence_labels
        else:
            return self._singleton_labels

    def simple_joint_counts(self, pairs=True):
        Cs = self.counts(pairs=False).T
        C = [
            Cs[0] + Cs[3],  # net count in favour of -1
            Cs[1] + Cs[4],  # net count in favour of 0
            Cs[2] + Cs[5],  # net count in favour of +1
        ]
        if pairs:
            Cc = dict(
                (k, v)
                for k, v in zip(self._coincidence_labels, self.counts(pairs=True).T)
            )
            C = C + [Cc["(-1, -1)"], Cc["(1, 1)"]]
        return np.array(C).T

    joint_count_labels = ["-1", "0", "1", "(-1, -1)", "(1, 1)"]


class Classifier:
    def __init__(self, task, pairs=True, reward=None):
        self.task = task
        self.pairs = pairs
        self.reward = reward
        self.basename = f"{self.__class__.__name__}"

    @property
    def name(self):
        n = self.basename
        if self.pairs:
            n += "/pairs"
        else:
            n += "/singletons"
        if self.reward is not None:
            n += "/reward"
        return n

    def train(self, trials):
        return self  # default no training needed

    def test(self, trials):
        return NotImplemented


class LinearClassifier(Classifier):
    def __init__(
        self, task, pairs=True, reward=None, model=linear_model.LogisticRegression
    ):
        super().__init__(task, pairs=pairs, reward=reward)
        self.model_class = model
        self.basename = model.__name__

    def _vector(self, trials):
        return trials.counts(pairs=self.pairs)

    def train(self, trials):
        classifier = copy.copy(self)
        classifier.training_trials = trials
        classifier.model = self.model_class()
        counts = self._vector(trials)
        classifier.pipeline = make_pipeline(StandardScaler(), classifier.model)
        classifier.pipeline.fit(counts, trials.M)
        return classifier

    def test(self, trials):
        counts = self._vector(trials)
        pred = self.pipeline.predict(counts)
        return Results(
            trials=trials,
            predictions=pred,
            classifier=self,
        )


class LinearCoincidenceClassifier(LinearClassifier):
    def __init__(
        self, task, pairs=True, reward=None, model=linear_model.LogisticRegression
    ):
        super().__init__(task, pairs=pairs, reward=reward, model=model)
        self.basename = "LinearCoincidence"

    def _vector(self, trials):
        return trials.simple_joint_counts(pairs=self.pairs)


class MAPClassifier(Classifier):
    def _vector(self, trials):
        return trials.counts(pairs=self.pairs)

    def __init__(self, task, pairs=True, reward=None):
        super().__init__(task, pairs=pairs, reward=reward)
        self.basename = "MAP"

        rv = task.random_variables
        M = rv["M"]
        A = rv["A"]
        V = rv["V"]

        prior = np.zeros(3)
        for m in [-1, 0, 1]:
            prior[m + 1] = plog(lea.P(M == m))

        weights = np.zeros((3, 6 + 3 * pairs))
        if pairs:
            for m in [-1, 0, 1]:
                if lea.P(M == m) == 0:
                    continue
                for a in [-1, 0, 1]:
                    for v in [-1, 0, 1]:
                        weights[m + 1, (a + 1) + 3 * (v + 1)] = plog(
                            lea.P(((A == a) & (V == v)).given(M == m))
                        )
        else:
            for m in [-1, 0, 1]:
                if lea.P(M == m) == 0:
                    continue
                for a in [-1, 0, 1]:
                    weights[m + 1, a + 1] = plog(lea.P((A == a).given(M == m)))
                for v in [-1, 0, 1]:
                    weights[m + 1, v + 4] = plog(lea.P((V == v).given(M == m)))

        self.prior = prior
        self.weights = weights

    def test(self, trials):
        C = self._vector(trials)
        evidence = np.einsum("mi,ri->rm", self.weights, C) + self.prior[None, :]
        if self.reward is None:
            pred = np.argmax(evidence, axis=1) - 1
        else:
            # evidence[m] is currently log(P(observations|M=m)P(M=m)), so we want to compute
            # expected_reward[g] = E[reward|M,g] where g is action decision and expectation is over M random variable
            # probs, evidence have shape (repeats, 3)
            probs = np.exp(evidence)
            expected_rewards = np.einsum("rm,mg->rg", probs, self.reward)
            pred = np.argmax(expected_rewards, axis=1) - 1
        return Results(
            trials=trials,
            predictions=pred,
            classifier=self,
            evidence=evidence,
        )


# This is not quite right because the weights calculation is off, but it's possible
# to find perfect weights because LinearCoincidenceClassifier is perfect and it's
# possible to solve equations to get them if you assume that because of symmetries
# the exact count of say (+1, 0) is the same as (0, +1) (which isn't true but
# doesn't change the probabilities or weights so the calculation comes out the same).
# class SimpleMAPClassifier(MAPClassifier):
#     def __init__(self, task, pairs=True, reward=None):
#         super().__init__(task, pairs=pairs, reward=reward)
#         self.basename = 'SimpleMAP'

#         rv = task.random_variables
#         M = rv['M']
#         A = rv['A']
#         V = rv['V']

#         prior = np.zeros(3)
#         for m in [-1, 0, 1]:
#             prior[m+1] = plog( lea.P(M==m) )

#         weights = np.zeros((3, 3+2*pairs))
#         for m in [-1, 0, 1]:
#             if lea.P(M==m):
#                 # for a in [-1, 0, 1]:
#                 #     weights[m+1, a+1] = plog( lea.P( (A==a).given(M==m) ) )
#                 for a in [-1, 1]:
#                     weights[m+1, a+1] = plog( lea.P( (A==a).given(M==m) ) ) * lea.P( ((A==a)&(V==a)).given(M==m) ) / ( 2*lea.P( ((A==a)&(V==a)).given(M==m) ) + lea.P( ((A==a)&(V==0)).given(M==m) ))
#                 weights[m+1, 1] = plog( lea.P( (A==0).given(M==m) ) )
#                 if pairs:
#                     for i, a in enumerate([-1, 1]):
#                         weights[m+1, 3+i] = plog( lea.P( ((A==a)&(V==a)).given(M==m) ) )

#         self.prior = prior
#         self.weights = weights

#     def _vector(self, trials):
#         return trials.simple_joint_counts(pairs=self.pairs)


@dataclass
class Results:
    trials: Trials
    predictions: np.ndarray
    classifier: Classifier
    evidence: Optional[np.ndarray] = None

    @property
    def accuracy(self):
        if not hasattr(self, "_accuracy"):
            self._accuracy = (
                sum(self.predictions == self.trials.M) / self.trials.repeats
            )
        return self._accuracy

    @property
    def confusions(self):
        # TODO: not tested
        if not hasattr(self, "_confusions"):
            C = self._confusions = np.zeros((3, 3), dtype=int)
            np.add.at(C, (self.trials.M + 1, self.predictions + 1), 1)
        return self._confusions

    @property
    def confusions_normalised(self):
        C = self.confusions
        C = (C * 1.0) / np.sum(C, axis=1)[:, None]
        return C

    def compute_expected_reward(self, reward):
        R = reward  # indices are (M, G)
        G = self.predictions  # indices are (repeat,)
        M = self.trials.M  # indices are (repeat,)
        return R[M + 1, G + 1].sum() / len(M)

    @property
    def expected_reward(self):
        if not hasattr(self, "_expected_reward"):
            self._expected_reward = self.compute_expected_reward(
                reward=self.classifier.reward
            )
        return self._expected_reward


def reward_matrix(reward):
    R = np.zeros((3, 3))
    for m in [-1, 0, 1]:
        for g in [-1, 0, 1]:
            R[m + 1, g + 1] = reward[m, g]
    return R


def ideal_accuracy(task, time_steps, repeats=10000, pairs=True):
    classifier = MAPClassifier(task, pairs=pairs)
    trials = task.generate_trials(repeats, time_steps)
    return classifier.test(trials).accuracy


########################################################################################################
########################### PLOTTING AND FIGURES #######################################################
########################################################################################################


def tasks_figure(
    tasks,
    classifier_types,
    N=[0, 1, 3, 5, 10, 30, 50, 90],
    repeats=10000,
    reward=None,
    pair_options=[False, True],
    reward_options=[False, True],
    same_axes=False,  # set to True or a string (to use that string as title)
    plot_options=None,  # dict of names or func with args (task=None, classifier=None, pairs=None, do_reward=None, name=None)
    name_func=None,  # function of (task, classifier, pairs, do_reward) to give name
    plot_func=plt.plot,  # can use semilogx for instance
):
    if plot_options is None:
        plot_options = {
            "MAP/pairs": dict(c="C2", ls="--"),
            "MAP/singletons": dict(c="C3", ls="--"),
            "MAP/pairs/reward": dict(c="C2", ls="-"),
            "MAP/singletons/reward": dict(c="C3", ls="-"),
            "SimpleMAP/pairs": dict(c="C2", ls="--", marker="o"),
            "SimpleMAP/singletons": dict(c="C3", ls="--", marker="o"),
            "SimpleMAP/pairs/reward": dict(c="C2", ls="-", marker="o"),
            "SimpleMAP/singletons/reward": dict(c="C3", ls="-", marker="o"),
            "LogisticRegression/pairs": dict(c="C2", ls=":"),
            "LogisticRegression/singletons": dict(c="C3", ls=":"),
            "LinearCoincidence/pairs": dict(c="C2", ls=":"),
            "LinearCoincidence/singletons": dict(c="C3", ls=":"),
            "baseline": dict(c="C5", ls=":"),
        }
    if not callable(plot_options):
        _plot_options = plot_options
        plot_options = lambda task=None, classifier=None, pairs=None, do_reward=None, name=None: _plot_options[
            name
        ]
    if reward is None:
        reward = reward_matrix(
            {
                # pairs (true, guess): reward
                (-1, -1): 10,
                (-1, 0): -0.1,
                (
                    -1,
                    1,
                ): -1,  # prey went left, reward if correct, small penalty if wrong action taken
                (0, -1): -1,
                (0, 0): -0.1,
                (0, 1): -1,  # no prey, small penalty if wrong action taken
                (1, -1): -1,
                (1, 0): -0.1,
                (
                    1,
                    1,
                ): 10,  # prey went right, reward if correct, small penalty if wrong action taken
            }
        )
    if name_func is None:
        if same_axes:
            name_func = (
                lambda task, classifier, pairs, do_reward: str(task)
                + ":"
                + classifier.name
            )
        else:
            name_func = lambda task, classifier, pairs, do_reward: classifier.name

    show_reward_axes = True in reward_options
    figcols = 1 if same_axes else len(tasks)
    figrows = 1 + show_reward_axes
    figheight = 8 if show_reward_axes else 5
    plt.figure(figsize=(5 * figcols, figheight))

    for i, task in enumerate(tasks):
        all_trials = [task.generate_trials(repeats, n) for n in N]
        all_training_trials = [task.generate_trials(repeats, n) for n in N]
        upper_plot = 1 + i * (not same_axes)
        lower_plot = 1 + i * (not same_axes) + figcols
        plt.subplot(figrows, figcols, upper_plot)
        if not same_axes:
            plt.title(str(task).replace("(", "\n").replace(")", ""))
        if isinstance(same_axes, str):
            plt.title(same_axes)
        for pairs in pair_options:
            for classifier_type in classifier_types:
                names = {}
                results = {}
                acc = {}
                er = {}
                for do_reward in reward_options:
                    classifier = classifier_type(
                        task, pairs=pairs, reward=reward if do_reward else None
                    )
                    names[do_reward] = name_func(task, classifier, pairs, do_reward)
                    results[do_reward] = [
                        classifier.train(training_trials).test(trials)
                        for trials, training_trials in zip(
                            all_trials, all_training_trials
                        )
                    ]
                    acc[do_reward] = [res.accuracy for res in results[do_reward]]
                    er[do_reward] = [
                        res.compute_expected_reward(reward)
                        for res in results[do_reward]
                    ]
                # plot it
                plt.subplot(figrows, figcols, upper_plot)
                for do_reward in reward_options:
                    plot_func(
                        N,
                        acc[do_reward],
                        label=names[do_reward],
                        **plot_options(
                            task, classifier, pairs, do_reward, names[do_reward]
                        ),
                    )
                if show_reward_axes:
                    plt.subplot(figrows, figcols, lower_plot)
                    for do_reward in reward_options:
                        plot_func(
                            N,
                            er[do_reward],
                            label=names[do_reward],
                            **plot_options(
                                task, classifier, pairs, do_reward, names[do_reward]
                            ),
                        )
        if not same_axes or (same_axes and i == len(tasks) - 1):
            plt.subplot(figrows, figcols, upper_plot)
            plt.axhline(
                task.baseline, label="baseline", **plot_options(name="baseline")
            )
            plt.ylim(0, 1)
            if i == 0 or (same_axes and i == len(tasks) - 1):
                plt.ylabel("P(correct)")
                plt.legend(loc="best")
            if show_reward_axes:
                plt.subplot(figrows, figcols, lower_plot)
                plt.axhline(
                    task.baseline_reward(reward),
                    label="baseline",
                    **plot_options(name="baseline"),
                )
                plt.xlabel("Number of time steps")
                if i == 0 or (same_axes and i == len(tasks) - 1):
                    plt.ylabel("Expected reward")
            else:
                plt.xlabel("Number of time steps")
    plt.tight_layout()


def image_extent_from_linear_range(linear_range):
    xmin = np.amin(linear_range)
    xmax = np.amax(linear_range)
    xdiff = linear_range[1] - linear_range[0]
    return (xmin - xdiff * 0.5, xmax + xdiff * 0.5)


if __name__ == "__main__":
    from matplotlib import cm

    plt.style.use("./style_sheet.mplstyle")

    # # Standard plot
    # tasks = [
    #     ClassicalTask(s=0.2),
    #     BalancedComodulationTask(s=0.4),
    #     DetectionTask(pm=0.1, pe=0.05, pn=0.3, pc=0.9, pi=0.01),
    # ]
    # classifier_types = [
    #     # LinearClassifier,
    #     LinearCoincidenceClassifier,
    #     MAPClassifier,
    #     # SimpleMAPClassifier,
    # ]
    # tasks_figure(tasks, classifier_types,
    #              reward_options=[False],
    #              )

    # # Task family plot
    # # s_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    # # tasks = [ClassicalTask(s=s) for s in s_range]
    # # s_range = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # # tasks = [BalancedComodulationTask(s=s) for s in s_range]
    # s_range = np.linspace(0, 1, 6)
    # family = DetectionTaskFamily(n=90)
    # tasks = [family.task(s) for s in s_range]
    # classifier_types = [MAPClassifier]
    # def plot_options(task=None, classifier=None, pairs=None, do_reward=None, name=None):
    #     if name=='baseline':
    #         return dict(c='C5', ls=':')
    #     return dict(
    #         c=cm.viridis( (task.s-np.amin(s_range))/(np.amax(s_range)-np.amin(s_range)) ),
    #     )
    # name_func = lambda task, classifier, pairs, do_reward: f's={task.s}'
    # tasks_figure(tasks, classifier_types, pair_options=[True], reward_options=[False],
    #              same_axes=tasks[0].__class__.__name__,
    #              plot_options=plot_options, name_func=name_func)

    # Robustness plot
    n = 90
    repeats = 10000

    def task_type(s):
        # return DetectionTask(pm=s, pe=0.05, pn=0.3, pc=(1-s), pi=0.01)
        return DetectionTask(pm=0.1, pe=s, pn=0.3, pc=0.9, pi=0.01)

    # task_type = BalancedComodulationTask
    classifier_type = LinearCoincidenceClassifier
    s_range = [0.1, 0.2, 0.3, 0.4, 0.5]  # , 0.6, 0.7, 0.8]
    all_tasks = [task_type(s=s) for s in s_range]
    all_trials_train = [task.generate_trials(repeats, n) for task in all_tasks]
    all_trials_test = [task.generate_trials(repeats, n) for task in all_tasks]
    # # debug counts
    # for trials in all_trials_test:
    #     C = trials.simple_joint_counts()
    #     print('M=-1:', C[trials.M==-1].mean(axis=0))
    #     print('M=+1:', C[trials.M==1].mean(axis=0))
    all_classifiers = []
    acc = np.zeros((len(s_range), len(s_range)))

    def normed(x):
        return (x - x.min()) / (x.max() - x.min())

    for i_train in range(len(s_range)):
        task_train = all_tasks[i_train]
        trials_train = all_trials_train[i_train]
        classifier = classifier_type(task_train, pairs=True).train(trials_train)
        all_classifiers.append(classifier)
        for i_test in range(len(s_range)):
            trials_test = all_trials_test[i_test]
            acc[i_train, i_test] = classifier.test(trials_test).accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(
        acc,
        origin="lower",
        aspect="equal",
        extent=image_extent_from_linear_range(s_range) * 2,
        vmin=0,
        vmax=1,
        cmap="plasma",
    )
    plt.xlabel("s_test")
    plt.ylabel("s_train")
    plt.colorbar()

    # plt.subplot(132)
    # for i_train in range(len(s_range)):
    #     plt.plot(s_range, acc[i_train, :], label=f's_train={s_range[i_train]}', c=cm.plasma(s_range[i_train]))
    # plt.xlabel('s_test')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='best')

    # plt.subplot(133)
    # weights = np.array([normed(classifier.model.coef_[0, :]) for classifier in all_classifiers])
    # plt.title('Normalised weights')
    # # weights = np.array([classifier.model.coef_[0, :] for classifier in all_classifiers])
    # plt.imshow(weights, origin='lower', aspect='auto', cmap='bwr',
    #            extent=(-0.5, weights.shape[1]-0.5, *image_extent_from_linear_range(s_range)))
    # plt.colorbar()
    # plt.xticks(np.arange(weights.shape[1]),
    #         #    Trials._coincidence_labels,
    #            Trials.joint_count_labels,
    #            rotation=90)
    # plt.ylabel('s_train')

    # # Ideal accuracy plot
    # n = 100
    # task_family = DetectionTaskFamily(n)
    # S = np.linspace(0, 1, 10)
    # plt.axhline(1/3, ls='--', c='k', lw=1, label='baseline')
    # for pairs in [True, False]:
    #     acc = [ideal_accuracy(task_family.task(s), n, pairs=pairs) for s in S]
    #     plt.plot(S, acc, label='FtA' if pairs else 'AtF')
    # plt.legend(loc='best')
    # plt.xlabel('s')
    # plt.ylabel('Ideal accuracy')
    # plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

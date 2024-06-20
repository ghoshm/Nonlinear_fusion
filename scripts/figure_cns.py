import multisensory as ms
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import os
import numpy as np
from scipy.optimize import fsolve
from collections import defaultdict

plt.style.use('./style_sheet.mplstyle')

def get_ideal_data_classical_comod():
    fname = 'figdata/ideal_data_classical_comod.json'
    if not os.path.exists(fname):
        tasks = [
            ms.ClassicalTask(s=0.1),
            ms.BalancedComodulationTask(s=0.2),
        ]
        classifier_type = ms.MAPClassifier
        N = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        repeats = 50000
        accuracies = {}
        for task in tasks:
            all_trials = [task.generate_trials(repeats, n) for n in N]
            all_training_trials = [task.generate_trials(repeats, n) for n in N]
            current_accuracies = {}
            for pairs in [False, True]:
                classifier = classifier_type(task, pairs=pairs)
                results = [classifier.train(training_trials).test(trials) for trials, training_trials in zip(all_trials, all_training_trials)]
                current_accuracies[pairs] = [res.accuracy for res in results]
            accuracies[task.__class__.__name__] = current_accuracies
        data = dict(
            N=N, repeats=repeats, tasks=[task.__class__.__name__ for task in tasks],
            accuracies=accuracies,
        )
        json.dump(data, open(fname, 'w'))
    return json.load(open(fname, 'r'))


def plot_ideal_data_classical_comod():
    data = get_ideal_data_classical_comod()
    N = data['N']
    task_names = data['tasks']
    accuracies = data['accuracies']
    names = {'ClassicalTask': 'Classical', 'BalancedComodulationTask': 'Comodulation'}
    plt.figure(figsize=(8, 4), dpi=100)
    for i, task_name in enumerate(['ClassicalTask', 'BalancedComodulationTask']):
        plt.subplot(1, 2, i+1)
        plt.title(names[task_name])
        plt.plot(N, 0.5*np.ones(len(N)), ls='-', c='lightgray', label='Baseline (chance)')
        for pairs in [True, False]:
            acc = accuracies[task_name][str(pairs).lower()]
            plt.plot(N, acc, c='C'+str(int(pairs)), ls='-' if pairs else '--', label='FtA' if pairs else 'AtF')
        plt.ylim(0.49, 1.01)
        if i==0:
            plt.legend(loc='best')
            plt.ylabel('Accuracy')
        plt.xlabel('Number of time steps')
    plt.tight_layout()
    fname = 'temp/fig_ideal_data_classical_comod'
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf')


def get_ideal_data_detection():
    fname = 'figdata/ideal_data_detection.json'
    if not os.path.exists(fname):
        #task = ms.DetectionTask(pm=0.1, pe=0.05, pc=0.8, pn=0.2, pi=0.01)
        task = ms.DetectionTask(pm=2/3, pe=0.057, pc=0.95, pn=1/3, pi=0.01)
        classifier_type = ms.MAPClassifier
        reward = ms.reward_matrix({
            # pairs (true, guess): reward
            (-1, -1): 10,      (-1,  0): -.1,           (-1,  1): -1,       # prey went left, reward if correct, small penalty if wrong action taken
            ( 0, -1): -1,      ( 0,  0): -.1,           ( 0,  1): -1,       # no prey, small penalty if wrong action taken
            ( 1, -1): -1,      ( 1,  0): -.1,           ( 1,  1): 10,       # prey went right, reward if correct, small penalty if wrong action taken
            })
        N = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        repeats = 50000
        all_trials = [task.generate_trials(repeats, n) for n in N]
        accuracies = {}
        mean_rewards = {}
        confusions = {}
        for do_rewards in [False]:#, True]:
            current_accuracies = {}
            current_mean_rewards = {}
            current_confusions = {}
            for pairs in [False, True]:
                classifier = classifier_type(task, pairs=pairs, reward=reward if do_rewards else None)
                results = [classifier.test(trials) for trials in all_trials]
                current_accuracies[pairs] = [res.accuracy for res in results]
                current_mean_rewards[pairs] = [res.compute_expected_reward(reward) for res in results]
                current_confusions[pairs] = [res.confusions_normalised.tolist() for res in results]
            accuracies[do_rewards] = current_accuracies
            mean_rewards[do_rewards] = current_mean_rewards
            confusions[do_rewards] = current_confusions
        data = dict(
            N=N, repeats=repeats,
            accuracies=accuracies,
            mean_rewards=mean_rewards,
            confusions=confusions,
            baseline=task.baseline,
            baseline_reward=task.baseline_reward(reward),
        )
        json.dump(data, open(fname, 'w'))
    return json.load(open(fname, 'r'))


def plot_ideal_data_detection():
    data = get_ideal_data_detection()
    N = data['N']
    accuracies = data['accuracies']
    mean_rewards = data['mean_rewards']
    confusions = data['confusions']
    plt.figure(figsize=(9, 4), dpi=100)
    # plt.figure(figsize=(5, 4), dpi=100)
    for show_rewards in [False]:#, True]:
        ydata = data['mean_rewards' if show_rewards else 'accuracies']
        plt.subplot(1, 2, show_rewards+1)
        cur_baseline = data['baseline_reward'] if show_rewards else data['baseline']
        if show_rewards:
            plt.plot(N, np.zeros(len(N)), ls='-', c='lightgray', label='Break-even')
        else:
            plt.plot(N, cur_baseline*np.ones(len(N)), ls='-', c='lightgray', label='Baseline')
        for pairs in [True, False]:
            for optimise_for_rewards in [False]:#, True]:
                y = ydata[str(optimise_for_rewards).lower()][str(pairs).lower()]
                label = ('FtA' if pairs else 'AtF')+(' opt. for reward' if optimise_for_rewards else '')
                plt.plot(N, y,
                         c='C'+str(pairs+2*optimise_for_rewards),
                         ls='-' if pairs else '--',
                         label=label if optimise_for_rewards==show_rewards else None,
                         alpha=1 if optimise_for_rewards==show_rewards else 0.3,
                        )
                if optimise_for_rewards and show_rewards:
                    # plot intersection of line with 0
                    f = lambda x: np.interp(x, N, y)
                    N0 = fsolve(f, 10)
                    plt.plot([N0], [f(N0)], 'o', c='C'+str(pairs+2*optimise_for_rewards), ms=10)
        if show_rewards:
            plt.ylim(ymax=0.7)
        else:
            #plt.ylim(0.49, 1.01)
            plt.ylim(0, 1)
        plt.legend(loc='best')
        plt.ylabel('Mean reward' if show_rewards else 'Accuracy')
        plt.xlabel('Number of time steps')
    plt.subplot(1, 2, 2)
    for pairs in [False, True]:
        C = np.array(confusions["false"][str(pairs).lower()]) # shape (N, true, guess)
        for i in range(2):
            acc = C[:, i+1, i+1]
            plt.plot(N, acc, c='C'+str(int(pairs)), ls='--' if i else '-', label=('FtA' if pairs else 'AtF')+' class '+str(i))
    plt.xlabel('Number of time steps')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.tight_layout()
    fname = 'temp/fig_ideal_data_detection'
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf')


def get_ideal_data_detection_task_family():
    fname = 'figdata/ideal_data_detection_task_family.json'
    if not os.path.exists(fname):
        num_steps = 90
        #task_type = ms.DetectionTaskFamily(num_steps, pm=0.1, pn=0.2, pe_range=np.linspace(0, 0.3, 10), pc_range=np.linspace(0.05, 0.95, 10), desired_accuracy=0.96).task
        family = ms.DetectionTaskFamily(num_steps, pm=2/3, pn=1/3, pe_range=np.linspace(0, 0.3, 20), pc_range=np.linspace(1/6, 0.95, 20))
        task_type = family.task
        classifier_type = ms.MAPClassifier
        reward = ms.reward_matrix({
            # pairs (true, guess): reward
            (-1, -1): 10,      (-1,  0): -.1,           (-1,  1): -1,       # prey went left, reward if correct, small penalty if wrong action taken
            ( 0, -1): -1,      ( 0,  0): -.1,           ( 0,  1): -1,       # no prey, small penalty if wrong action taken
            ( 1, -1): -1,      ( 1,  0): -.1,           ( 1,  1): 10,       # prey went right, reward if correct, small penalty if wrong action taken
            })
        repeats = 50000
        s_range = np.linspace(0, 1, 11)
        accuracies = defaultdict(list)
        mean_rewards = defaultdict(list)
        for s in s_range:
            task = task_type(s=1-s)
            trials = task.generate_trials(repeats, num_steps)
            for pairs in [False, True]:
                classifier = classifier_type(task, pairs=pairs)#, reward=reward)
                res = classifier.test(trials)
                accuracies[str(pairs).lower()].append(res.accuracy)
                mean_rewards[str(pairs).lower()].append(res.compute_expected_reward(reward))
        data = dict(
            num_steps=num_steps, repeats=repeats,
            s_range=list(s_range),
            accuracies=accuracies,
            mean_rewards=mean_rewards,
        )
        json.dump(data, open(fname, 'w'))
    return json.load(open(fname, 'r'))


def plot_detection_task_family():
    data = get_ideal_data_detection_task_family()
    s_range = data['s_range']
    accuracies = data['accuracies']
    mean_rewards = data['mean_rewards']
    plt.figure(figsize=(4, 4), dpi=100)
    for pairs in [False, True]:
        # plt.plot(s_range, mean_rewards[str(pairs).lower()], label='FtA' if pairs else 'AtF')
        plt.plot(s_range, accuracies[str(pairs).lower()], label='FtA' if pairs else 'AtF')
    plt.legend(loc='best')
    plt.ylabel('Mean reward')
    plt.xlabel('Signal sparsity')
    plt.xticks([0, 1], ["Sparse", "Dense"])
    plt.tight_layout()
    fname = 'temp/fig_ideal_data_detection_task_family'
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf')


if __name__=='__main__':
    # plot_ideal_data_classical_comod()
    plot_ideal_data_detection()
    # plot_detection_task_family()
    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots_adjust
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import random
import json

random.seed(1027)

plt.rcParams.update({'font.size': 40})
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

pwd = "/home/longtao/Code/SPC_EXP/corner5-new-plot"
results = {}
label_algorithm = {"att_com_hrl_contextual": "SPC", "att_com_uniform": "SPC w. uniform",
                   "att_com_contextual": "SPC w/o. HRL", "ppo_none": "IPPO w/o. teacher",
                   "qmix_none": "QMix w/o. teacher", "shared_ppo_uniform": "IPPO w. uniform"}


def tensorboard_smooth_func(scalars, weight=0.6):
    assert weight >= 0 and weight < 1
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def moving_avg_smooth_func(scalars, window=7):
    assert window % 2 == 1
    return scalars.rolling(window).mean().shift(-int(window / 2))


def corner(metric='win_mean', smooth=True):
    teacher_algo = {
        "alp-gmm-ppo": "ALP-GMM",
        "ppo": "None",
        "uniform-ppo": "Uniform",
        "contextual-bandit-ppo": "Contextual Bandit",
        "bandit-no-context-ppo": "Bandit",
        "vacl-ppo": "VACL",
    }
    results = {}
    for algorithm in teacher_algo.keys():
        exp_path = pwd + "/gfootball-corner-" + algorithm
        if os.path.isdir(exp_path):
            sr = []
            for trial in os.listdir(exp_path):
                trial_path = exp_path + '/' + trial
                steps = []
                values = []
                if os.path.isdir(trial_path):
                    with open(exp_path + '/' + trial + '/' + 'result.json') as f:
                        for line in f:
                            d = json.loads(line)
                            if d["num_env_steps_sampled"] < 1e6:
                                if "evaluation" in d:
                                    steps.append(d["num_env_steps_sampled"])
                                    values.append(d["evaluation"]["custom_metrics"][metric])
                    tmp = pd.DataFrame(list(zip(steps, values)), columns=['steps', 'val'])
                    if smooth:
                        # Smoothing.
                        tmp['metrics'] = tensorboard_smooth_func(tmp['val'], 0.8)
                        # tmp['metrics'] = moving_avg_smooth_func(tmp['val'], 7)
                        sr.append(tmp[['steps', 'metrics']])
                    else:
                        sr.append(tmp)
            merged = sr[0]
            for i in range(1, len(sr)):
                merged = merged.merge(sr[i], how='left', on='steps')
                merged.columns = ['steps', 'val0'] + [f'val{j}' for j in range(1, i + 1)]
            merged = merged.set_index('steps').stack().reset_index(level=1, drop=True).to_frame(metric).reset_index()
            results.update({algorithm: merged})

    fig = plt.figure(figsize=[32, 18])
    for algorithm in results.keys():
        sns.lineplot('steps', metric, data=results[algorithm], label=teacher_algo[algorithm])
    plt.xlabel('Training Timesteps')
    if metric == 'score_mean':
        ylabel = "Goal Diff. in Eval"
    else:
        ylabel = "Win Rate in Eval"
    plt.ylabel(ylabel)
    # plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: '{0:g} M'.format(int(x / 1e6))))
    plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: '1 M' if x == 1e6 else '{:0.1f} M'.format(x / 1e6)))
    # plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: '{0:g} K'.format(int(x / 1e3))))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(True)
    return fig


def plot():
    for i in ['corner', ]:
        metrics = ['win_mean', 'score_mean']
        for metric in metrics:
            pdf = PdfPages(os.path.join(pwd, f'{i}_{metric}.pdf'))
            plot1 = globals()[i](metric)
            plt.show()
            pdf.savefig(plot1)
            pdf.close()


if __name__ == "__main__":
    plot()

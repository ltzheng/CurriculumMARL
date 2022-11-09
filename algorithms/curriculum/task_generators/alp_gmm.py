"""
Absolute Learning Progress - Gaussian Mixture Model (https://arxiv.org/abs/1910.07224).
Code modified from https://github.com/flowersteam/TeachMyAgent/blob/master/TeachMyAgent/teachers/algos/alp_gmm.py.
"""

import logging
import numpy as np
import ray
from ray import cloudpickle as pickle
from sklearn.mixture import GaussianMixture
from collections import deque


def proportional_choice(v, random_state, eps=0.0):
    """
    Return an index of `v` chosen proportionally to values contained in `v`.
    Args:
        v: List of values
        random_state: Random generator
        eps: Epsilon used for an Epsilon-greedy strategy
    """
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]


@ray.remote
class ALPGMMTaskGenerator:
    def __init__(
        self,
        seed,
        num_agents,
        gmm_fitness_func="aic",
        warm_start=False,
        nb_em_init=1,
        fit_rate=250,
        alp_window_size=200,
        potential_ks=np.arange(2, 5, 1),
        random_task_ratio=0.2,
        nb_bootstrap=None,
        initial_dist=None,
        debug=False,
    ):
        """ALP-GMM for discrete task space, i.e. there is no KDTree to find closest historical task.

        Args:
            seed: Random seed.
            gmm_fitness_func: Fitness criterion when selecting the best GMM.
            warm_start: Restart new fit by initializing with last fit.
            nb_em_init: Number of Expectation-Maximization trials when fitting.
            fit_rate: Number of episodes between two fit of the GMM.
            alp_window_size: Size of ALP first-in-first-out window.
            potential_ks: Range of number of Gaussians to try when fitting the GMM.
            random_task_ratio: Ratio of randomly sampled tasks v.s. tasks sampling using GMM.
            nb_bootstrap: Number of bootstrapping episodes, must be >= to fit_rate.
            initial_dist: Initial Gaussian distribution. If None, bootstrap with random tasks.
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.random_state = np.random.RandomState(self.seed)
        self.num_agent_list = num_agents
        self.gmm_fitness_func = gmm_fitness_func
        self.warm_start = warm_start
        self.nb_em_init = nb_em_init
        self.fit_rate = fit_rate
        self.alp_window_size = alp_window_size
        self.potential_ks = potential_ks
        self.random_task_ratio = random_task_ratio
        self.nb_bootstrap = nb_bootstrap if nb_bootstrap is not None else fit_rate
        self.initial_dist = initial_dist
        assert self.nb_bootstrap >= self.fit_rate

        self.alp_window = deque(maxlen=self.alp_window_size)
        self.reward_history = {k: deque(maxlen=self.alp_window_size) for k in self.num_agent_list}

        self.sample_history = list()

        # Init GMMs
        self.potential_gmms = [
            GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=self.seed,
                warm_start=self.warm_start,
                n_init=self.nb_em_init,
            )
            for k in self.potential_ks
        ]
        self.gmm = None

        self.steps = 0
        self.bootstrap = True

        self.infos = {
            "weights": deque(maxlen=self.alp_window_size),
            "covariances": deque(maxlen=self.alp_window_size),
            "means": deque(maxlen=self.alp_window_size),
            "tasks_lps": deque(maxlen=self.alp_window_size),
            "episodes": deque(maxlen=self.alp_window_size),
            "tasks_origin": deque(maxlen=self.alp_window_size),
        }

        self.debug = debug
        if self.debug:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s",
                handlers=[logging.FileHandler(f"ray_results/alp_gmm.log")],
            )

    def episodic_update(self, task, reward):
        """Get the episodic reward of a task."""
        # Compute ALP
        self.reward_history[task].append(reward)
        if len(self.reward_history[task]) >= self.alp_window_size:
            window = int(self.alp_window_size / 2)
            lp = np.mean(list(self.reward_history[task])[-window:]) - np.mean(list(self.reward_history[task])[:window])
            alp = np.abs(lp)
            self.alp_window.append(np.array([task] + [alp]))
        else:
            lp = 0

        if self.bootstrap:
            if self.steps >= self.nb_bootstrap:
                self.bootstrap = False
        elif self.steps % self.fit_rate == 0:
            # Fit batch of GMMs with varying number of Gaussians.
            cur_alp_window = np.array(self.alp_window)
            self.potential_gmms = [
                g.fit(cur_alp_window) for g in self.potential_gmms
            ]
            # Compute fitness and keep best GMM.
            if self.gmm_fitness_func == "bic":  # Bayesian Information Criterion
                fitnesses = [m.bic(cur_alp_window) for m in self.potential_gmms]
            elif self.gmm_fitness_func == "aic":  # Akaike Information Criterion
                fitnesses = [m.aic(cur_alp_window) for m in self.potential_gmms]
            elif self.gmm_fitness_func == "aicc":  # Modified AIC
                n = self.fit_rate
                fitnesses = []
                for l, m in enumerate(self.potential_gmms):
                    k = self.get_nb_gmm_params(m)
                    penalty = (2 * k * (k + 1)) / (n - k - 1)
                    fitnesses.append(m.aic(cur_alp_window) + penalty)
            else:
                raise NotImplementedError

            self.gmm = self.potential_gmms[np.argmin(fitnesses)]

            self.infos["weights"].append(self.gmm.weights_.copy())
            self.infos["covariances"].append(self.gmm.covariances_.copy())
            self.infos["means"].append(self.gmm.means_.copy())
            self.infos["tasks_lps"].append(lp)

        self.steps += 1

    def sample_task(self):
        """Sample a new task."""
        # Bootstrap phase
        if (
            self.bootstrap
            or self.random_state.random() < self.random_task_ratio
            or self.gmm is None
        ):
            if self.initial_dist and self.bootstrap:
                # Expert initial bootstrap Gaussian task sampling.
                new_task = np.random.choice(self.num_agent_list, p=self.initial_dist)
                task_origin = -2
            else:
                # Random task sampling.
                new_task = np.random.choice(self.num_agent_list)
                task_origin = -1
            new_task = new_task

        # ALP-based task sampling.
        else:
            # 1 - Retrieve the mean ALP value of each Gaussian in the GMM.
            self.alp_means = []
            for pos, _, w in zip(
                    self.gmm.means_, self.gmm.covariances_, self.gmm.weights_
            ):
                self.alp_means.append(pos[-1])

            # 2 - Sample Gaussian proportionally to its mean ALP.
            idx = proportional_choice(self.alp_means, self.random_state, eps=0.0)
            task_origin = idx

            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension.
            new_task = self.random_state.multivariate_normal(
                self.gmm.means_[idx], self.gmm.covariances_[idx]
            )[:-1]
            new_task = round(new_task[0])

        new_task = min(self.num_agent_list, key=lambda x: abs(x-new_task))
        self.infos["tasks_origin"].append(task_origin)

        self.sample_history.append(new_task)
        return new_task  # number of agents here

    def get_infos(self):
        infos = {}
        for k in ["tasks_origin", "tasks_lps"]:
            infos[k] = np.mean(self.infos[k])
        return infos

    def task_probs(self):
        task_num = {k: 0 for k in self.num_agent_list}
        for s in self.sample_history:
            task_num[s] += 1
        probs = {k: task_num[k] / len(self.sample_history) for k in self.num_agent_list}
        self.sample_history = list()
        return probs

    def save(self) -> bytes:
        return pickle.dumps(
            {
                "infos": self.infos,
                "alp_window": self.alp_window,
                "reward_history": self.reward_history,
                "gmm": self.gmm,
            }
        )

    def restore(self, objs: bytes) -> None:
        objs = pickle.loads(objs)
        self.infos = objs["infos"]
        self.alp_window = objs["alp_window"]
        self.reward_history = objs["reward_history"]
        self.gmm = objs["gmm"]

    def get_name(self):
        return "ALP-GMM"


def test_alp_gmm():
    num_iterations = 3000
    task_generator = ALPGMMTaskGenerator.options(name="task_generator").remote(
        seed=123,
        num_agents=[1, 2, 3],
        potential_ks=np.arange(2, 4, 1),
    )
    rewards = [10, 0, -10]

    for t in range(num_iterations):
        action = ray.get(task_generator.sample_task.remote())
        reward = rewards[action - 1]
        task_generator.episodic_update.remote(action, reward)
        rewards[0] += 0.01  # simulate learning progress
        if (t + 1) % 500 == 0:
            print(f"Iter {t+1}, probs: {ray.get(task_generator.task_probs.remote())}")


if __name__ == "__main__":
    print("Testing ALP-GMM:")
    ray.init()
    test_alp_gmm()
    ray.shutdown()

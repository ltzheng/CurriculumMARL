import numpy as np
import ray
from ray import cloudpickle as pickle
from sklearn.cluster import Birch


class EXP3(object):
    def __init__(self, gamma, num_tasks, seed, min_rew, max_rew):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.log_weights = np.zeros(self.num_tasks)
        self.max_rew = max_rew
        self.min_rew = min_rew
        self.train_reward_history = []
        self.sample = 0

    @property
    def task_probabilities(self):
        weights = np.exp(self.log_weights - np.sum(self.log_weights))
        probs = (1 - self.gamma) * weights / np.sum(weights) + self.gamma / self.num_tasks
        probs[probs <= 0] = 0.01
        probs /= probs.sum()
        return probs

    def sample_task(self):
        """Samples a task, according to current Exp3 belief."""
        # return np.random.choice(self.num_tasks, p=self.task_probabilities)
        return self.sample

    # def update(self, task_i, reward):
    #     reward = self.rescale_reward(reward)
    #     self.update_weights(self.sample, reward)

    def update_weights(self, task_i, reward):
        reward_corrected = reward / self.task_probabilities[task_i]
        self.log_weights[task_i] += self.gamma * reward_corrected / self.num_tasks

    def rescale_reward(self, reward):
        # reward = max(min(reward, self.max_rew), self.min_rew)
        reward = (reward - self.min_rew) / (self.max_rew - self.min_rew)
        return reward

    def update_train_reward(self, reward):
        """Get the episodic reward of a task."""
        reward = self.rescale_reward(reward)
        self.train_reward_history.append(reward)
    
    def update_eval_reward(self, reward):
        reward = self.rescale_reward(reward)
        tmp_reward = 0.8 * reward + 0.2 * sum(self.train_reward_history) / len(self.train_reward_history)
        self.update_weights(self.sample, tmp_reward)
        self.train_reward_history = []
        # Samples a task, according to current Exp3 belief.
        self.sample = np.random.choice(self.num_tasks, p=self.task_probabilities)


@ray.remote
class ContextualBanditTaskGenerator:
    def __init__(
        self,
        seed,
        num_contexts,
        gamma,
        num_agents,
        min_rew,
        max_rew,
    ):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_agents = num_agents
        self.num_contexts = num_contexts
        self.algo = [
            EXP3(
                gamma=gamma,
                num_tasks=len(self.num_agents),
                seed=self.seed,
                min_rew=min_rew,
                max_rew=max_rew,
            )
            for _ in range(self.num_contexts)
        ]
        self.context_classifier = Birch(n_clusters=self.num_contexts)
        self.context_class = 0
        self.context_history = list()

        # self.alp_window_size = alp_window_size
        # self.reward_history = [
        #     {k: deque(maxlen=self.alp_window_size) for k in self.num_agents}
        #     for _ in range(self.num_contexts)
        # ]

    def episodic_update_train_reward(self, reward):
        """Get the episodic reward of every training task."""
        self.algo[self.context_class].update_train_reward(reward)

    # def episodic_update(self, task, reward):
    #     """Get the episodic reward of every training task."""
    #     # self.algo[self.context_class].update(self.num_agents.index(task), reward)
    #     # Compute ALP
    #     self.reward_history[self.context_class][task].append(reward)
    #     if len(self.reward_history[self.context_class][task]) >= self.alp_window_size:
    #         window = 50
    #         lp = np.mean(list(self.reward_history[self.context_class][task])[-window:]) - \
    #              np.mean(list(self.reward_history[self.context_class][task])[:window])
    #         alp = np.abs(lp)
    #         if len(self.context_history) >= 2 * self.num_contexts:
    #             self.algo[self.context_class].update(self.num_agents.index(task), alp)

    def update_eval_reward(self, reward):
        """Get the evaluation reward after training iterations."""
        self.algo[self.context_class].update_eval_reward(reward)

    def update_context(self, context):
        self.context_history.append(context)
        if len(self.context_history) < 2 * self.num_contexts:
            self.context_class = len(self.context_history) % self.num_contexts
        elif len(self.context_history) == 2 * self.num_contexts:
            self.context_classifier.partial_fit(list(self.context_history))
            self.context_class = len(self.context_history) % self.num_contexts
        else:
            self.context_history.pop(0)
            self.context_classifier.partial_fit([context])
            self.context_class = self.context_classifier.predict([context])[0]

    def context_task_probs(self):
        """Return the task probs under every contexts."""
        probs = []
        for i in range(self.num_contexts):
            probs.append(self.algo[i].task_probabilities)
        return probs

    def sample_task(self):
        if len(self.context_history) < 2 * self.num_contexts:
            sample = np.random.choice(self.num_agents)
        else:
            sample = self.num_agents[self.algo[self.context_class].sample_task()]
        return sample

    def save(self) -> bytes:
        return pickle.dumps(
            {
                "context_classifier": self.context_classifier,
                "context_history": self.context_history,
                "context_class": self.context_class,
                "algo": self.algo,
            }
        )

    def restore(self, objs: bytes) -> None:
        objs = pickle.loads(objs)
        self.context_classifier = objs["context_classifier"]
        self.context_history = objs["context_history"]
        self.context_class = objs["context_class"]
        self.algo = objs["algo"]

    def get_name(self):
        return "contextual-bandit"

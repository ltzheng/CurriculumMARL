"""
Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems (VACL)

Paper: https://arxiv.org/abs/2111.04613
Code: This implementation borrows code from https://github.com/jiayu-ch15/Variational-Automatic-Curriculum-Learning
"""

import ray
import numpy as np
import math
import random
from ray import cloudpickle as pickle
from scipy.spatial.distance import cdist
import copy


def sort_by_novelty(list1, list2, topk: int):
    """Compute distance between each pair of the two lists using Euclidean distance (2-norm).
    The novelty is measured by the sum of the top-K distance.

    Returns:
        The given list sorted by novelty.
    """
    dist = cdist(np.array(list1).reshape(len(list1), -1), np.array(list2).reshape(len(list2), -1), metric='euclidean')
    if len(list2) < topk + 1:
        dist_k = dist
        novelty = np.sum(dist_k, axis=1) / len(list2)
    else:
        dist_k = np.partition(dist, kth=topk + 1, axis=1)[:, 0:topk + 1]
        novelty = np.sum(dist_k, axis=1) / topk

    zipped = zip(list1, novelty)
    sorted_zipped = sorted(zipped, key=lambda x: (x[1], np.mean(x[0])))
    result = zip(*sorted_zipped)
    return [list(x) for x in result][0]


def gradient_of_state(state, buffer, h, use_rbf=True):
    """Compute the gradient of given state w.r.t. the buffer."""
    gradient = np.zeros(state.shape)
    for buffer_state in buffer:
        if use_rbf:
            dist0 = state - np.array(buffer_state).reshape(-1)
            gradient += 2 * dist0 * np.exp(-dist0 ** 2 / h) / h
        else:
            gradient += 2 * (state - np.array(buffer_state).reshape(-1))
    norm = np.linalg.norm(gradient, ord=2)
    if norm > 0.0:
        gradient = gradient / np.linalg.norm(gradient, ord=2)
        gradient_zero = False
    else:
        gradient_zero = True
    return gradient, gradient_zero


@ray.remote
class VACLTaskGenerator:
    """Create a VACL instance with its node buffer w/o Entity Progression."""

    def __init__(
        self,
        seed,
        num_agents,
        solved_prop=0.05,
        buffer_length=2000,
        reproduction_num=150,
        epsilon=0.1,
        delta=0.1,
        h=1.0,
        Rmin=0.5,
        Rmax=0.95,
        del_switch="novelty",
        topk=5,
        num_initial_tasks=1000,
    ):
        random.seed(seed)
        np.random.seed(seed)
        self.solved_prop = solved_prop
        self.num_agents = num_agents
        self.buffer_length = buffer_length
        # Gradient step parameters
        self.reproduction_num = reproduction_num
        self.epsilon = epsilon
        self.delta = delta
        self.h = h
        # VACL parameters
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.del_switch = del_switch
        self.topk = topk
        self.num_initial_tasks = num_initial_tasks

        self.active_task_buffer = [[1 / len(self.num_agents)] * len(self.num_agents)] * self.num_initial_tasks
        self.active_task_buffer = sort_by_novelty(self.active_task_buffer, self.active_task_buffer, self.topk)
        self.all_solved_task_buffer = []
        self.newly_solved_tasks = []
        self.solved_tasks_indices = []
        self.active_tasks_indices = []
        self.current_samples = None
        self.num_sampled_active_tasks = 0
        self.stats = {}

        self.sample_task()
        self.training_scores = []  # scores of tasks with next num_agents

    def update_buffer(self):
        """Update the node buffer. VACL teacher only collects scores of active tasks.
            1) Task Expansion:
                a) Add solved tasks to all_solved_task_buffer, and delete them from active_task_buffer.
                b) Delete non-active tasks if active_task_buffer is full.
            2) Maintain buffer size with given criteria.
        """
        total_del_num = 0
        del_easy_num = 0
        del_hard_num = 0

        # Task expansion.
        self.newly_solved_tasks = []
        for (i, score) in self.training_scores:
            if score > self.Rmax:  # solved
                self.newly_solved_tasks.append(copy.deepcopy(self.active_task_buffer[
                                                            self.active_tasks_indices[i] - total_del_num]))
                del self.active_task_buffer[self.active_tasks_indices[i] - total_del_num]
                total_del_num += 1
                del_easy_num += 1
            elif score < self.Rmin:  # unsolved and buffer is full
                if len(self.active_task_buffer) >= self.buffer_length:
                    del self.active_task_buffer[self.active_tasks_indices[i] - total_del_num]
                    total_del_num += 1
                    del_hard_num += 1

        # Maintain buffer size.
        self.all_solved_task_buffer += self.newly_solved_tasks
        if len(self.active_task_buffer) > self.buffer_length:
            if self.del_switch == 'novelty':  # novelty deletion (for diversity)
                self.active_task_buffer = sort_by_novelty(
                    self.active_task_buffer, self.active_task_buffer, self.topk)[
                                          len(self.active_task_buffer) - self.buffer_length:]
            elif self.del_switch == 'random':  # random deletion
                del_num = len(self.active_task_buffer) - self.buffer_length
                del_index = random.sample(range(len(self.active_task_buffer)), del_num)
                del_index = np.sort(del_index)
                total_del_num = 0
                for i in range(del_num):
                    del self.active_task_buffer[del_index[i] - total_del_num]
                    total_del_num += 1
            else:  # FIFO queue deletion
                self.active_task_buffer = self.active_task_buffer[len(self.active_task_buffer) - self.buffer_length:]
        if len(self.all_solved_task_buffer) > self.buffer_length:
            self.all_solved_task_buffer = self.all_solved_task_buffer[
                                     len(self.all_solved_task_buffer) - self.buffer_length:]

        self.stats = {
            "len_active_task_buffer": len(self.active_task_buffer),
            "num_newly_solved_tasks": len(self.newly_solved_tasks),
            "del_easy_num": del_easy_num,
            "del_hard_num": del_hard_num,
            "num_sampled_active_tasks": self.num_sampled_active_tasks
        }
        self.training_scores = []

    def compute_gradient(self, use_gradient_noise=True):
        """Compute gradients and add new tasks to buffer.

        Args:
            use_gradient_noise: for exploration purpose.
        """
        if self.newly_solved_tasks:  # not all sampled tasks unsolved in the last episode
            for _ in range(self.reproduction_num):
                for task in self.newly_solved_tasks:
                    gradient, gradient_zero = gradient_of_state(np.array(task).reshape(-1), self.all_solved_task_buffer,
                                                                self.h)
                    assert len(task) == len(self.num_agents)
                    probs = copy.deepcopy(task)
                    for i in range(len(task)):
                        # Execute gradient step.
                        if not gradient_zero:
                            probs[i] += gradient[i] * self.epsilon
                        else:
                            probs[i] += -2 * self.epsilon * random.random() + self.epsilon
                        # Rejection sampling
                        if use_gradient_noise:
                            probs[i] += -2 * self.delta * random.random() + self.delta
                    # Softmax
                    probs = list(np.exp(probs) / np.sum(np.exp(probs)))
                    self.active_task_buffer.append(probs)

    def sample_task(self):
        """Sample tasks from node buffer after sampling gradients."""
        if random.random() < self.solved_prop and len(self.all_solved_task_buffer) > 0:
            # Sample solved tasks to avoid forgetting.
            i = np.random.choice(len(self.all_solved_task_buffer))
            sample = self.all_solved_task_buffer[i]
            solved = True
        else:
            i = np.random.choice(len(self.active_task_buffer))
            sample = np.random.choice(self.num_agents, p=self.active_task_buffer[i])
            solved = False
        return i, solved, sample

    def episodic_update(self, i, solved, score):
        if not solved:
            self.training_scores.append((i, score))

    def save(self) -> bytes:
        return pickle.dumps(
            {
                "active_task_buffer": self.active_task_buffer,
                "all_solved_task_buffer": self.all_solved_task_buffer,
                "newly_solved_tasks": self.newly_solved_tasks,
                "solved_tasks_indices": self.solved_tasks_indices,
                "active_tasks_indices": self.active_tasks_indices,
                "current_samples": self.current_samples,
                "num_sampled_active_tasks": self.num_sampled_active_tasks,
                "stats": self.stats,
                "training_scores": self.training_scores,
            }
        )

    def restore(self, objs: bytes) -> None:
        objs = pickle.loads(objs)
        self.active_task_buffer = objs["active_task_buffer"]
        self.all_solved_task_buffer = objs["all_solved_task_buffer"]
        self.newly_solved_tasks = objs["newly_solved_tasks"]
        self.solved_tasks_indices = objs["solved_tasks_indices"]
        self.active_tasks_indices = objs["active_tasks_indices"]
        self.current_samples = objs["current_samples"]
        self.num_sampled_active_tasks = objs["num_sampled_active_tasks"]
        self.stats = objs["stats"]
        self.training_scores = objs["training_scores"]

    def get_infos(self):
        return self.stats

    def get_name(self):
        return "vacl"

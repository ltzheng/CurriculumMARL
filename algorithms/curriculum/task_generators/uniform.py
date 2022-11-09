"""
A teacher uniformly samples over the task space.
"""

import numpy as np
import ray


@ray.remote
class UniformTaskGenerator:
    def __init__(
        self,
        seed,
        num_agents,
    ):
        self._seed = seed
        np.random.seed(self._seed)
        self._tasks = num_agents

    def sample_task(self):
        return np.random.choice(self._tasks)

    def get_name(self):
        return "uniform"

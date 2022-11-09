import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class BaseMultiAgentHierarchicalEnv(MultiAgentEnv):
    """Wraps env to be compatible with (multi-agent) context-based hierarchical RL.

    This class supports 2-layer abstraction with fixed high-level interval.
    However, it is easy to extend to a multi-layer setting and add a terminal function.
    High-level policy should be named for "high_level_policy".
    Low-level policy should be named for "agent_{i}".
    """

    def __init__(
        self,
        num_agents,
        high_level_interval,
        context_type,
        context_size,
    ):
        # HRL configurations.
        self.num_agents = num_agents
        self.high_level_interval = high_level_interval
        self.context_type = context_type
        self.context_size = context_size

        # Observation and action spaces.
        self._env = None
        self.high_level_observation_space = None
        self.high_level_action_space = None
        self.low_level_observation_space = None
        self.low_level_action_space = None

        # Book-keeping for transition between high and low level policies
        self.context = None
        self.env_obs = None
        self.env_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.env_info = None

        # Book-keeping for high level policy within a macro-step
        self.steps = 0
        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        super().__init__()

    def reset(self, **kwargs):
        """Reset environment and return high level observations."""
        self.env_obs = self._env.reset(**kwargs)
        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        return self.high_level_obs

    def step(self, action_dict):
        policy_name = list(action_dict.keys())[0]
        if policy_name.startswith("agent_"):
            return self._low_level_step(action_dict)
        elif policy_name.startswith("high_level_"):
            return self._high_level_step(action_dict)
        else:
            raise ValueError(f"Unsupported policy name: {policy_name}.")

    def _high_level_step(self, action_dict):
        # context encoding
        actions = self.high_level_actions(action_dict)
        if self.context_type == "discrete":  # one-hot encoding
            one_hot_context = np.zeros((len(actions), self.context_size))
            for i, act in enumerate(actions):
                one_hot_context[i, act] = 1
            self.context = one_hot_context
        else:
            self.context = actions

        self.low_level_accumulated_rew = {i: 0.0 for i in range(self.num_agents)}
        obs = self.low_level_obs
        rew = self.low_level_rewards
        done = {"__all__": False}
        info = self.low_level_infos

        return obs, rew, done, info

    def _low_level_step(self, action_dict):
        self.env_obs, self.env_rewards, done, self.env_info = self._env.step(
            self.low_level_actions(action_dict)
        )
        for i in range(self.num_agents):
            self.low_level_accumulated_rew[i] += self.env_rewards[i]
        self.steps += 1

        # Handle env termination & transitions back to higher level
        if done or self.steps == self.high_level_interval:
            self.steps = 0
            obs = self.high_level_obs
            reward = self.high_level_rewards
            infos = self.high_level_infos
        else:
            obs = self.low_level_obs
            reward = self.low_level_rewards
            infos = self.low_level_infos

        done = {"__all__": done}

        return obs, reward, done, infos

    @property
    def low_level_obs(self):
        return {
            f"agent_{i}": np.concatenate((self.env_obs[i], self.context[i]))
            for i in range(self.num_agents)
        }

    def low_level_actions(self, action_dict):
        return [action_dict[f"agent_{i}"] for i in range(self.num_agents)]

    @property
    def low_level_infos(self):
        return {f"agent_{i}": {} for i in range(self.num_agents)}

    @property
    def low_level_rewards(self):
        return {f"agent_{i}": self.env_rewards[i] for i in range(self.num_agents)}

    @property
    def high_level_obs(self):
        raise NotImplementedError

    def high_level_actions(self, action_dict):
        raise NotImplementedError

    @property
    def high_level_rewards(self):
        raise NotImplementedError

    @property
    def high_level_infos(self):
        raise NotImplementedError

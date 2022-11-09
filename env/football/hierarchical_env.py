import gym
import numpy as np

from env.hierarchical_env import BaseMultiAgentHierarchicalEnv
from env.football.base_env import BaseFootballEnv


class FootballPvEHierarchicalEnv(BaseMultiAgentHierarchicalEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent."""

    def __init__(self, **kwargs):
        hrl_config = kwargs.pop("hrl_config")
        hrl_config["num_agents"] = kwargs["number_of_left_players_agent_controls"]
        BaseMultiAgentHierarchicalEnv.__init__(self, **hrl_config)

        self._env = BaseFootballEnv(**kwargs)

        # high-level obs and act spaces
        self.high_level_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(115,)
        )
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.context_size,)
            )
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Discrete(self.context_size)
        else:
            raise NotImplementedError("Unsupported high-level action space.")

        # low-level obs and act spaces
        self.low_level_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(115 + self.context_size,)
        )
        self.low_level_action_space = gym.spaces.Discrete(19)

    def reset(self, **kwargs):
        return BaseMultiAgentHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        return BaseMultiAgentHierarchicalEnv.step(self, actions)

    @property
    def high_level_obs(self):
        return {f"high_level_{i}": self.env_obs[i] for i in range(self.num_agents)}

    def high_level_actions(self, action_dict):
        return [action_dict[f"high_level_{i}"] for i in range(self.num_agents)]

    @property
    def high_level_rewards(self):
        return {f"high_level_{i}": self.low_level_accumulated_rew[i] for i in range(self.num_agents)}

    @property
    def high_level_infos(self):
        return {f"high_level_{i}": self.env_info for i in range(self.num_agents)}


class FootballPvEHierarchicalComEnv(BaseMultiAgentHierarchicalEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent."""

    def __init__(self, **kwargs):
        hrl_config = kwargs.pop("hrl_config")
        hrl_config["num_agents"] = kwargs["number_of_left_players_agent_controls"]
        BaseMultiAgentHierarchicalEnv.__init__(self, **hrl_config)

        self._env = BaseFootballEnv(**kwargs)

        # high-level obs and act spaces
        self.high_level_observation_space = gym.spaces.Tuple([gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(115,)
        ) for _ in range(self.num_agents)])
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Tuple([gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.context_size,)
            ) for _ in range(self.num_agents)])
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.context_size) for _ in range(self.num_agents)])
        else:
            raise NotImplementedError("Unsupported high-level action space.")

        # low-level obs and act spaces
        self.low_level_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(115 + self.context_size,)
        )
        self.low_level_action_space = gym.spaces.Discrete(19)

    def reset(self, **kwargs):
        return BaseMultiAgentHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        return BaseMultiAgentHierarchicalEnv.step(self, actions)

    @property
    def high_level_obs(self):
        return {"high_level_policy": [self.env_obs[i] for i in range(self.num_agents)]}

    def high_level_actions(self, action_dict):
        return list(action_dict["high_level_policy"])

    @property
    def high_level_rewards(self):
        return {"high_level_policy": sum(self.low_level_accumulated_rew.values())}

    @property
    def high_level_infos(self):
        rew_list = [self.low_level_accumulated_rew[i] for i in range(self.num_agents)]
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        self.env_info["rewards"] = rewards
        self.env_info["num_agents"] = self.num_agents
        return {"high_level_policy": self.env_info}

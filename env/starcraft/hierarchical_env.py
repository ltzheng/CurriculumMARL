from gym.spaces import Discrete, Box, Dict, Tuple
import numpy as np
import re

from env.hierarchical_env import BaseMultiAgentHierarchicalEnv
from env.starcraft.base_env import BaseStarCraft2Env


class StarCraft2PvEHierarchicalEnv(BaseMultiAgentHierarchicalEnv):
    def __init__(self, **kwargs):
        hrl_config = kwargs.pop("hrl_config")
        hrl_config["num_agents"] = int(re.findall("\d+", kwargs["map_name"])[0])
        BaseMultiAgentHierarchicalEnv.__init__(self, **hrl_config)
        self._env = BaseStarCraft2Env(**kwargs)

        # high-level obs and act spaces
        self.high_level_observation_space = Box(-1, 1, shape=(self._env.get_obs_size(),))
        if self.context_type == "continuous":
            self.high_level_action_space = Box(
                low=-1.0, high=1.0, shape=(self.context_size,)
            )
        elif self.context_type == "discrete":
            self.high_level_action_space = Discrete(self.context_size)
        else:
            raise NotImplementedError("Unsupported high-level action space.")

        # low-level obs and act spaces
        self.low_level_observation_space = Dict({
            "observations": Box(-1, 1, shape=(self._env.get_obs_size() + self.context_size,)),
            "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
        })
        self.low_level_action_space = Discrete(self._env.get_total_actions())

    def reset(self, **kwargs):
        return BaseMultiAgentHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        return BaseMultiAgentHierarchicalEnv.step(self, actions)

    @property
    def high_level_obs(self):
        return {f"high_level_{i}": self.env_obs[i]["observations"] for i in range(self.num_agents)}

    def high_level_actions(self, action_dict):
        return [action_dict[f"high_level_{i}"] for i in range(self.num_agents)]

    @property
    def high_level_rewards(self):
        return {f"high_level_{i}": self.low_level_accumulated_rew[i] for i in range(self.num_agents)}

    @property
    def high_level_infos(self):
        return {f"high_level_{i}": self.env_info for i in range(self.num_agents)}

    @property
    def low_level_obs(self):
        return {
            f"agent_{i}": {
                "action_mask": self.env_obs[i]["action_mask"],
                "observations": np.concatenate((self.env_obs[i]["observations"], self.context[i])),
            } for i in range(self.num_agents)
        }


class StarCraft2PvEHierarchicalComEnv(BaseMultiAgentHierarchicalEnv):
    def __init__(self, **kwargs):
        hrl_config = kwargs.pop("hrl_config")
        hrl_config["num_agents"] = int(re.findall("\d+", kwargs["map_name"])[0])
        BaseMultiAgentHierarchicalEnv.__init__(self, **hrl_config)
        self._env = BaseStarCraft2Env(**kwargs)

        # high-level obs and act spaces
        self.high_level_observation_space = Tuple([
            Box(-1, 1, shape=(self._env.get_obs_size(),)) for _ in range(self.num_agents)
        ])
        if self.context_type == "continuous":
            self.high_level_action_space = Tuple([Box(
                low=-1.0, high=1.0, shape=(self.context_size,)
            ) for _ in range(self.num_agents)])
        elif self.context_type == "discrete":
            self.high_level_action_space = Tuple([Discrete(self.context_size) for _ in range(self.num_agents)])
        else:
            raise NotImplementedError("Unsupported high-level action space.")

        # low-level obs and act spaces
        self.low_level_observation_space = Dict({
            "observations": Box(-1, 1, shape=(self._env.get_obs_size() + self.context_size,)),
            "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
        })
        self.low_level_action_space = Discrete(self._env.get_total_actions())

    def reset(self, **kwargs):
        return BaseMultiAgentHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        return BaseMultiAgentHierarchicalEnv.step(self, actions)

    @property
    def high_level_obs(self):
        return {"high_level_policy": [self.env_obs[i]["observations"] for i in range(self.num_agents)]}

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

    @property
    def low_level_obs(self):
        return {
            f"agent_{i}": {
                "action_mask": self.env_obs[i]["action_mask"],
                "observations": np.concatenate((self.env_obs[i]["observations"], self.context[i])),
            } for i in range(self.num_agents)
        }

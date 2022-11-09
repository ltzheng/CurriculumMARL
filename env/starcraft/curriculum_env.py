import ray
import gym
from gym.spaces import Discrete, Box, Dict
import copy
import numpy as np
import re

from env.starcraft.base_env import BaseStarCraft2Env
from env.starcraft.multi_agent_env import StarCraft2PvEEnv
from env.starcraft.hierarchical_env import StarCraft2PvEHierarchicalEnv, StarCraft2PvEHierarchicalComEnv
from ray.rllib.utils.spaces.repeated import Repeated


SC_MAPS = {
    3: "3m",
    5: "5m_vs_6m",
    8: "8m",
    10: "10m_vs_11m",
    25: "25m",
    27: "27m_vs_30m",
}


OPPO_NUM = {
    3: 3,
    5: 6,
    8: 8,
    10: 11,
    25: 25,
    27: 30,
}


def curriculum_reset(cls):
    task_generator = ray.get_actor("task_generator")
    name = ray.get(task_generator.get_name.remote())
    if name in ["vacl"]:
        cls.vacl_idx, cls.vacl_solved, cls.num_agents = ray.get(task_generator.sample_task.remote())
    else:
        cls.num_agents = ray.get(task_generator.sample_task.remote())


def curriculum_on_episode_end(cls, rew_list):
    task_generator = ray.get_actor("task_generator")
    name = ray.get(task_generator.get_name.remote())
    if name in ["ALP-GMM", "contextual-bandit"]:
        if name in ["contextual-bandit"]:
            task_generator.episodic_update_train_reward.remote(np.mean(rew_list))
        if name in ["ALP-GMM"]:
            task_generator.episodic_update.remote(cls.num_agents, np.mean(rew_list))
        if name in ["vacl"]:
            task_generator.episodic_update.remote(cls.vacl_idx, cls.vacl_solved, np.mean(rew_list))


# def get_obs_size(cls):
#     # features: move(4), enemy(5*num), ally(5*num), own(1)
#     obs_size = (cls.max_num_agents + OPPO_NUM[cls.max_num_agents]) * 5
#     return obs_size


# def get_total_actions(cls):
#     return 6 + OPPO_NUM[cls.max_num_agents]


# def get_avail_agent_actions(cls, i):
#     # no-op, stop, move north, south, east, west, attack
#     avail_actions = cls._env.get_avail_agent_actions(i)
#     return avail_actions
#     # padded_actions = [0] * (OPPO_NUM[cls.max_num_agents] - OPPO_NUM[cls.num_agents])
#     # return avail_actions + padded_actions


# def pad_agent_obs(cls, orig_obs):
#     orig_obs = list(orig_obs)
#     move_feats = orig_obs[:4]
#     enemy_feats = orig_obs[4:5*cls.num_agents+4]+[-1]*5*(OPPO_NUM[cls.max_num_agents]-OPPO_NUM[cls.num_agents])
#     ally_feats = orig_obs[5*cls.num_agents+4:-1]+[-1]*5*(cls.max_num_agents-cls.num_agents)
#     own_feats = [orig_obs[-1]]
#     return np.array(move_feats + enemy_feats + ally_feats + own_feats)


class StarCraft2CurriculumPvEEnv(StarCraft2PvEEnv):
    """Env for curriculum + PPO parameter sharing."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.num_agents = int(re.findall("\d+", kwargs["map_name"])[0])
        self.in_evaluation = kwargs.pop("in_evaluation")
        self.env_config = copy.deepcopy(kwargs)
        StarCraft2PvEEnv.__init__(self, **kwargs)

    def reset(self, **kwargs):
        if not self.in_evaluation:
            curriculum_reset(self)
        obs_list = BaseStarCraft2Env.reset(self, **kwargs)
        return self.group_items(obs_list)

    def step(self, actions: dict):
        act = self.ungroup_items(actions)
        padded_action = act + [-100] * (self.max_num_agents - self.num_agents)
        obs_list, rew_list, done, info = BaseStarCraft2Env.step(self, padded_action)
        # for i in range(self.num_agents):
        #     orig_obs = list(obs_list[i]["observations"])
        #     obs_list[i]["observations"] = self.pad_agent_obs(orig_obs)

        if not self.in_evaluation and done:
            curriculum_on_episode_end(self, rew_list)
        done = {"__all__": done}
        info["num_agents"] = self.num_agents

        return (
            self.group_items(obs_list),
            self.group_items(rew_list),
            done,
            self.group_items(info),
        )


class StarCraft2CurriculumPvEComEnv(BaseStarCraft2Env):
    """Env for curriculum + PPO + attention communication."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.num_agents = int(re.findall("\d+", kwargs["map_name"])[0])
        self.in_evaluation = kwargs.pop("in_evaluation")
        self.env_config = copy.deepcopy(kwargs)
        BaseStarCraft2Env.__init__(self, **kwargs)
        self.setup_space()

    def setup_space(self):
        self.action_space = gym.spaces.Tuple([Discrete(self.get_total_actions()) for _ in range(self.max_num_agents)]
        )
        self.observation_space = Repeated(Dict({
                "observations": Box(-1, 1, shape=(self.get_obs_size(),)),
                "action_mask": Box(0, 1, shape=(self.get_total_actions(),)),
            }), max_len=self.max_num_agents
        )

    def reset(self, **kwargs):
        if not self.in_evaluation:
            curriculum_reset(self)
            self.setup_space()
        obs_list = BaseStarCraft2Env.reset(self, **kwargs)
        # for i in range(self.num_agents):
        #     orig_obs = list(obs_list[i]["observations"])
        #     obs_list[i]["observations"] = self.pad_agent_obs(orig_obs)
        return [obs_list[i] for i in range(self.num_agents)]

    def step(self, actions):
        # deal with unavailable agents
        act = list(actions)[:self.num_agents]
        padded_action = act + [-100] * (self.max_num_agents - self.num_agents)
        obs_list, rew_list, done, info = BaseStarCraft2Env.step(self, padded_action)
        # for i in range(self.num_agents):
        #     orig_obs = list(obs_list[i]["observations"])
        #     obs_list[i]["observations"] = self.pad_agent_obs(orig_obs)

        if not self.in_evaluation and done:
            curriculum_on_episode_end(self, rew_list)

        mean_rewards = sum(rew_list) / len(rew_list)
        rew_list = rew_list + [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info["rewards"] = rewards
        info["num_agents"] = self.num_agents

        return [obs_list[i] for i in range(self.num_agents)], mean_rewards, done, info


class StarCraft2CurriculumPvEHierarchicalEnv(StarCraft2PvEHierarchicalEnv):
    """Env for curriculum + PPO parameter sharing + HRL."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.num_agents = int(re.findall("\d+", kwargs["map_name"])[0])
        self.in_evaluation = kwargs.pop("in_evaluation")
        self.env_config = copy.deepcopy(kwargs)
        StarCraft2PvEHierarchicalEnv.__init__(self, **kwargs)
        self.setup_space()

    def setup_space(self):
        # high-level obs and act spaces
        self.high_level_observation_space = Box(-1, 1, shape=(self.get_obs_size(),))
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
            "observations": Box(-1, 1, shape=(self.get_obs_size() + self.context_size,)),
            "action_mask": Box(0, 1, shape=(self.get_total_actions(),)),
        })
        self.low_level_action_space = Discrete(self.get_total_actions())

    def reset(self, **kwargs):
        if not self.in_evaluation:
            curriculum_reset(self)
            self.setup_space()
        return StarCraft2PvEHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        policy_name = list(actions.keys())[0]
        if policy_name.startswith("agent_"):
            for i in range(self.num_agents, self.max_num_agents):
                actions[f"agent_{i}"] = -100
        obs, rew, done, info = StarCraft2PvEHierarchicalEnv.step(self, actions)
        if not self.in_evaluation and done["__all__"]:
            curriculum_on_episode_end(self, list(rew.values()))
        return obs, rew, done, info

    @property
    def high_level_obs(self):
        obs = copy.deepcopy(self.env_obs)
        # for i in range(self.num_agents):
        #     orig_obs = list(self.env_obs[i]["observations"])
        #     obs[i]["observations"] = self.pad_agent_obs(orig_obs)
        return {f"high_level_{i}": obs[i]["observations"] for i in range(self.num_agents)}

    @property
    def low_level_obs(self):
        obs = copy.deepcopy(self.env_obs)
        # for i in range(self.num_agents):
        #     orig_obs = list(self.env_obs[i]["observations"])
        #     obs[i]["observations"] = self.pad_agent_obs(orig_obs)
        return {
            f"agent_{i}": {
                "action_mask": self.get_avail_agent_actions(i),
                "observations": np.concatenate((obs[i]["observations"], self.context[i])),
            } for i in range(self.num_agents)
        }

    @property
    def high_level_infos(self):
        self.env_info["num_agents"] = self.num_agents
        return {f"high_level_{i}": self.env_info for i in range(self.num_agents)}


class StarCraft2CurriculumPvEHierarchicalComEnv(StarCraft2PvEHierarchicalComEnv):
    """Env for curriculum + PPO attention communication + HRL."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.num_agents = int(re.findall("\d+", kwargs["map_name"])[0])
        self.in_evaluation = kwargs.pop("in_evaluation")
        self.env_config = copy.deepcopy(kwargs)
        StarCraft2PvEHierarchicalComEnv.__init__(self, **kwargs)
        self.setup_space()

    def setup_space(self):
        # high-level obs and act spaces
        self.high_level_observation_space = Repeated(
            Box(-1, 1, shape=(self.get_obs_size(),)), max_len=self.max_num_agents
        )
        if self.context_type == "continuous":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Box(low=-1.0, high=1.0, shape=(self.context_size,))
                 for _ in range(self.max_num_agents)]
            )
        elif self.context_type == "discrete":
            self.high_level_action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(self.context_size)
                 for _ in range(self.max_num_agents)]
            )
        else:
            raise NotImplementedError("Unsupported high-level action space.")

        # low-level obs and act spaces
        self.low_level_observation_space = Dict({
            "observations": Box(-1, 1, shape=(self.get_obs_size() + self.context_size,)),
            "action_mask": Box(0, 1, shape=(self.get_total_actions(),)),
        })
        self.low_level_action_space = Discrete(self.get_total_actions())

    def reset(self, **kwargs):
        if not self.in_evaluation:
            curriculum_reset(self)
            self.setup_space()
        return StarCraft2PvEHierarchicalComEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        policy_name = list(actions.keys())[0]
        if policy_name.startswith("agent_"):
            for i in range(self.num_agents, self.max_num_agents):
                actions[f"agent_{i}"] = -100
        obs, rew, done, info = StarCraft2PvEHierarchicalComEnv.step(self, actions)
        if not self.in_evaluation and done["__all__"]:
            curriculum_on_episode_end(self, list(rew.values()))
        return obs, rew, done, info

    @property
    def high_level_obs(self):
        obs = copy.deepcopy(self.env_obs)
        # for i in range(self.num_agents):
        #     orig_obs = list(self.env_obs[i]["observations"])
        #     obs[i]["observations"] = self.pad_agent_obs(orig_obs)
        return {"high_level_policy": [obs[i]["observations"] for i in range(self.num_agents)]}

    @property
    def low_level_obs(self):
        obs = copy.deepcopy(self.env_obs)
        # for i in range(self.num_agents):
        #     orig_obs = list(self.env_obs[i]["observations"])
        #     obs[i]["observations"] = self.pad_agent_obs(orig_obs)
        return {
            f"agent_{i}": {
                "action_mask": self.get_avail_agent_actions(i),
                "observations": np.concatenate((obs[i]["observations"], self.context[i])),
            } for i in range(self.num_agents)
        }

    def high_level_actions(self, action_dict):
        return StarCraft2PvEHierarchicalComEnv.high_level_actions(self, action_dict)[:self.num_agents]

    @property
    def high_level_infos(self):
        rew_list = [self.low_level_accumulated_rew[i] for i in range(self.num_agents)] + \
                   [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        self.env_info["rewards"] = rewards
        self.env_info["num_agents"] = self.num_agents
        return {"high_level_policy": self.env_info}

import ray
import gym
import copy
import numpy as np

from env.football.base_env import BaseFootballEnv
from env.football.multi_agent_env import FootballPvEEnv
from env.football.hierarchical_env import FootballPvEHierarchicalEnv, FootballPvEHierarchicalComEnv
from ray.rllib.utils.spaces.repeated import Repeated


def curriculum_reset(cls, parent_cls):
    task_generator = ray.get_actor("task_generator")
    name = ray.get(task_generator.get_name.remote())
    if name in ["vacl"]:
        cls.vacl_idx, cls.vacl_solved, cls.num_agents = ray.get(task_generator.sample_task.remote())
    else:
        cls.num_agents = ray.get(task_generator.sample_task.remote())
    cls.env_config["number_of_left_players_agent_controls"] = cls.num_agents
    cls.close()
    parent_cls.__init__(cls, **cls.env_config)


def curriculum_on_episode_end(cls, episode_score):
    task_generator = ray.get_actor("task_generator")
    name = ray.get(task_generator.get_name.remote())
    if name in ["contextual-bandit"]:
        # task_generator.episodic_update.remote(cls.num_agents, episode_score)
        task_generator.episodic_update_train_reward.remote(episode_score)
    if name in ["ALP-GMM"]:
        task_generator.episodic_update.remote(cls.num_agents, episode_score)
    if name in ["vacl"]:
        task_generator.episodic_update.remote(cls.vacl_idx, cls.vacl_solved, episode_score)


class FootballCurriculumPvEEnv(FootballPvEEnv):
    """Env for curriculum + PPO parameter sharing."""
    def __init__(self, **kwargs):
        self.env_config = copy.deepcopy(kwargs)
        FootballPvEEnv.__init__(self, **kwargs)

    def reset(self, **kwargs):
        if not self.env_config["in_evaluation"]:
            curriculum_reset(self, FootballPvEEnv)
        return FootballPvEEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        act = self.ungroup_items(actions)
        obs_list, rew_list, done, info = BaseFootballEnv.step(self, act)

        if not self.env_config["in_evaluation"] and done:
            curriculum_on_episode_end(self, info["score"])

        done = {"__all__": done}
        info["num_agents"] = self.num_agents

        return (
            self.group_items(obs_list),
            self.group_items(rew_list),
            done,
            self.group_items(info),
        )


class FootballCurriculumPvEComEnv(BaseFootballEnv):
    """Env for curriculum + PPO + attention communication."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.env_config = copy.deepcopy(kwargs)
        BaseFootballEnv.__init__(self, **kwargs)
        self.num_agents = kwargs["number_of_left_players_agent_controls"]
        self.setup_space()

    def setup_space(self):
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(19) for _ in range(self.max_num_agents)]
        )
        self.observation_space = Repeated(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)), max_len=self.max_num_agents
        )

    def reset(self, **kwargs):
        if not self.env_config["in_evaluation"]:
            curriculum_reset(self, BaseFootballEnv)
            self.setup_space()
        obs_list = BaseFootballEnv.reset(self, **kwargs)
        return [obs_list[i] for i in range(self.num_agents)]

    def step(self, actions):
        # deal with unavailable agents
        act = list(actions)[:self.num_agents]
        obs_list, rew_list, done, info = BaseFootballEnv.step(self, act)

        if not self.env_config["in_evaluation"] and done:
            curriculum_on_episode_end(self, info["score"])

        all_rewards = sum(rew_list)
        rew_list = rew_list.tolist() + [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info["rewards"] = rewards
        info["num_agents"] = self.num_agents

        return [obs_list[i] for i in range(self.num_agents)], all_rewards, done, info


class FootballCurriculumPvEHierarchicalEnv(FootballPvEHierarchicalEnv):
    """Env for curriculum + PPO parameter sharing + HRL."""
    def __init__(self, **kwargs):
        self.env_config = copy.deepcopy(kwargs)
        FootballPvEHierarchicalEnv.__init__(self, **kwargs)

    def reset(self, **kwargs):
        if not self.env_config["in_evaluation"]:
            curriculum_reset(self, FootballPvEHierarchicalEnv)
        return FootballPvEHierarchicalEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        obs, rew, done, info = FootballPvEHierarchicalEnv.step(self, actions)

        if not self.env_config["in_evaluation"] and done["__all__"]:
            curriculum_on_episode_end(self, list(info.values())[0]["score"])

        return obs, rew, done, info

    @property
    def high_level_infos(self):
        self.env_info["num_agents"] = self.num_agents
        return {f"high_level_{i}": self.env_info for i in range(self.num_agents)}


class FootballCurriculumPvEHierarchicalComEnv(FootballPvEHierarchicalComEnv):
    """Env for curriculum + PPO attention communication + HRL."""
    def __init__(self, **kwargs):
        self.max_num_agents = kwargs.pop("max_num_agents")
        self.env_config = copy.deepcopy(kwargs)
        FootballPvEHierarchicalComEnv.__init__(self, **kwargs)
        self.setup_space()

    def setup_space(self):
        self.high_level_observation_space = Repeated(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)), max_len=self.max_num_agents
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

    def reset(self, **kwargs):
        if not self.env_config["in_evaluation"]:
            curriculum_reset(self, FootballPvEHierarchicalComEnv)
            self.setup_space()
        return FootballPvEHierarchicalComEnv.reset(self, **kwargs)

    def step(self, actions: dict):
        obs, rew, done, info = FootballPvEHierarchicalComEnv.step(self, actions)

        if not self.env_config["in_evaluation"] and done["__all__"]:
            curriculum_on_episode_end(self, list(info.values())[0]["score"])

        return obs, rew, done, info

    def high_level_actions(self, action_dict):
        return FootballPvEHierarchicalComEnv.high_level_actions(self, action_dict)[:self.num_agents]

    @property
    def high_level_infos(self):
        rew_list = [self.low_level_accumulated_rew[i] for i in range(self.num_agents)] + \
                   [0.0] * (self.max_num_agents - self.num_agents)
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        self.env_info["rewards"] = rewards
        self.env_info["num_agents"] = self.num_agents
        return {"high_level_policy": self.env_info}

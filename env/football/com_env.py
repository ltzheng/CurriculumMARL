import gym
import numpy as np

from env.football.base_env import BaseFootballEnv


class FootballPvEComEnv(BaseFootballEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent communication."""

    def __init__(self, **kwargs):
        BaseFootballEnv.__init__(self, **kwargs)
        self.num_agents = kwargs["number_of_left_players_agent_controls"]
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(19) for _ in range(self.num_agents)])
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Box(low=-np.inf, high=np.inf, shape=(115,)) for _ in range(self.num_agents)])

    def reset(self, **kwargs):
        return tuple(BaseFootballEnv.reset(self, **kwargs))

    def step(self, actions):
        obs_list, rew_list, done, info = BaseFootballEnv.step(self, list(actions))
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info["rewards"] = rewards
        all_rewards = sum(rewards.values()) / len(rewards.values())

        return tuple(obs_list), all_rewards, done, info

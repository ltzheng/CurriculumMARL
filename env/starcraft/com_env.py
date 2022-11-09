from gym.spaces import Discrete, Box, Dict, Tuple
import re

from env.starcraft.base_env import BaseStarCraft2Env


class StarCraft2PvEComEnv(BaseStarCraft2Env):
    def __init__(self, **kwargs):
        self.num_agents = int(re.findall("\d+", kwargs["map_name"])[0])
        BaseStarCraft2Env.__init__(self, **kwargs)
        self.action_space = Tuple([
            Discrete(self._env.get_total_actions()) for _ in range(self.num_agents)])
        self.observation_space = Tuple([Dict({
            "observations": Box(-1, 1, shape=(self._env.get_obs_size(),)),
            "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
        }) for _ in range(self.num_agents)])

    def reset(self, **kwargs):
        return tuple(BaseStarCraft2Env.reset(self, **kwargs))

    def step(self, actions):
        obs_list, rew_list, done, info = BaseStarCraft2Env.step(self, list(actions))
        rewards = {i: rew for i, rew in enumerate(rew_list)}
        info["rewards"] = rewards
        all_rewards = sum(rewards.values()) / len(rewards.values())

        return tuple(obs_list), all_rewards, done, info

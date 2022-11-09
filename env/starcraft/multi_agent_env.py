from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env.starcraft.base_env import BaseStarCraft2Env


class StarCraft2PvEEnv(BaseStarCraft2Env, MultiAgentEnv):
    """Wraps a smac StarCraft env to be compatible with RLlib"""

    def reset(self, **kwargs):
        obs_list = BaseStarCraft2Env.reset(self, **kwargs)
        self.num_agents = len(obs_list)
        return self.group_items(obs_list)

    def step(self, actions: dict):
        act = self.ungroup_items(actions)
        obs_list, rew_list, done, info = BaseStarCraft2Env.step(self, act)
        done = {"__all__": done}

        return (
            self.group_items(obs_list),
            self.group_items(rew_list),
            done,
            self.group_items(info),
        )

    def group_items(self, item):
        """Converts items to dict mapping."""
        if isinstance(item, dict):
            return {f"agent_{i}": item for i in range(self.num_agents)}
        else:
            return {f"agent_{i}": item[i] for i in range(self.num_agents)}

    def ungroup_items(self, item):
        """Converts dict mapping to list."""
        return [item[f"agent_{i}"] for i in range(self.num_agents)]

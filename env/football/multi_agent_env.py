from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env.football.base_env import BaseFootballEnv


class FootballPvEEnv(BaseFootballEnv, MultiAgentEnv):
    """Wraps Google Football env to be compatible with RLlib multi-agent."""

    def __init__(self, **kwargs):
        BaseFootballEnv.__init__(self, **kwargs)
        self.num_agents = kwargs["number_of_left_players_agent_controls"]

    def reset(self, **kwargs):
        return self.group_items(BaseFootballEnv.reset(self, **kwargs))

    def step(self, actions: dict):
        act = self.ungroup_items(actions)
        obs_list, rew_list, done, info = BaseFootballEnv.step(self, act)
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

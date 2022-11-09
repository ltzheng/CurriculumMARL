from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env.football.base_env import BaseFootballEnv


class SelfPlayFootballEnv(BaseFootballEnv, MultiAgentEnv):
    """Wraps Google Football 1v1 env to be compatible with self-play."""

    def __init__(self, **kwargs):
        BaseFootballEnv.__init__(self, **kwargs)

    def reset(self):
        return self.group_items(BaseFootballEnv.reset(self))

    def step(self, actions: dict):
        act = self.ungroup_items(actions)
        obs_list, rew_list, done, info_list = BaseFootballEnv.step(self, act)
        infos = self.group_items(info_list)

        if done:
            if info_list["score"] > 0:
                if self.num_left_agents > 0:
                    infos["left"]["result"] = "win"
                if self.num_right_agents > 0:
                    infos["right"]["result"] = "lose"
            # elif info_list["score"] == 0:  # no goal -> goalie wins (there is no draw)
            #     if self.num_left_agents > 0:
            #         infos["left"]["result"] = "draw"
            #     if self.num_right_agents > 0:
            #         infos["right"]["result"] = "draw"
            else:
                if self.num_left_agents > 0:
                    infos["left"]["result"] = "lose"
                if self.num_right_agents > 0:
                    infos["right"]["result"] = "win"
        return (
            self.group_items(obs_list),
            self.group_items(rew_list),
            {"__all__": done},
            infos,
        )

    def group_items(self, item):
        """Converts items to dict mapping."""
        if isinstance(item, dict):
            if self.num_left_agents == 0:
                return {"right": item}
            elif self.num_right_agents == 0:
                return {"left": item}
            else:
                return {"left": item, "right": item}
        else:
            if self.num_left_agents == 0:
                return {"right": item[0]}
            elif self.num_right_agents == 0:
                return {"left": item[0]}
            else:
                return {"left": item[0], "right": item[1]}

    def ungroup_items(self, item):
        """Converts dict mapping to list."""
        if self.num_left_agents == 0:
            return [item["right"]]
        elif self.num_right_agents == 0:
            return [item["left"]]
        else:
            return [item["left"]] + [item["right"]]

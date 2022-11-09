from ray.rllib.algorithms.callbacks import DefaultCallbacks


class PvEMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs) -> None:
        """Runs when an episode is done."""
        if episode.last_info_for():
            for k, v in episode.last_info_for().items():
                if k != "rewards":  # avoid isnan fault in ppo-comm
                    episode.custom_metrics[k] = int(v) if isinstance(v, bool) else v
        elif episode.last_info_for("group_1"):
            for k, v in episode.last_info_for("group_1")["_group_info"][0].items():
                episode.custom_metrics[k] = int(v) if isinstance(v, bool) else v
        else:
            for policy_name in ["agent_0", "high_level_policy", "high_level_0"]:
                if episode.last_info_for(policy_name):
                    for k, v in episode.last_info_for(policy_name).items():
                        if k != "rewards":  # avoid isnan fault in ppo-comm
                            episode.custom_metrics[k] = int(v) if isinstance(v, bool) else v

import gym
import gfootball.env as football_env
import numpy as np


class BaseFootballEnv(gym.Env):
    """Wraps Google Football env to be compatible RLlib."""

    def __init__(
        self,
        env_name="",
        stacked=False,
        representation="simple115v2",
        rewards="scoring",
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=10,
        logdir="",
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        channel_dimensions=(96, 72),
        other_config_options=None,
        court_range=None,
        in_evaluation=False,
    ):
        if other_config_options is None:
            other_config_options = {}
        if in_evaluation:
            assert rewards == "scoring"
        self._env = football_env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir=logdir,
            extra_players=extra_players,
            number_of_left_players_agent_controls=number_of_left_players_agent_controls,
            number_of_right_players_agent_controls=number_of_right_players_agent_controls,
            channel_dimensions=channel_dimensions,
            other_config_options=other_config_options,
        )

        self.episode_accumulated_score = 0

        self.action_set = other_config_options.get("action_set", "default")
        if self.action_set == "v2":
            self.action_space = gym.spaces.Discrete(20)
            self.num_non_built_in_ai_actions = 0
            self.total_num_actions = 0
        else:
            self.action_space = gym.spaces.Discrete(19)

        self.representation = representation
        if self.representation == "simple115v2":
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(115,)
            )
        else:
            raise NotImplementedError

        self.court_range = court_range
        self.num_left_agents = number_of_left_players_agent_controls
        self.num_right_agents = number_of_right_players_agent_controls

    def reset(self, **kwargs):
        """Reset the environment.
        Returns:
            obs_list (list): The initial observation.
        """
        obs_list = self._env.reset(**kwargs)
        self.episode_accumulated_score = 0
        if len(obs_list.shape) == 1:
            # there is only one agent, convert (115,) to (1,115)
            obs_list = np.expand_dims(obs_list, axis=0)
        return obs_list

    def step(self, actions: list):
        """Steps in the environment.
        Returns:
            obs_list (list): New observations for each ready agent.
            rew_list (list): Reward values for each ready agent.
            done (bool): Done values for each ready agent.
            info (dict): Optional info values for each agent.
        """
        if self.action_set == "v2":
            self.total_num_actions += len(actions)
            for act in actions:
                if act != 19:
                    self.num_non_built_in_ai_actions += 1

        obs_list, rew_list, done, info = self._env.step(actions)
        if len(obs_list.shape) == 1:  # there is only one agent
            obs_list = np.expand_dims(obs_list, axis=0)
            rew_list = np.array([rew_list])
        self.episode_accumulated_score += info["score_reward"]

        if self.court_range is not None:
            if self.representation == "simple115v2":
                ball_pos = obs_list[0][88:91]
            if self.num_left_agents > 0:
                if ball_pos[0] < self.court_range:
                    done = True
            else:
                if -ball_pos[0] < self.court_range:
                    done = True

        if done:
            info = {"score": self.episode_accumulated_score}  # left-right goal difference
            if self.episode_accumulated_score > 0:
                info["win"] = 1
            else:
                info["win"] = 0
            if self.action_set == "v2":
                info["prob_non_built_in_ai_action"] = (
                    self.num_non_built_in_ai_actions / self.total_num_actions
                )
        else:
            info = {}
        return obs_list, rew_list, done, info

    def close(self):
        """Close the environment."""
        self._env.close()

    def render(self, mode="human"):
        self._env.render()

    def update_env(self, **kwargs):
        self.close()
        self._env = football_env.create_environment(**kwargs)
        self.action_set = kwargs["other_config_options"].get("action_set", "default")

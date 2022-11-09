import gym
from gym.spaces import Discrete, Box, Dict
from smac.env import StarCraft2Env
import numpy as np
import random
import logging
from pysc2.lib import protocol
from s2clientprotocol import sc2api_pb2 as sc_pb


class VariableStarCraft2Env(StarCraft2Env):
    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array([int(a) for a in actions if a != -100])]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if action != -100:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        return reward, terminated, info


class BaseStarCraft2Env(gym.Env):
    """Wraps a smac StarCraft env to be compatible with RLlib"""

    def __init__(self, **smac_args):
        """Create a new multi-agent StarCraft env compatible with RLlib.
        
        Args:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.
        """
        self._env = VariableStarCraft2Env(**smac_args)
        self.observation_space = Dict({
            "observations": Box(-1, 1, shape=(self.get_obs_size(),)),
            "action_mask": Box(0, 1, shape=(self.get_total_actions(),)),
        })
        self.action_space = Discrete(self.get_total_actions())

    def reset(self, **kwargs):
        """Resets the env and returns observations from ready agents."""
        obs_list, state_list = self._env.reset(**kwargs)
        return_obs = []
        for i, obs in enumerate(obs_list):
            return_obs.append({
                "action_mask": self.get_avail_agent_actions(i),
                "observations": obs,
            })
        return return_obs

    def step(self, actions: list):
        """Steps in the environment.
        Returns:
            obs_list (list): New observations for each ready agent.
            rew_list (list): Reward values for each ready agent.
            done (bool): Done values for each ready agent.
            info (dict): Optional info values for each agent.
        """
        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        rew_list = []
        return_obs = []
        for i, obs in enumerate(obs_list):
            rew_list.append(rew / len(obs_list))
            return_obs.append({
                "action_mask": self.get_avail_agent_actions(i),
                "observations": obs,
            })

        return return_obs, rew_list, done, info

    def close(self):
        """Close the environment."""
        self._env.close()

    def render(self, mode="human"):
        self._env.render()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_obs_size(self):
        return self._env.get_obs_size()

    def get_total_actions(self):
        return self._env.get_total_actions()

    def get_avail_agent_actions(self, i):
        return self._env.get_avail_agent_actions(i)

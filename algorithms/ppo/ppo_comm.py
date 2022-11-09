"""
PPO with multi-agent communication.
"""

import logging
from typing import Type

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo.ppo import PPO
from algorithms.ppo.ppo_comm_policy import PPOCommPolicy, PPOInvariantCommPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict

logger = logging.getLogger(__name__)


class PPOComm(PPO):
    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return PPOCommPolicy


class PPOInvariantComm(PPO):
    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return PPOInvariantCommPolicy

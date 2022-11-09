"""
Curriculum-capable MARL algorithms.
"""

import copy
import logging
import gym
import importlib
from collections import defaultdict
from typing import (
    DefaultDict,
    Dict,
    Set,
    Type,
)
import math

import ray
from ray.actor import ActorHandle
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.algorithm import Algorithm
from algorithms.ppo.ppo_context_policy import PPOContextPolicy
from algorithms.ppo.ppo_comm_policy import PPOInvariantCommPolicy
from algorithms.ppo.ppo_hrl import PPOHRLConfig, PPOHRL
from algorithms.curriculum.teacher import TEACHERS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.execution.parallel_requests import AsyncRequestsManager
from ray.rllib.offline.estimators import (
    OffPolicyEstimator,
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
)
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict,
    ResultDict,
)

logger = logging.getLogger(__name__)


class PPOCurriculumConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOHRLConfig instance."""
        super().__init__(algo_class=algo_class or PPO)
        self.teacher_config = {}
        self._allow_unknown_configs = True


class PPOHRLCurriculumConfig(PPOHRLConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOHRLConfig instance."""
        super().__init__(algo_class=algo_class or PPO)
        self.teacher_config = {}
        self.policies["high_level_policy"] = PolicySpec(policy_class=PPOInvariantCommPolicy)


class PPOCurriculum(PPO):
    _allow_unknown_subkeys = PPO._allow_unknown_subkeys + [
        "teacher_config",
    ]
    _override_all_subkeys_if_type_changes = (
        PPO._override_all_subkeys_if_type_changes + ["teacher_config"]
    )

    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if self.teacher_name == "bandit":
            return PPOContextPolicy
        else:
            return PPOTorchPolicy

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PPOCurriculumConfig().to_dict()

    def setup_teacher(self):
        # Create the Teacher object.
        self.teacher_name = self.config["teacher_config"].pop("name", None)
        if self.teacher_name == "bandit-no-context":
            self.teacher = TEACHERS["bandit"](trainer=self, trainer_config=self.config)
        else:
            assert self.teacher_name in TEACHERS
            self.teacher = TEACHERS[self.teacher_name](trainer=self, trainer_config=self.config)

    @override(PPO)
    def setup(self, config: PartialAlgorithmConfigDict):
        self.config = self.merge_trainer_configs(
            self.get_default_config(), config, self._allow_unknown_configs
        )
        self.config["env"] = self._env_id
        self.validate_framework(self.config)
        update_global_seed_if_necessary(self.config["framework"], self.config["seed"])

        # Create the Teacher object.
        self.setup_teacher()

        self.validate_config(self.config)
        self._record_usage(self.config)
        self.callbacks = self.config["callbacks"]()

        log_level = self.config.get("log_level")
        if log_level in ["WARN", "ERROR"]:
            logger.info(
                "Current log_level is {}. For more information, "
                "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
                "-vv flags.".format(log_level)
            )
        if self.config.get("log_level"):
            logging.getLogger("rllib").setLevel(self.config["log_level"])

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)
        self.workers = None
        self.train_exec_impl = None

        # Create rollout workers for collecting samples for training.
        self.workers = WorkerSet(
            env_creator=self.env_creator,
            validate_env=self.validate_env,
            policy_class=self.get_default_policy_class(self.config),
            trainer_config=self.config,
            num_workers=self.config["num_workers"],
            local_worker=True,
            logdir=self.logdir,
        )

        self._remote_workers_for_metrics = self.workers.remote_workers()
        self.workers.sync_weights()
        self.config["multiagent"][
            "policies"
        ] = self.workers.local_worker().policy_dict
        self.setup_eval_workers()
        self.setup_reward_estimators()
        self.callbacks.on_algorithm_init(algorithm=self)

    @override(Algorithm)
    def step(self) -> ResultDict:
        results = PPO.step(self)
        results = self.teacher.update_curriculum(result=results)
        return results

    @override(PPO)
    def __getstate__(self) -> dict:
        state = PPO.__getstate__(self)
        if hasattr(self, "teacher") and self.config["teacher_config"].get("task_generator", True):
            name = ray.get(self.teacher.task_generator.get_name.remote())
            if name != "uniform":
                state["teacher"] = ray.get(self.teacher.task_generator.save.remote())
        return state

    @override(PPO)
    def __setstate__(self, state: dict):
        PPO.__setstate__(self, state)
        if hasattr(self, "teacher") and "teacher" in state and self.config["teacher_config"].get("task_generator", True):
            name = ray.get(self.teacher.task_generator.get_name.remote())
            if name != "uniform":
                self.teacher.task_generator.restore.remote(state["teacher"])

    def setup_eval_workers(self):
        # Update with evaluation settings:
        user_eval_config = copy.deepcopy(self.config["evaluation_config"])

        # Assert that user has not unset "in_evaluation".
        assert (
            "in_evaluation" not in user_eval_config
            or user_eval_config["in_evaluation"] is True
        )

        # Merge user-provided eval config with the base config. This makes sure
        # the eval config is always complete, no matter whether we have eval
        # workers or perform evaluation on the (non-eval) local worker.
        eval_config = merge_dicts(self.config, user_eval_config)
        self.config["evaluation_config"] = eval_config

        if self.config.get("evaluation_num_workers", 0) > 0 or self.config.get(
            "evaluation_interval"
        ):
            logger.debug(f"Using evaluation_config: {user_eval_config}.")

            # Validate evaluation config.
            self.validate_config(eval_config)

            # Set the `in_evaluation` flag.
            eval_config["in_evaluation"] = True

            # Evaluation duration unit: episodes.
            # Switch on `complete_episode` rollouts. Also, make sure
            # rollout fragments are short so we never have more than one
            # episode in one rollout.
            if eval_config["evaluation_duration_unit"] == "episodes":
                eval_config.update(
                    {
                        "batch_mode": "complete_episodes",
                        "rollout_fragment_length": 1,
                    }
                )
            # Evaluation duration unit: timesteps.
            # - Set `batch_mode=truncate_episodes` so we don't perform rollouts
            #   strictly along episode borders.
            # Set `rollout_fragment_length` such that desired steps are divided
            # equally amongst workers or - in "auto" duration mode - set it
            # to a reasonably small number (10), such that a single `sample()`
            # call doesn't take too much time and we can stop evaluation as soon
            # as possible after the train step is completed.
            else:
                eval_config.update(
                    {
                        "batch_mode": "truncate_episodes",
                        "rollout_fragment_length": 10
                        if self.config["evaluation_duration"] == "auto"
                        else int(
                            math.ceil(
                                self.config["evaluation_duration"]
                                / (self.config["evaluation_num_workers"] or 1)
                            )
                        ),
                    }
                )

            self.config["evaluation_config"] = eval_config

            _, env_creator = self._get_env_id_and_creator(
                eval_config.get("env"), eval_config
            )

            # Create a separate evaluation worker set for evaluation.
            # If evaluation_num_workers=0, use the evaluation set's local
            # worker for evaluation, otherwise, use its remote workers
            # (parallelized evaluation).
            self.evaluation_workers: WorkerSet = WorkerSet(
                env_creator=env_creator,
                validate_env=None,
                policy_class=self.get_default_policy_class(self.config),
                trainer_config=eval_config,
                num_workers=self.config["evaluation_num_workers"],
                # Don't even create a local worker if num_workers > 0.
                local_worker=False,
                logdir=self.logdir,
            )

            if self.config["enable_async_evaluation"]:
                self._evaluation_async_req_manager = AsyncRequestsManager(
                    workers=self.evaluation_workers.remote_workers(),
                    max_remote_requests_in_flight_per_worker=1,
                    return_object_refs=True,
                )
                self._evaluation_weights_seq_number = 0

    def setup_reward_estimators(self):
        self.reward_estimators: Dict[str, OffPolicyEstimator] = {}
        ope_types = {
            "is": ImportanceSampling,
            "wis": WeightedImportanceSampling,
            "dm": DirectMethod,
            "dr": DoublyRobust,
        }
        for name, method_config in self.config["off_policy_estimation_methods"].items():
            method_type = method_config.pop("type")
            if method_type in ope_types:
                deprecation_warning(
                    old=method_type,
                    new=str(ope_types[method_type]),
                    error=False,
                )
                method_type = ope_types[method_type]
            elif isinstance(method_type, str):
                logger.log(0, "Trying to import from string: " + method_type)
                mod, obj = method_type.rsplit(".", 1)
                mod = importlib.import_module(mod)
                method_type = getattr(mod, obj)
            if isinstance(method_type, type) and issubclass(
                method_type, OffPolicyEstimator
            ):
                policy = self.get_policy()
                gamma = self.config["gamma"]
                self.reward_estimators[name] = method_type(
                    policy, gamma, **method_config
                )
            else:
                raise ValueError(
                    f"Unknown off_policy_estimation type: {method_type}! Must be "
                    "either a class path or a sub-class of ray.rllib."
                    "offline.estimators.off_policy_estimator::OffPolicyEstimator"
                )


class PPOCommCurriculum(PPOCurriculum):
    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return PPOInvariantCommPolicy


class PPOHRLCurriculum(PPOCurriculum, PPOHRL):
    _allow_unknown_subkeys = PPO._allow_unknown_subkeys + [
        "teacher_config", "high_level_policy_config", "low_level_policy_config",
    ]
    _override_all_subkeys_if_type_changes = (
        PPO._override_all_subkeys_if_type_changes + ["teacher_config", "high_level_policy_config", "low_level_policy_config"]
    )

    @classmethod
    @override(PPOCurriculum)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PPOHRLCurriculumConfig().to_dict()

    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return None

    @override(PPOCurriculum)
    def setup(self, config: PartialAlgorithmConfigDict):
        self.config = self.merge_trainer_configs(
            self.get_default_config(), config, self._allow_unknown_configs
        )
        self.config["env"] = self._env_id
        self.validate_framework(self.config)
        update_global_seed_if_necessary(self.config["framework"], self.config["seed"])

        # Create the Teacher object.
        self.setup_teacher()

        # Build the multi-agent config.
        with self.env_creator(self.config["env_config"]) as temp_env:
            if isinstance(temp_env.high_level_action_space, gym.spaces.Tuple):
                from algorithms.ppo.ppo_comm import PPOInvariantCommPolicy
                high_level_policy_cls = PPOInvariantCommPolicy
            else:
                from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
                high_level_policy_cls = PPOTorchPolicy
            self.config["multiagent"]["policies"]["high_level_policy"] = PolicySpec(
                policy_class=high_level_policy_cls,
                observation_space=temp_env.high_level_observation_space,
                action_space=temp_env.high_level_action_space,
                config=self.config["high_level_policy_config"],
            )

            low_level_policy_cls = self.config["multiagent"]["policies"]["low_level_policy"].policy_class
            self.config["multiagent"]["policies"]["low_level_policy"] = PolicySpec(
                policy_class=low_level_policy_cls,
                observation_space=temp_env.low_level_observation_space,
                action_space=temp_env.low_level_action_space,
                config=self.config["low_level_policy_config"],
            )

        self.validate_config(self.config)
        self._record_usage(self.config)
        self.callbacks = self.config["callbacks"]()

        log_level = self.config.get("log_level")
        if log_level in ["WARN", "ERROR"]:
            logger.info(
                "Current log_level is {}. For more information, "
                "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
                "-vv flags.".format(log_level)
            )
        if self.config.get("log_level"):
            logging.getLogger("rllib").setLevel(self.config["log_level"])

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)
        self.workers = None
        self.train_exec_impl = None

        # Create rollout workers for collecting samples for training.
        self.workers = WorkerSet(
            env_creator=self.env_creator,
            validate_env=self.validate_env,
            policy_class=self.get_default_policy_class(self.config),
            trainer_config=self.config,
            num_workers=self.config["num_workers"],
            local_worker=True,
            logdir=self.logdir,
        )

        self._remote_workers_for_metrics = self.workers.remote_workers()
        self.workers.sync_weights()
        self.config["multiagent"][
            "policies"
        ] = self.workers.local_worker().policy_dict
        self.setup_eval_workers()
        self.setup_reward_estimators()

        self.high_level_batches = []
        self.high_level_steps = 0
        self.callbacks.on_algorithm_init(algorithm=self)

    @override(Algorithm)
    def step(self) -> ResultDict:
        results = PPO.step(self)
        results = self.teacher.update_curriculum(result=results)
        return results

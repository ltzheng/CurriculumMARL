"""
A hierarchical PPO.
"""

import copy
import logging
import numpy as np
import gym
import importlib
from collections import defaultdict
from typing import (
    DefaultDict,
    Dict,
    Optional,
    Set,
    Type,
)
import math

import ray
from ray.actor import ActorHandle
from ray.util.debug import log_once
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.common import (
    LEARN_ON_BATCH_TIMER,
    LOAD_BATCH_TIMER,
)
from ray.rllib.execution.parallel_requests import AsyncRequestsManager
from ray.rllib.offline.estimators import (
    OffPolicyEstimator,
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import concat_samples, DEFAULT_POLICY_ID
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from algorithms.ppo.ppo_comm_policy import PPOCommPolicy
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict,
    ResultDict,
)
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_TRAINED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.sgd import do_minibatch_sgd

logger = logging.getLogger(__name__)


def train_one_step(algorithm, train_batch, high_level=False) -> Dict:
    """Function that improves the all policies in `train_batch` on the local worker."""
    if high_level:
        config = algorithm.config["high_level_policy_config"]
        pid = "high_level_policy"
    else:
        config = algorithm.config["low_level_policy_config"]
        pid = "low_level_policy"
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get("num_sgd_iter", 1)
    sgd_minibatch_size = config.get("sgd_minibatch_size", 0)

    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        # Subsample minibatches (size=`sgd_minibatch_size`) from the
        # train batch and loop through train batch `num_sgd_iter` times.
        if num_sgd_iter > 1 or sgd_minibatch_size > 0:
            info = do_minibatch_sgd(
                train_batch,
                {
                    pid: local_worker.get_policy(pid)
                },
                local_worker,
                num_sgd_iter,
                sgd_minibatch_size,
                [],
            )
        # Single update step using train batch.
        else:
            info = local_worker.learn_on_batch(train_batch)

    learn_timer.push_units_processed(train_batch.count)

    if algorithm.reward_estimators:
        info[DEFAULT_POLICY_ID]["off_policy_estimation"] = {}
        for name, estimator in algorithm.reward_estimators.items():
            info[DEFAULT_POLICY_ID]["off_policy_estimation"][name] = estimator.train(
                train_batch
            )
    return info


def multi_gpu_train_one_step(algorithm, train_batch, high_level=False) -> Dict:
    """Multi-GPU version of train_one_step."""
    if high_level:
        config = algorithm.config["high_level_policy_config"]
        policy_id = "high_level_policy"
    else:
        config = algorithm.config["low_level_policy_config"]
        policy_id = "low_level_policy"
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get("num_sgd_iter", 1)
    sgd_minibatch_size = config.get("sgd_minibatch_size", config["train_batch_size"])

    # Determine the number of devices (GPUs or 1 CPU) we use.
    num_devices = int(math.ceil(config["num_gpus"] or 1))

    # Make sure total batch size is dividable by the number of devices.
    # Batch size per tower.
    per_device_batch_size = sgd_minibatch_size // num_devices
    # Total batch size.
    batch_size = per_device_batch_size * num_devices
    assert batch_size % num_devices == 0
    assert batch_size >= num_devices, "Batch size too small!"

    # Handle everything as if multi-agent.
    train_batch = train_batch.as_multi_agent()

    # Load data into GPUs.
    load_timer = algorithm._timers[LOAD_BATCH_TIMER]
    with load_timer:
        num_loaded_samples = {}
        batch = train_batch[policy_id]
        # Decompress SampleBatch, in case some columns are compressed.
        batch.decompress_if_needed()

        # Load the entire train batch into the Policy's only buffer
        # (idx=0). Policies only have >1 buffers, if we are training
        # asynchronously.
        num_loaded_samples[policy_id] = local_worker.policy_map[
            policy_id
        ].load_batch_into_buffer(batch, buffer_index=0)

    # Execute minibatch SGD on loaded data.
    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        # Use LearnerInfoBuilder as a unified way to build the final
        # results dict from `learn_on_loaded_batch` call(s).
        # This makes sure results dicts always have the same structure
        # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
        # tf vs torch).
        learner_info_builder = LearnerInfoBuilder(num_devices=num_devices)

        for policy_id, samples_per_device in num_loaded_samples.items():
            policy = local_worker.policy_map[policy_id]
            num_batches = max(1, int(samples_per_device) // int(per_device_batch_size))
            logger.debug("== sgd epochs for {} ==".format(policy_id))
            for _ in range(num_sgd_iter):
                permutation = np.random.permutation(num_batches)
                for batch_index in range(num_batches):
                    # Learn on the pre-loaded data in the buffer.
                    # Note: For minibatch SGD, the data is an offset into
                    # the pre-loaded entire train batch.
                    results = policy.learn_on_loaded_batch(
                        permutation[batch_index] * per_device_batch_size, buffer_index=0
                    )

                    learner_info_builder.add_learn_on_batch_results(results, policy_id)

        # Tower reduce and finalize results.
        learner_info = learner_info_builder.finalize()

    load_timer.push_units_processed(train_batch.count)
    learn_timer.push_units_processed(train_batch.count)

    if algorithm.reward_estimators:
        learner_info[DEFAULT_POLICY_ID]["off_policy_estimation"] = {}
        for name, estimator in algorithm.reward_estimators.items():
            learner_info[DEFAULT_POLICY_ID]["off_policy_estimation"][
                name
            ] = estimator.train(train_batch)

    return learner_info


class PPOHRLConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOHRLConfig instance."""
        super().__init__(algo_class=algo_class or PPO)
        self.policies = {
            "high_level_policy": PolicySpec(policy_class=PPOCommPolicy),
            "low_level_policy": PolicySpec(policy_class=PPOTorchPolicy),
        }
        self.policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "low_level_policy" if agent_id.startswith(
            "agent_") else "high_level_policy"
        self.policies_to_train = ["high_level_policy", "low_level_policy"]
        self.high_level_policy_config = PPOConfig().to_dict()
        self.low_level_policy_config = PPOConfig().to_dict()
        self._allow_unknown_configs = True


class PPOHRL(PPO):
    _allow_unknown_subkeys = PPO._allow_unknown_subkeys + [
        "high_level_policy_config", "low_level_policy_config",
    ]
    _override_all_subkeys_if_type_changes = (
        PPO._override_all_subkeys_if_type_changes + ["high_level_policy_config", "low_level_policy_config"]
    )

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PPOHRLConfig().to_dict()

    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return None

    @override(PPO)
    def setup(self, config: PartialAlgorithmConfigDict):
        # Setup our config: Merge the user-supplied config (which could
        # be a partial config dict with the class' default).
        self.config = self.merge_trainer_configs(
            self.get_default_config(), config, self._allow_unknown_configs
        )
        self.config["env"] = self._env_id

        # Validate the framework settings in config.
        self.validate_framework(self.config)

        # Set Trainer's seed after we have - if necessary - enabled
        # tf eager-execution.
        update_global_seed_if_necessary(self.config["framework"], self.config["seed"])

        # Build the multi-agent config.
        with self.env_creator(self.config["env_config"]) as temp_env:
            if isinstance(temp_env.high_level_observation_space, gym.spaces.Tuple):
                from algorithms.ppo.ppo_comm_policy import PPOCommPolicy
                high_level_policy_cls = PPOCommPolicy
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
            logging.getLogger("ray.rllib").setLevel(self.config["log_level"])

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        # Create a dict, mapping ActorHandles to sets of open remote
        # requests (object refs). This way, we keep track, of which actors
        # inside this Trainer (e.g. a remote RolloutWorker) have
        # already been sent how many (e.g. `sample()`) requests.
        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)

        self.workers: Optional[WorkerSet] = None
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

        # By default, collect metrics for all remote workers.
        self._remote_workers_for_metrics = self.workers.remote_workers()
        # Ensure remote workers are initially in sync with the local worker.
        self.workers.sync_weights()

        # Now that workers have been created, update our policies
        # dict in config[multiagent] (with the correct original/
        # unpreprocessed spaces).
        self.config["multiagent"][
            "policies"
        ] = self.workers.local_worker().policy_dict

        # Evaluation WorkerSet setup.
        self.setup_eval_workers()
        self.setup_reward_estimators()

        self.high_level_batches = []
        self.high_level_steps = 0
        self.callbacks.on_algorithm_init(algorithm=self)

    @ExperimentalAPI
    @override(PPO)
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        high_level_config = self.config["high_level_policy_config"]
        low_level_config = self.config["low_level_policy_config"]
        high_level_batch_size = high_level_config["train_batch_size"]
        low_level_batch_size = low_level_config["train_batch_size"]
        low_level_batches = []

        if self._by_agent_steps:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_agent_steps=low_level_batch_size,
                concat=False,
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_env_steps=low_level_batch_size,
                concat=False,
            )
        for b in train_batch:
            high_level_batch = b.policy_batches.pop("high_level_policy")
            self.high_level_batches.append(high_level_batch)
            self.high_level_steps += high_level_batch.count
            low_level_batch = b.policy_batches.pop("low_level_policy")
            low_level_batches.append(low_level_batch)

        low_level_train_batch = concat_samples(low_level_batches)
        low_level_train_batch = low_level_train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += low_level_train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += low_level_train_batch.env_steps()

        high_level_updated = False
        if self.high_level_steps >= high_level_batch_size:
            high_level_train_batch = concat_samples(self.high_level_batches)
            high_level_train_batch = high_level_train_batch.as_multi_agent()
            self._counters["high_level_agent_steps_sampled"] += high_level_train_batch.agent_steps()
            high_level_train_batch = standardize_fields(high_level_train_batch, ["advantages"])
            if high_level_config["simple_optimizer"]:
                high_level_train_results = train_one_step(self, high_level_train_batch, high_level=True)
            else:
                high_level_train_results = multi_gpu_train_one_step(self, high_level_train_batch, high_level=True)
            self.high_level_batches = []
            self.high_level_steps = 0
            high_level_updated = True

        low_level_train_batch = standardize_fields(low_level_train_batch, ["advantages"])
        if low_level_config["simple_optimizer"]:
            low_level_train_results = train_one_step(self, low_level_train_batch)
        else:
            low_level_train_results = multi_gpu_train_one_step(self, low_level_train_batch)

        self._counters[NUM_ENV_STEPS_TRAINED] += low_level_train_batch.count
        self._counters[NUM_AGENT_STEPS_TRAINED] += low_level_train_batch.agent_steps()

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        self.after_training_step(low_level_train_batch, low_level_train_results, low_level_config)
        if high_level_updated:
            self.after_training_step(high_level_train_batch, high_level_train_results, high_level_config)
            train_results = dict(high_level_train_results, **low_level_train_results)
        else:
            train_results = low_level_train_results

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    def after_training_step(self, train_batch, train_results, config):
        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > config["vf_clip_param"]
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

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

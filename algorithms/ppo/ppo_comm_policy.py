"""
Policy class for PPO with multi-agent communication.
"""

import logging
import gym
import numpy as np
from typing import Dict, List, Optional, Type, Union

from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_advantages,
)
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType, AgentID
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.repeated import Repeated

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.
    Modification of the original `compute_gae_for_sample_batch` for multi-agent PPO.
    """

    # Extract the rewards for all agents
    # from the info dict to the samplebatch_infos_rewards dict.
    samplebatch_infos_rewards = {'0': sample_batch[SampleBatch.INFOS]}
    if not sample_batch[SampleBatch.INFOS].dtype == "float32":  # i.e., not a np.zeros((n,)) array in the first call
        samplebatch_infos = SampleBatch.concat_samples([
            SampleBatch({k: [v] for k, v in s.items() if k == "rewards"})
            for s in sample_batch[SampleBatch.INFOS]
        ])
        samplebatch_infos_rewards = SampleBatch.concat_samples([
            SampleBatch({str(k): [v] for k, v in s.items()})
            for s in samplebatch_infos["rewards"]
        ])

    # Add items to sample batches of each agents
    batches = []
    for key, action_space in zip(samplebatch_infos_rewards.keys(), policy.action_space):
        i = int(key)
        sample_batch_agent = sample_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = (samplebatch_infos_rewards[key])
        if isinstance(action_space, gym.spaces.box.Box):
            assert len(action_space.shape) == 1
            a_w = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.discrete.Discrete):
            a_w = 1
        else:
            raise UnsupportedSpaceException

        sample_batch_agent[SampleBatch.ACTIONS] = sample_batch[SampleBatch.ACTIONS][:, a_w * i:a_w * (i + 1)]
        sample_batch_agent[SampleBatch.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS][:, i]

        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = sample_batch.get_single_step_input_dict(
                policy.model.view_requirements, index="last")
            all_values = policy._value(**input_dict)
            last_r = all_values[i].item()

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        batches.append(
            compute_advantages(
                sample_batch_agent,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"],
                use_critic=policy.config.get("use_critic", True)
            )
        )

    # Overwrite the original sample batch
    for k in [
        SampleBatch.REWARDS,
        SampleBatch.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ]:
        sample_batch[k] = np.stack([b[k] for b in batches], axis=-1)

    return sample_batch


def validate_spaces(policy, obs_space, action_space, config):
    """Validate observation- and action-spaces."""
    orig_obs_space = getattr(obs_space, "original_space", obs_space)
    if not isinstance(orig_obs_space, gym.spaces.Tuple) and \
            not isinstance(orig_obs_space, gym.spaces.Dict) and \
            not isinstance(orig_obs_space, Repeated):
        raise UnsupportedSpaceException(
            f"Observation space must be a Tuple or a Dict of Tuple or a Repeated, got {orig_obs_space}.")
    if not isinstance(action_space, gym.spaces.Tuple):
        raise UnsupportedSpaceException(
            f"Action space must be a Tuple, got {action_space}.")
    if not isinstance(action_space.spaces[0], gym.spaces.Discrete) and \
            not isinstance(action_space.spaces[0], gym.spaces.Box):
        raise UnsupportedSpaceException(
            f"Expect Box or Discrete action space, got {action_space.spaces[0]}")


class ValueNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.

    This is exactly the same mixin class as in ppo_torch_policy,
    but that one calls .item() on self.model.value_function()[0],
    which will not work for us since our value function returns
    multiple values. Instead, we call .item() in
    compute_gae_for_sample_batch above.
    """

    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0]

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Defines extra fetches per action computation.

        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist: The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }


class PPOCommPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO + communication."""

    def __init__(self, observation_space, action_space, config):
        validate_spaces(self, observation_space, action_space, config)
        config = dict(PPOConfig().to_dict(), **config)
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self._initialize_loss_from_dummy_batch()

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        Modification of the original ppo loss for multi-agent.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)
        curr_entropy = curr_action_dist.entropy()

        loss_data = []
        for i in range(len(train_batch[SampleBatch.VF_PREDS][0])):
            # Only calculate kl loss if necessary (kl-coeff > 0.0).
            if self.config["kl_coeff"] > 0.0:
                action_kl = prev_action_dist.kl(curr_action_dist)[..., i]
                mean_kl_loss = reduce_mean_valid(action_kl)

            mean_entropy = reduce_mean_valid(curr_entropy[..., i])

            surrogate_loss = torch.min(
                train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio[..., i],
                train_batch[Postprocessing.ADVANTAGES][..., i]
                * torch.clamp(
                    logp_ratio[..., i], 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )
            mean_policy_loss = reduce_mean_valid(-surrogate_loss)

            # Compute a value function loss.
            if self.config["use_critic"]:
                value_fn_out = model.value_function()[..., i]
                vf_loss = torch.pow(
                    value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0
                )
                vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
                mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
            # Ignore the value function.
            else:
                vf_loss_clipped = mean_vf_loss = 0.0

            total_loss = reduce_mean_valid(
                -surrogate_loss
                + self.config["vf_loss_coeff"] * vf_loss_clipped
                - self.entropy_coeff * curr_entropy[..., i]
            )

            # Add mean_kl_loss (already processed through `reduce_mean_valid`),
            # if necessary.
            if self.config["kl_coeff"] > 0.0:
                total_loss += self.kl_coeff * mean_kl_loss

            loss_data.append(
                {
                    "total_loss": total_loss,
                    "mean_policy_loss": mean_policy_loss,
                    "mean_vf_loss": mean_vf_loss,
                    "mean_entropy": mean_entropy,
                }
            )

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        total_loss = torch.sum(torch.stack([o["total_loss"] for o in loss_data]))
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = torch.mean(
            torch.stack([o["mean_policy_loss"] for o in loss_data])
        )
        model.tower_stats["mean_vf_loss"] = torch.mean(
            torch.stack([o["mean_vf_loss"] for o in loss_data])
        )
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function()
        )
        model.tower_stats["mean_entropy"] = torch.mean(
            torch.stack([o["mean_entropy"] for o in loss_data])
        )
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


class PPOInvariantCommPolicy(PPOCommPolicy):
    """PyTorch policy class used with PPOComCurriculumTrainer."""

    @override(PPOCommPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for Proximal Policy Objective under `complete_episodes` mode.
        Modification of the original ppo loss for multi-agent with variable agent number.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # policy's latent representation
        pred_act = model.predict_function()
        pred_loss_func = torch.nn.NLLLoss()
        action_dim = pred_act.shape[2]
        batch_pred_loss = 0

        # Split the train_batch into sub-batches with different num_agents, respectively.
        len_vf_pred = len(train_batch[SampleBatch.VF_PREDS][0])
        sgd_batch_size = len(train_batch[SampleBatch.DONES])
        num_agents_batch_size = len(train_batch[SampleBatch.INFOS])

        split_indices = [0]  # indices of each sub-batch
        if len_vf_pred != 1 and sgd_batch_size != 0 and num_agents_batch_size != 0:  # not dummy init
            # num_agents of each sub-batch
            batch_num_agents = [train_batch[SampleBatch.INFOS][0]["num_agents"]]
            for i, s in enumerate(train_batch[SampleBatch.INFOS]):
                if i > 0 and s["num_agents"] != batch_num_agents[-1]:
                    batch_num_agents.append(s["num_agents"])
                    split_indices.append(i)
        elif sgd_batch_size == 0 or num_agents_batch_size == 0:
            return torch.randn(1, requires_grad=True)
        else:  # dummy init
            batch_num_agents = [1]
        split_indices.append(len(train_batch[SampleBatch.INFOS]))
        assert len(batch_num_agents) == len(split_indices) - 1

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        batch_logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )
        batch_curr_entropy = curr_action_dist.entropy()
        batch_action_kl = prev_action_dist.kl(curr_action_dist)

        # Stats holder for the whole train_batch.
        batch_total_loss = torch.zeros((max(batch_num_agents),)).to(batch_logp_ratio.device)
        batch_total_policy_loss = torch.zeros((max(batch_num_agents),)).to(batch_logp_ratio.device)
        batch_total_vf_loss = torch.zeros((max(batch_num_agents),)).to(batch_logp_ratio.device)
        batch_total_entropy = torch.zeros((max(batch_num_agents),)).to(batch_logp_ratio.device)
        batch_total_kl_loss = 0

        # Calculate loss for each sub-batch.
        for idx in range(len(batch_num_agents)):
            # index-selection for current num_agents (ignore dummy agents)
            num_agents = batch_num_agents[idx]
            logp_ratio = batch_logp_ratio[split_indices[idx]:split_indices[idx + 1], :num_agents]
            curr_entropy = batch_curr_entropy[split_indices[idx]:split_indices[idx + 1], :num_agents]
            action_kl = batch_action_kl[split_indices[idx]:split_indices[idx + 1], :num_agents]
            advantages = train_batch[Postprocessing.ADVANTAGES][split_indices[idx]:split_indices[idx + 1], :num_agents]
            value_targets = train_batch[Postprocessing.VALUE_TARGETS][split_indices[idx]:split_indices[idx + 1], :num_agents]
            value_fn_out = model.value_function()[split_indices[idx]:split_indices[idx + 1], :num_agents]

            # pred_action_loss w.r.t current num_agents
            batch_pred_loss += pred_loss_func(nn.LogSoftmax(dim=1)(
                pred_act[split_indices[idx]:split_indices[idx + 1], :num_agents].reshape(-1, action_dim)),
                train_batch[SampleBatch.ACTIONS][split_indices[idx]:split_indices[idx + 1], :num_agents].reshape(-1).to(torch.long))

            for i in range(num_agents):
                # Only calculate kl loss if necessary (kl-coeff > 0.0).
                if self.config["kl_coeff"] > 0.0:
                    sum_kl_loss = torch.sum(action_kl[..., i])
                else:
                    sum_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

                batch_total_entropy[i] += torch.sum(curr_entropy[..., i])

                surrogate_loss = torch.min(
                    advantages[..., i] * logp_ratio[..., i],
                    advantages[..., i] * torch.clamp(
                        logp_ratio[..., i], 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                    ),
                )
                batch_total_policy_loss[i] += torch.sum(-surrogate_loss)

                # Compute a value function loss.
                if self.config["use_critic"]:
                    vf_loss = torch.pow(
                        value_fn_out[..., i] - value_targets[..., i], 2.0
                    )
                    vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
                    batch_total_vf_loss[i] += torch.sum(vf_loss_clipped)
                # Ignore the value function.
                else:
                    vf_loss_clipped = batch_total_vf_loss[i] = 0.0

                batch_total_loss[i] += torch.sum(
                    -surrogate_loss
                    + self.config["vf_loss_coeff"] * vf_loss_clipped
                    - self.entropy_coeff * curr_entropy[..., i]
                )

                # Add kl_loss if necessary.
                if self.config["kl_coeff"] > 0.0:
                    batch_total_loss[i] += self.kl_coeff * sum_kl_loss

                batch_total_kl_loss += sum_kl_loss

        # Store stats.
        loss_data = []
        for i in range(max(batch_num_agents)):
            loss_data.append(
                {
                    "total_loss": torch.div(batch_total_loss[i], sgd_batch_size),
                    "mean_policy_loss": torch.div(batch_total_policy_loss[i], sgd_batch_size),
                    "mean_vf_loss": torch.div(batch_total_vf_loss[i], sgd_batch_size),
                    "mean_entropy": torch.div(batch_total_entropy[i], sgd_batch_size),
                }
            )

        # Sum the loss of each agent.
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["pred_loss"] = batch_pred_loss
        total_loss = torch.sum(torch.stack([o["total_loss"] for o in loss_data])) + 0.1 * batch_pred_loss
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = torch.mean(
            torch.stack([o["mean_policy_loss"] for o in loss_data])
        )
        model.tower_stats["mean_vf_loss"] = torch.mean(
            torch.stack([o["mean_vf_loss"] for o in loss_data])
        )
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function()
        )
        model.tower_stats["mean_entropy"] = torch.mean(
            torch.stack([o["mean_entropy"] for o in loss_data])
        )
        model.tower_stats["mean_kl_loss"] = torch.div(batch_total_kl_loss, sgd_batch_size)

        return total_loss

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                "pred_loss": torch.mean(
                    torch.stack(self.get_tower_stats("pred_loss"))
                ),
            }
        )

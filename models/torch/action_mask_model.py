import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict

torch, nn = try_import_torch()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, gym.spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

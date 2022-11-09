import numpy as np
import gym
from typing import List, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class ContextualFullyConnectedNetwork(TorchModelV2, nn.Module):
    """FCNet with RNN-based context prediction for the bandit."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        self.fcnet = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_fcnet",
        )

        # Bandit context decoder.
        self.pred_act = None
        self.rnn_hidden_dim = kwargs["rnn_hidden_dim"]
        self.fc1 = nn.Linear(int(np.product(obs_space.shape)), self.rnn_hidden_dim)
        self.context_decoder = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.fcnet(input_dict)

        # RNN-based context generator.
        obs = input_dict["obs_flat"].float()
        x = nn.functional.relu(self.fc1(obs))  # [B, cell_size]
        hx = state[0].reshape(-1, self.rnn_hidden_dim)
        B, T = len(seq_lens), max(seq_lens)
        x = x.reshape([B, T, self.rnn_hidden_dim])
        hxs = []
        for t in range(T):
            hx = self.context_decoder(x[:, t].reshape(-1, self.rnn_hidden_dim), hx)
            hxs.append(hx)
        hxs = torch.stack(hxs, dim=1).squeeze().reshape([-1, self.rnn_hidden_dim])
        self.pred_act = self.fc2(hxs)  # [B, act_dim]
        self.last_hx = hxs[-1,...].cpu().detach().numpy().flatten()

        return logits, [hxs]

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return self.fcnet.value_function()

    def predict_function(self):
        assert self.pred_act is not None, "must call forward() first"
        return self.pred_act

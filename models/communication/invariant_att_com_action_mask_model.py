import gym

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict

from models.communication.att_com_action_mask_model import AttComActionMaskModel

torch, nn = try_import_torch()


class InvariantAttComActionMaskModel(AttComActionMaskModel):
    """Attention-based communication model for variable-number of agents with action masks."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            **kwargs,
    ):
        AttComActionMaskModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        # bandit context decoder
        self._pred_act = None
        self.rnn_hidden_dim = 32
        self.fc1 = nn.Linear(self.obs_dim, self.rnn_hidden_dim)
        self.context_decoder = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        if isinstance(self.action_space[0], gym.spaces.Box):
            self.fc2 = nn.Linear(self.rnn_hidden_dim, int(self.act_dim / 2))
        else:
            self.fc2 = nn.Linear(self.rnn_hidden_dim, self.act_dim)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            self.fc1.weight.new(self.max_num_agents,
                                self.rnn_hidden_dim).zero_().squeeze(0)
        ]

    @override(AttComActionMaskModel)
    def forward(self, input_dict, state, seq_lens):
        # preprocess obs and calculate agent num
        unpreprocessed_obs_list = input_dict["obs"].unbatch_repeat_dim()  # [max_len, B, obs_dim]
        obs_list = []
        action_mask_list = []
        count_zeros = [torch.count_nonzero(o) for o in unpreprocessed_obs_list["observations"]]
        if sum(count_zeros) != 0:
            for i in range(len(count_zeros)):
                obs_list.append(unpreprocessed_obs_list["observations"][i])
                action_mask_list.append(unpreprocessed_obs_list["action_mask"][i])
        else:  # dummy initialization
            obs_list = unpreprocessed_obs_list["observations"]
            action_mask_list = unpreprocessed_obs_list["action_mask"]

        obs = torch.stack(obs_list).swapaxes(0, 1)  # [B, num_agents, obs_dim]
        action_mask = torch.stack(action_mask_list).swapaxes(0, 1)  # [B, num_agents, mask_dim]
        batch_size, num_agents = obs.shape[0], obs.shape[1]
        logits, values = self.model_forward(obs, action_mask)

        # add rnn-based context generator
        x = nn.functional.relu(self.fc1(obs)) #.reshape(-1, self.rnn_hidden_dim)
        hx = state[0].reshape(-1, self.rnn_hidden_dim)
        B, T = len(seq_lens), max(seq_lens)
        x = x.reshape([B, T, num_agents, self.rnn_hidden_dim])

        unavailable_agent_num = self.max_num_agents - num_agents
        if unavailable_agent_num > 0:
            x_zero_padding = torch.zeros((B, T, unavailable_agent_num, self.rnn_hidden_dim),
                                              dtype=x.dtype, device=logits.device)
            x = torch.cat((x, x_zero_padding), dim=2)
        hxs = []

        for t in range(T):
            hx = self.context_decoder(x[:, t].reshape(-1, self.rnn_hidden_dim), hx)
            hxs.append(hx)
        hxs = torch.stack(hxs, dim=1).squeeze().reshape([batch_size, self.max_num_agents, self.rnn_hidden_dim])
        pred_act = self.fc2(hxs).reshape([batch_size, self.max_num_agents * self.act_dim])
        self.last_hx = hxs[-1,...].cpu().detach().numpy().flatten()

        # deal with unavailable agents
        # i.e., zero-padding (i) action logits and (ii) value function prediction
        # then ignore them in the loss function

        if unavailable_agent_num > 0:
            action_zero_padding = torch.zeros((batch_size, unavailable_agent_num * self.act_dim),
                                              dtype=logits.dtype, device=logits.device)
            outputs = torch.cat((logits, action_zero_padding), dim=1)
            value_zero_padding = torch.zeros((batch_size, unavailable_agent_num),
                                             dtype=values.dtype, device=values.device)
            self._cur_value = torch.cat((values, value_zero_padding), dim=1)
            self._pred_act = pred_act
        else:
            outputs = logits
            self._cur_value = values
            self._pred_act = pred_act
        self._pred_act = torch.reshape(self._pred_act, [batch_size, self.max_num_agents, self.act_dim])

        return outputs, [hxs]

    def predict_function(self):
        assert self._pred_act is not None, "must call forward() first"
        return self._pred_act

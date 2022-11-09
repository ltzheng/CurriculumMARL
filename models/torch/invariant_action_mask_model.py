import gym

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

from models.communication.layers import mlp_encoder, mlp_decoder

torch, nn = try_import_torch()


ACT_DIM = {
    6: 9,
    11: 12,
    16: 14,
    21: 17,
    50: 31,
    57: 36,
}


class TorchInvariantActionMaskModel(TorchModelV2, nn.Module):
    """ActionMaskingModel with invariant obs encoder and action decoder only for StarCraft2."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        if kwargs["encoder_hidden_layers"] is None:
            encoder_hidden_layers = [64]
        else:
            encoder_hidden_layers = kwargs["encoder_hidden_layers"]
        if kwargs["decoder_hidden_layers"] is None:
            decoder_hidden_layers = [256]
        else:
            decoder_hidden_layers = kwargs["decoder_hidden_layers"]
        if kwargs["act_emb_dim"] is None:
            act_emb_dim = 64
        else:
            act_emb_dim = kwargs["act_emb_dim"]

        self.obs_emb_dim = encoder_hidden_layers[-1]
        self.num_heads = kwargs["num_heads"]
        self.head_dim = kwargs["head_dim"]
        self.att_dim = self.num_heads * self.head_dim
        self.act_emb_dim = act_emb_dim

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, gym.spaces.Dict)
        self.obs_dim = 5
        self.num_agents = int(orig_space["observations"].shape[0] / 5)  # Each agent has a 5-dim feature
        self.act_dim = ACT_DIM[self.num_agents]

        # observation and action encoders
        self.obs_encoder = mlp_encoder([self.obs_dim] + encoder_hidden_layers)

        # attention-based communication
        self.qkv_layer = nn.Linear(self.obs_emb_dim, 3 * self.att_dim)

        # action & value decoders
        self.action_decoder = mlp_decoder([(self.obs_emb_dim + self.att_dim) * self.num_agents] + decoder_hidden_layers + [self.act_dim])
        self.value_decoder = mlp_decoder([(self.obs_emb_dim + self.att_dim) * self.num_agents] + decoder_hidden_layers + [1])

        self._cur_value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["observations"]
        outputs, self._cur_value = self.model_forward(obs, action_mask)

        return outputs, state

    def model_forward(self, obs, action_mask):
        batch_size, num_agents = obs.shape[0], int(obs.shape[1] / 5)
        obs = torch.reshape(obs, [batch_size, num_agents, self.obs_dim])

        # embeddings
        obs_emb = self.obs_encoder(obs)  # [B, num_agents, emb_dim]

        # multi-head attention
        qkv = self.qkv_layer(obs_emb)  # [B, num_agents, att_dim * 3]
        queries, keys, values = torch.chunk(input=qkv, chunks=3, dim=-1)  # [B, num_agents, att_dim]

        queries = torch.reshape(queries, [batch_size, num_agents, self.num_heads, self.head_dim])
        keys = torch.reshape(keys, [batch_size, num_agents, self.num_heads, self.head_dim])
        values = torch.reshape(values, [batch_size, num_agents, self.num_heads, self.head_dim])

        score = torch.einsum("bihd,bjhd->bijh", queries, keys)  # [B, num_agents, num_agents, num_heads]
        score = score / self.head_dim ** 0.5

        # mask of the self-communication
        mask = torch.eye(num_agents, dtype=keys.dtype, device=keys.device)[None, :, :, None]  # [1, num_agents, num_agents, 1]
        mask_inf = torch.clamp(torch.log(1 - mask), FLOAT_MIN, FLOAT_MAX)
        masked_score = score + mask_inf

        wmat = nn.functional.softmax(masked_score, dim=2)  # [B, num_agents, num_agents, num_heads]
        attended_values = torch.einsum("bijh,bjhd->bihd", wmat, values)  # [B, num_agents, num_heads, head_dim]
        attended_values = torch.reshape(attended_values, [batch_size, num_agents, self.att_dim])

        encoded = torch.cat([obs_emb, attended_values], dim=-1)  # [B, num_agents, obs_emb_dim+att_dim]
        encoded = torch.reshape(encoded, [batch_size, num_agents * (self.obs_emb_dim + self.att_dim)])
        logits = self.action_decoder(encoded)
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        outputs = torch.reshape(masked_logits, [batch_size, self.act_dim])
        values = self.value_decoder(encoded).squeeze(-1)

        return outputs, values

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

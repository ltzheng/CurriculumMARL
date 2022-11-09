import gym

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

from models.communication.layers import mlp_encoder, mlp_decoder

torch, nn = try_import_torch()


class AttComActionMaskModel(TorchModelV2, nn.Module):
    """Attention-based communication model for fixed-number of agents with action masks."""

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
            encoder_hidden_layers = [256, 256]
        else:
            encoder_hidden_layers = kwargs["encoder_hidden_layers"]
        if kwargs["decoder_hidden_layers"] is None:
            decoder_hidden_layers = [256]
        else:
            decoder_hidden_layers = kwargs["decoder_hidden_layers"]
        self.obs_emb_dim = encoder_hidden_layers[-1]
        self.num_heads = kwargs["num_heads"]
        self.head_dim = kwargs["head_dim"]
        self.att_dim = self.num_heads * self.head_dim

        orig_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(orig_space, gym.spaces.Tuple):
            self.obs_dim = orig_space[0]["observations"].shape[0]
            self.max_num_agents = len(orig_space)
        elif isinstance(orig_space, gym.spaces.Dict):
            self.obs_dim = orig_space["obs"].shape[0]
            self.max_num_agents = len(orig_space["obs"])
        else:  # Repeated in ray.rllib
            self.obs_dim = getattr(orig_space, "child_space")["observations"].shape[0]
            self.max_num_agents = getattr(orig_space, "max_len", None)
        self.act_dim = int(num_outputs / self.max_num_agents)

        # observation and action encoders
        self.obs_encoder = mlp_encoder([self.obs_dim] + encoder_hidden_layers)

        # attention-based communication
        self.qkv_layer = nn.Linear(self.obs_emb_dim, 3 * self.att_dim)

        # action & value decoders
        self.action_decoder = mlp_decoder([self.obs_emb_dim + self.att_dim] + decoder_hidden_layers + [self.act_dim])
        self.value_decoder = mlp_decoder([self.obs_emb_dim + self.att_dim] + decoder_hidden_layers + [1])

        self._cur_value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        raw_obs_list = input_dict["obs"]
        obs_list = []
        action_mask_list = []
        for obs in raw_obs_list:
            obs_list.append(obs["observations"])
            action_mask_list.append(obs["action_mask"])
        obs = torch.stack(obs_list).swapaxes(0, 1)  # [B, num_agents, obs_dim]
        action_mask = torch.stack(action_mask_list).swapaxes(0, 1)  # [B, num_agents, mask_dim]
        outputs, self._cur_value = self.model_forward(obs, action_mask)

        return outputs, state

    def model_forward(self, obs, action_mask):
        batch_size, num_agents = obs.shape[0], obs.shape[1]

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

        encoded = torch.cat([obs_emb, attended_values], dim=-1)  # [B, num_agents, obs_dim+att_dim]
        logits = self.action_decoder(encoded)
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        outputs = torch.reshape(masked_logits, [batch_size, num_agents * self.act_dim])
        values = self.value_decoder(encoded).squeeze(2)

        return outputs, values

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

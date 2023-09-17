import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, Optional, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.policy.rnn_sequencing import add_time_dimension

torch, nn = try_import_torch()

SS_FULL = 30
SS_AGENT = 12
SS_FRI = 8


class GTrXLNet(RecurrentNetwork, nn.Module):
    """A GTrXL net Model described in [2].
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        *,
        num_transformer_units: int = 1,
        attention_dim: int = 256,
        num_heads: int = 8,
        memory_inference: int = 128,
        memory_training: int = 128,
        head_dim: int = 64,
        position_wise_mlp_dim: int = 256,
        init_gru_gate_bias: float = 1.0
    ):
        """Initializes a GTrXLNet.

        Args:
            num_transformer_units: The number of Transformer repeats to
                use (denoted L in [2]).
            attention_dim: The input and output dimensions of one
                Transformer unit.
            num_heads: The number of attention heads to use in parallel.
                Denoted as `H` in [3].
            memory_inference: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as inference
                input. The first transformer unit will receive this number of
                past observations (plus the current one), instead.
            memory_training: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as training
                input (plus the actual input sequence of len=max_seq_len).
                The first transformer unit will receive this number of
                past observations (plus the input sequence), instead.
            head_dim: The dimension of a single(!) attention head within
                a multi-head attention unit. Denoted as `d` in [3].
            position_wise_mlp_dim: The dimension of the hidden layer
                within the position-wise MLP (after the multi-head attention
                block within one Transformer unit). This is the size of the
                first of the two layers within the PositionwiseFeedforward. The
                second layer always has size=`attention_dim`.
            init_gru_gate_bias: Initial bias values for the GRU gates
                (two GRUs per Transformer unit, one after the MHA, one after
                the position-wise MLP).
        """

        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)

        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory_inference = memory_inference
        self.memory_training = memory_training
        self.head_dim = head_dim
        self.max_seq_len = model_config["max_seq_len"]
        self.obs_dim = SS_FULL #observation_space.shape[0]

        self.inp_act1 = SlimFC(in_size=SS_AGENT, out_size=128, initializer=torch.nn.init.xavier_uniform_, activation_fn=nn.ReLU)
        self.inp_act2 = SlimFC(in_size=SS_FULL-SS_AGENT, out_size=128, initializer=torch.nn.init.xavier_uniform_, activation_fn=nn.ReLU)
        self.inp_val1 = SlimFC(in_size=SS_FULL+4, out_size=128, initializer=torch.nn.init.xavier_uniform_)
        self.inp_val2 = SlimFC(in_size=3*(SS_FULL+4), out_size=128, initializer=torch.nn.init.xavier_uniform_)
        self.att_val = SlimFC(in_size=self.attention_dim, out_size=128, initializer=torch.nn.init.xavier_uniform_)

        self.act_out = SlimFC(in_size=self.attention_dim, out_size=self.num_outputs, initializer=torch.nn.init.xavier_uniform_)
        self.val_out = SlimFC(in_size=128*3, out_size=1, initializer=torch.nn.init.xavier_uniform_)

        self.layers = []
        attention_layers = []
        # 2) Create L Transformer blocks according to [2].
        for i in range(self.num_transformer_units):
            # RelativeMultiHeadAttention part.
            MHA_layer = SkipConnection(
                RelativeMultiHeadAttention(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    input_layernorm=True,
                    output_activation=nn.ReLU,
                ),
                fan_in_layer=None #GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            # Position-wise MultiLayerPerceptron part.
            E_layer = SkipConnection(
                nn.Sequential(
                    torch.nn.LayerNorm(self.attention_dim),
                    SlimFC(
                        in_size=self.attention_dim,
                        out_size=position_wise_mlp_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU,
                    ),
                    SlimFC(
                        in_size=position_wise_mlp_dim,
                        out_size=self.attention_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU,
                    ),
                ),
                fan_in_layer= None #GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            attention_layers.extend([MHA_layer, E_layer])

        self.attention_layers = nn.Sequential(*attention_layers)
        self.layers.extend(attention_layers)

        # Setup trajectory views (`memory-inference` x past memory outs).
        for i in range(self.num_transformer_units):
            space = Box(-1.0, 1.0, shape=(self.attention_dim,))
            self.view_requirements["state_in_{}".format(i)] = ViewRequirement(
                "state_out_{}".format(i),
                shift="-{}:-1".format(self.memory_inference),
                # Repeat the incoming state every max-seq-len times.
                batch_repeat_value=self.max_seq_len,
                space=space,
            )
            self.view_requirements["state_out_{}".format(i)] = ViewRequirement(
                space=space, used_for_training=False
        )

    def _mha_compute(self, x, state, memory_outs, seq_lens, with_memory=True):
        for i in range(len(self.layers)):
            if i % 2 == 0:
                s = add_time_dimension(state[1//2], seq_lens=seq_lens, framework="torch", time_major=False)
                x = self.layers[i](x, memory=s)
                #x = self.layers[i](x, memory=state[i // 2])
            else:
                x = self.layers[i](x)
                if with_memory:
                    memory_outs.append(x)

        if with_memory:
            memory_outs = memory_outs[:-1]
        return x, memory_outs if with_memory else x
    
    def _mha_compute_y(self, x, state, seq_lens):
        for i in range(len(self.layers)):
            if i % 2 == 0:
                s = add_time_dimension(state[1//2], seq_lens=seq_lens, framework="torch", time_major=False)
                x = self.layers[i](x, memory=s)
                #x = self.layers[i](x, memory=state[i // 2])
            else:
                x = self.layers[i](x)
        return x

    @override(ModelV2)
    def forward(
        self, input_dict, state: List[TensorType], seq_lens: TensorType):
        assert seq_lens is not None

        memory_outs = []

        ag = add_time_dimension(input_dict[SampleBatch.OBS]["obs_1_own"][:, :SS_AGENT], seq_lens=seq_lens, framework="torch", time_major=False)
        fo = add_time_dimension(input_dict[SampleBatch.OBS]["obs_1_own"][:, SS_AGENT:], seq_lens=seq_lens, framework="torch", time_major=False)
        x = torch.cat((self.inp_act1(ag), self.inp_act2(fo)),dim=2)
        memory_outs.append(x)

        for i in range(len(self.layers)):
            if i % 2 == 0:
                x = self.layers[i](x, memory=state[i // 2])
            else:
                x = self.layers[i](x)
                memory_outs.append(x)

        out = self.act_out(x)

        v1 = add_time_dimension(torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1), seq_lens=seq_lens, framework="torch", time_major=False)
        v2 = add_time_dimension(torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1), seq_lens=seq_lens, framework="torch", time_major=False)
        y = torch.cat((self.inp_val1(v1), self.inp_val2(v2), self.att_val(x)),dim=2)
        self._value_out = self.val_out(y)

        return torch.reshape(out, [-1, self.num_outputs]), [
            torch.reshape(m, [-1, self.attention_dim]) for m in memory_outs
        ]

    # TODO: (sven) Deprecate this once trajectory view API has fully matured.
    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        return [torch.zeros(self.attention_dim)]
        #return []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])
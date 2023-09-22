import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

class CommanderGru(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._val = None

        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(200, 200, batch_first=True)
        self.rnn_val = nn.GRU(200, 200, batch_first=True)

        self.inp1 = SlimFC(
            4,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            27,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            10,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp4 = SlimFC(
            41,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.act_out = SlimFC(
            500,num_outputs, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )
        

        self.v1 = SlimFC(
            42,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            42,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v3 = SlimFC(
            42,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v4 = SlimFC(
            126,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.val_out = SlimFC(
            500,1, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        return [torch.zeros(200), torch.zeros(200)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:4]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,4:31]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,31:]
        self._inp4 = input_dict["obs"]["obs_1_own"]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((self._v1, self._v2, self._v3), dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x_full = self.inp4(self._inp4)
        y, h = self.rnn_act(add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        x_full = nn.functional.normalize(x_full + y.reshape(-1, 200))
        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2), self.v3(self._v3)),dim=1)
        z_full = self.v4(self._v4)
        w, k = self.rnn_val(add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        z_full = nn.functional.normalize(z_full + w.reshape(-1, 200))
        z = torch.cat((z, z_full), dim=1)
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

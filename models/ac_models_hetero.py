from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import torch.nn.functional as F
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3

OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

SS_AGENT_AC1 = 12
SS_AGENT_AC2 = 10

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

class Esc1(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.shared_layer = SHARED_LAYER

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None

        self.inp1 = SlimFC(
            7,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            18,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            5,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        self.inp1_val = SlimFC(
            OBS_ESC_AC1+ACTION_DIM_AC1+OBS_ESC_AC2+ACTION_DIM_AC2,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:7]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,7:25]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,25:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        x = self.inp1_val(self._v1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])
    
class Esc2(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.shared_layer = SHARED_LAYER

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None

        self.inp1 = SlimFC(
            6,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            18,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            5,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        self.inp1_val = SlimFC(
            OBS_ESC_AC1+ACTION_DIM_AC1+OBS_ESC_AC2+ACTION_DIM_AC2,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:6]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,6:24]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,24:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        x = self.inp1_val(self._v1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class Fight1(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._val = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        self.inp1 = SlimFC(
            SS_AGENT_AC1,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            OBS_AC1-SS_AGENT_AC1,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            OBS_AC1,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v1 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            OBS_AC2+ACTION_DIM_AC2,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v3 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1+OBS_AC2+ACTION_DIM_AC2,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT_AC1]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT_AC1:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._v1),self.v2(self._v2)),dim=1)
        y_full = self.v3(self._v3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

class Fight2(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._val = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        self.inp1 = SlimFC(
            SS_AGENT_AC2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            OBS_AC2-SS_AGENT_AC2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            OBS_AC2,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v1 = SlimFC(
            OBS_AC2+ACTION_DIM_AC2,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v3 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1+OBS_AC2+ACTION_DIM_AC2,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT_AC2]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT_AC2:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)


        y = torch.cat((self.v1(self._v1),self.v2(self._v2)),dim=1)
        y_full = self.v3(self._v3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

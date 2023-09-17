import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

SS_FULL = 47
SS_AGENT = 7
SS_OPPS = 24
SS_FRI = 16
ACTION_DIM = 1

SS_AGENT_GLOB = 30
SS_FULL_GLOB = 150

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

SHARED_LAYER_LIGHT = SlimFC(
    400,
    400,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

SHARED_LAYER_450 = SlimFC(
    450,
    450,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

class CommanderAtt(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp4 = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 1, batch_first=True)
        self.att_val = nn.MultiheadAttention(100, 1, batch_first=True)

        self.inp1 = SlimFC(
            SS_AGENT,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            SS_OPPS,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            SS_FRI,50,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp4 = SlimFC(
            SS_FULL,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            500, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            SS_AGENT,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            SS_OPPS,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v3 = SlimFC(
            SS_FRI,50,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v4 = SlimFC(
            SS_FULL,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,1,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(1).to('cuda')]
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"][:,SS_AGENT:SS_AGENT+SS_OPPS]
        self._inp3 = input_dict["obs"][:, SS_AGENT+SS_OPPS:]
        self._inp4 = input_dict["obs"]
        self.batch_len = self._inp1.shape[0]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x_full = self.inp4(self._inp4)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._inp1),self.v2(self._inp2), self.v3(self._inp3)),dim=1)
        y_full = self.v4(self._inp4)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])
    

class CommanderPolicy(TorchModelV2, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self.shared_layer = SHARED_LAYER

        self.inp1 = SlimFC(
            SS_AGENT,180,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            SS_OPPS,220,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            SS_FRI,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            500, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            SS_AGENT,180,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            SS_OPPS,220,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v3 = SlimFC(
            SS_FRI,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,1,activation_fn=None,initializer=torch.nn.init.orthogonal_
        )
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"][:,SS_AGENT:SS_AGENT+SS_OPPS]
        self._inp3 = input_dict["obs"][:, SS_AGENT+SS_OPPS:]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        y = torch.cat((self.v1(self._inp1), self.v2(self._inp2), self.v3(self._inp3)),dim=1) 
        y = self.shared_layer(y)
        self._val = self.val_out(y)
        return torch.reshape(self._val, [-1])
    


class CommanderAttGlob(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(150, 1, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 1, batch_first=True)

        self.inp1 = SlimFC(
            SS_FULL_GLOB,350,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            SS_FULL_GLOB,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            500, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            SS_FULL_GLOB,350,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            SS_FULL_GLOB,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,1,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(1).to('cuda')]
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]
        self.batch_len = self._inp1.shape[0]

        x = self.inp1(self._inp1)
        x_full = self.inp2(self._inp1)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        y = self.v1(self._inp1)
        y_full = self.v2(self._inp1)
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


class CommanderPolicyGlob(TorchModelV2, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp4 = None
        self._inp5 = None

        self.shared_layer = SHARED_LAYER

        self.inp1 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp4 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp5 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            500, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v3 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v4 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v5 = SlimFC(
            SS_AGENT_GLOB,100,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            500,1,activation_fn=None,initializer=torch.nn.init.orthogonal_
        )
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:SS_AGENT_GLOB]
        self._inp2 = input_dict["obs"][:,SS_AGENT_GLOB:SS_AGENT_GLOB*2]
        self._inp3 = input_dict["obs"][:, SS_AGENT_GLOB*2:SS_AGENT_GLOB*3]
        self._inp4 = input_dict["obs"][:, SS_AGENT_GLOB*3:SS_AGENT_GLOB*4]
        self._inp5 = input_dict["obs"][:, SS_AGENT_GLOB*4:]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3), self.inp4(self._inp4), self.inp5(self._inp5)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        y = torch.cat((self.v1(self._inp1), self.v2(self._inp2), self.v3(self._inp3), self.v4(self._inp4), self.v5(self._inp5)),dim=1) 
        y = self.shared_layer(y)
        self._val = self.val_out(y)
        return torch.reshape(self._val, [-1])
    



class CommanderPolicyLight(TorchModelV2, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self.shared_layer = SHARED_LAYER_LIGHT

        self.inp1 = SlimFC(
            6,140,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            21,180,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            10,80,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            400, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            6,140,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            21,180,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v3 = SlimFC(
            10,80,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            400,1,activation_fn=None,initializer=torch.nn.init.orthogonal_
        )
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:6]
        self._inp2 = input_dict["obs"][:,6:27]
        self._inp3 = input_dict["obs"][:,27:]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        y = torch.cat((self.v1(self._inp1), self.v2(self._inp2), self.v3(self._inp3)),dim=1) 
        y = self.shared_layer(y)
        self._val = self.val_out(y)
        return torch.reshape(self._val, [-1])

class CommanderAttLight(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp4 = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER_LIGHT
        self.att_act = nn.MultiheadAttention(150, 1, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 1, batch_first=True)

        self.inp1 = SlimFC(
            27,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            10,50,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            37,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            400, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            27,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            10,50,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v3 = SlimFC(
            37,150,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            400,1,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:27]
        self._inp2 = input_dict["obs"][:,27:]
        self._inp3 = input_dict["obs"]
        self.batch_len = self._inp1.shape[0]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._inp1),self.v2(self._inp2)),dim=1)
        y_full = self.v3(self._inp3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])


class CommanderPolicyGlobLight(TorchModelV2, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None

        self.shared_layer = SHARED_LAYER_LIGHT

        self.inp1 = SlimFC(
            30,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            25,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            400, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            30,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            25,200,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            400,1,activation_fn=None,initializer=torch.nn.init.orthogonal_
        )
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"][:,:30]
        self._inp2 = input_dict["obs"][:,30:]

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        y = torch.cat((self.v1(self._inp1), self.v2(self._inp2)),dim=1) 
        y = self.shared_layer(y)
        self._val = self.val_out(y)
        return torch.reshape(self._val, [-1])
    

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


class CommanderGru2(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._val = None

        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(150, 150, 2, batch_first=True)
        self.rnn_val = nn.GRU(200, 200, 2, batch_first=True)

        self.inp1 = SlimFC(
            4,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            27,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            10,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp4 = SlimFC(
            41,150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
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
        return [torch.zeros(300), torch.zeros(400)]
    
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
        y, h = self.rnn_act(add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False), state[0].reshape(2,-1,150))
        #x_full = nn.functional.normalize(x_full + y.reshape(-1, 150))
        x = torch.cat((x, y.reshape(-1, 150)), dim=1) # STATT x_full
        x = self.shared_layer(x)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2), self.v3(self._v3)),dim=1)
        z_full = self.v4(self._v4)
        w, k = self.rnn_val(add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False), state[1].reshape(2,-1,200))
        #z_full = nn.functional.normalize(z_full + w.reshape(-1, 200))
        z = torch.cat((z, w.reshape(-1, 200)), dim=1) #STATT z_full
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        #return x, [torch.squeeze(h.reshape(-1,300),0), torch.squeeze(k.reshape(-1,400),0)]
        return x, [h.reshape(-1,300), k.reshape(-1,400)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])


class CommanderGruBi(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._val = None

        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(200, 100, batch_first=True, bidirectional=True)
        self.rnn_val = nn.GRU(200, 100, batch_first=True, bidirectional=True)

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
        y, h = self.rnn_act(add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False), state[0].reshape(2,-1,100))
        #x_full = nn.functional.normalize(x_full + y.reshape(-1, 200))
        x = torch.cat((x, y.reshape(-1, 200)), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2), self.v3(self._v3)),dim=1)
        z_full = self.v4(self._v4)
        w, k = self.rnn_val(add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False), state[1].reshape(2,-1,100))
        #z_full = nn.functional.normalize(z_full + w.reshape(-1, 200))
        z = torch.cat((z, w.reshape(-1, 200)), dim=1)
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        #return x, [torch.squeeze(h.reshape(-1,200),0), torch.squeeze(k.reshape(-1,200),0)]
        return x, [h.reshape(-1,200), k.reshape(-1,200)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])


class CommanderFC(TorchModelV2, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None


        self.inp1 = SlimFC(
            41,400,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            400,400,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.act_out = SlimFC(
            400, num_outputs,activation_fn= None,initializer=torch.nn.init.orthogonal_
        )
        self.v1 = SlimFC(
            41,400,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.v2 = SlimFC(
            400,400,activation_fn= nn.Tanh,initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            400,1,activation_fn=None,initializer=torch.nn.init.orthogonal_
        )
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]

        x = self.inp1(self._inp1)
        x = self.inp2(x)
        x = self.act_out(x)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        y = self.v1(self._inp1)
        y = self.v2(y)
        self._val = self.val_out(y)
        return torch.reshape(self._val, [-1])
    




class CommanderGruAtt(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._val = None

        self.dim = 200
        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(self.dim, self.dim, batch_first=True)
        self.att_act = nn.MultiheadAttention(self.dim, 2, batch_first=True)
        self.rnn_val = nn.GRU(self.dim, self.dim, batch_first=True)
        self.att_val = nn.MultiheadAttention(self.dim, 2, batch_first=True)

        self.inp1 = SlimFC(
            4,70, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            27,180, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            10,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp4 = SlimFC(
            41,self.dim, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
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
            126,self.dim, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.val_out = SlimFC(
            500,1, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        return [torch.zeros(self.dim), torch.zeros(self.dim)]
    
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
        x_full_tn = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y, h = self.rnn_act(x_full_tn, torch.unsqueeze(state[0], 0))
        y_att, _ = self.att_act(x_full_tn, x_full_tn, x_full_tn, need_weights = False)
        x_full = nn.functional.normalize(x_full + y.reshape(-1, self.dim) + y_att.reshape(-1, self.dim))
        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2), self.v3(self._v3)),dim=1)
        z_full = self.v4(self._v4)
        z_full_tn = add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False)
        w, k = self.rnn_val(z_full_tn, torch.unsqueeze(state[1], 0))
        w_att, _ = self.att_val(z_full_tn, z_full_tn, z_full_tn, need_weights=False)
        z_full = nn.functional.normalize(z_full + w.reshape(-1, self.dim) + w_att.reshape(-1, self.dim))
        z = torch.cat((z, z_full), dim=1)
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])
    

class CommanderGruBiAtt(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._val = None

        self.dim = 200
        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(self.dim, 100, batch_first=True, bidirectional=True)
        self.att_act = nn.MultiheadAttention(self.dim, 2, batch_first=True)
        self.rnn_val = nn.GRU(self.dim, 100, batch_first=True, bidirectional=True)
        self.att_val = nn.MultiheadAttention(self.dim, 2, batch_first=True)

        self.inp1 = SlimFC(
            31,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            41,self.dim, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.act_out = SlimFC(
            500,num_outputs, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )
        
        self.v1 = SlimFC(
            42,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            300,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v4 = SlimFC(
            126,self.dim, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.val_out = SlimFC(
            500,1, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        return [torch.zeros(self.dim), torch.zeros(self.dim)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:31]
        self._inp2 = input_dict["obs"]["obs_1_own"]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((self._v1, self._v2, self._v3), dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x = self.inp1(self._inp1)
        x_full = self.inp2(self._inp2)
        x_full_tn = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)

        y_att, _ = self.att_act(x_full_tn, x_full_tn, x_full_tn, need_weights = False)
        x_full = nn.functional.normalize(x_full + y_att.reshape(-1, self.dim))

        y, h = self.rnn_act(x_full_tn, state[0].reshape(2,-1,100))
        x = torch.cat((x, x_full, y.reshape(-1, self.dim)), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)


        z = torch.cat((self.v1(self._v1), self.v1(self._v2), self.v1(self._v3)),dim=1)
        z = self.v2(z)

        z_full = self.v4(self._v4)
        z_full_tn = add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False)

        w_att, _ = self.att_val(z_full_tn, z_full_tn, z_full_tn, need_weights=False)
        z_full = nn.functional.normalize(z_full + w_att.reshape(-1, self.dim))

        w, k = self.rnn_val(z_full_tn, state[1].reshape(2,-1,100))
        z = torch.cat((z, z_full, w.reshape(-1, self.dim)), dim=1)
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        return x, [h.reshape(-1, self.dim), k.reshape(-1, self.dim)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

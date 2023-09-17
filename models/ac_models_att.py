import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

SS_FULL = 30
SS_AGENT = 12
SS_FRI = 8
ACTION_DIM = 4

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.xavier_uniform_,
)
    

#class CCAtt(TorchModelV2, nn.Module):
class CCAtt(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        #TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._inp1 = None
        self._v1 = None
        self.val = None

        self.att_act1 = nn.MultiheadAttention(512,8, batch_first=True)
        self.att_act2 = nn.MultiheadAttention(128,8, batch_first=True)
        self.att_val1 = nn.MultiheadAttention(512,8, batch_first=True)
        self.att_val2 = nn.MultiheadAttention(128,8, batch_first=True)

        self.norm_act1 = nn.LayerNorm(512)
        self.norm_act2 = nn.LayerNorm(128)
        self.norm_val1 = nn.LayerNorm(512)
        self.norm_val2 = nn.LayerNorm(128)

        self.inp1 = SlimFC(
            SS_FULL,
            512,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.att_out = SlimFC(
            512,
            400,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            400,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v1 = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            512,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v_att_out = SlimFC(
            512,
            400,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            400,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)] #specify for having seq_lens
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"]
        #self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = add_time_dimension(self.inp1(self._inp1), seq_lens=seq_lens, framework="torch", time_major=False)
        at, _ = self.att_act1(x,x,x,need_weights=False)
        x = self.norm_act1(at) + x

        # at, _ = self.att_act2(x,x,x,need_weights=False)
        # x = self.norm_act2(at) + x

        x = self.att_out(x.reshape(-1, 512))
        x = self.act_out(x)

        v = add_time_dimension(self.v1(self._v1), seq_lens=seq_lens, framework="torch", time_major=False)
        at, _ = self.att_val1(v,v,v,need_weights=False)
        v = self.norm_val1(at) + v

        # at, _ = self.att_val2(v,v,v,need_weights=False)
        # v = self.norm_val2(at) + v

        v = self.v_att_out(v.reshape(-1,512))
        self.val = self.val_out(v)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self.val is not None, "must call forward first!"
        return torch.reshape(self.val, [-1])
    

class CCAttQ(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        #TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._inp1 = None
        self._v1 = None
        self.val = None

        self.att_act1 = nn.MultiheadAttention(128,8, batch_first=True)
        self.att_act2 = nn.MultiheadAttention(128,8, batch_first=True)
        self.att_act3 = nn.MultiheadAttention(256,8, batch_first=True)
        self.att_val1 = nn.MultiheadAttention(128,8, batch_first=True)
        self.att_val2 = nn.MultiheadAttention(128,8, batch_first=True)
        self.att_val3 = nn.MultiheadAttention(256,8, batch_first=True)

        self.norm_act1 = nn.LayerNorm(128)
        self.norm_act2 = nn.LayerNorm(128)
        self.norm_act3 = nn.LayerNorm(256)
        self.norm_val1 = nn.LayerNorm(128)
        self.norm_val2 = nn.LayerNorm(128)
        self.norm_val3 = nn.LayerNorm(256)

        self.inp1 = SlimFC(
            SS_FULL,
            128,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.q = SlimFC(
            128,
            256,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.kv = SlimFC(
            128,
            256,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            256,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.v1 = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            128,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v_q = SlimFC(
            128,
            256,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v_kv = SlimFC(
            128,
            256,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.val_out = SlimFC(
            256,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)] #specify for having seq_lens
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"]
        #self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = add_time_dimension(self.inp1(self._inp1), seq_lens=seq_lens, framework="torch", time_major=False)
        at, _ = self.att_act1(x,x,x,need_weights=False)
        x = self.norm_act1(at) + x
        at, _ = self.att_act2(x,x,x,need_weights=False)
        x = self.norm_act2(at) + x

        x_q = self.q(x)
        x_kv = self.kv(x)
        at, _ = self.att_act3(x_q, x_kv, x_kv, need_weights = False)
        x = self.norm_act3(at) + x_q
        x = self.act_out(x.reshape(-1, 256))



        v = add_time_dimension(self.v1(self._v1), seq_lens=seq_lens, framework="torch", time_major=False)
        at, _ = self.att_val1(v,v,v,need_weights=False)
        v = self.norm_val1(at) + v
        at, _ = self.att_val2(v,v,v,need_weights=False)
        v = self.norm_val2(at) + v

        v_q = self.v_q(v)
        v_kv = self.v_kv(v)
        at, _ = self.att_val3(v_q, v_kv, v_kv, need_weights = False)
        v = self.norm_val3(at) + v_q
        self.val = self.val_out(v.reshape(-1, 256))

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self.val is not None, "must call forward first!"
        return torch.reshape(self.val, [-1])
    

class CCAttCC(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        #TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None
        self._v2 = None
        self.value = None

        self.v_att = nn.MultiheadAttention(512,8, batch_first=True)

        self.v_norm = nn.LayerNorm(512)

        self.inp1 = SlimFC(
            SS_AGENT,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT-SS_FRI,
            190,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FRI,
            90,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.mid = SlimFC(
            500,
            400,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            400,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1_val = SlimFC(
            SS_FULL+ACTION_DIM,
            512,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            3*(SS_FULL+ACTION_DIM),
            512,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.mid_val = SlimFC(
            512,
            350,
            activation_fn= nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            350,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [torch.zeros(1)] #specify for having seq_lens
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,SS_FULL-SS_FRI:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]), dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.mid(x)
        x = self.act_out(x)

        v_a = add_time_dimension(self.inp1_val(self._v1), seq_lens=seq_lens, framework="torch", time_major=False)
        v_r = add_time_dimension(self.inp2_val(self._v2), seq_lens=seq_lens, framework="torch", time_major=False)

        at, _ = self.v_att(v_a,v_r,v_r,need_weights=False)
        v = self.v_norm(at) + v_a

        v = self.mid_val(v.reshape(-1,512))
        self.value = self.val_out(v)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        assert self.value is not None, "must call forward first!"
        return torch.reshape(self.value, [-1])
    

class CCAttFullSkip(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None
        self._v5 = None
        self._val = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(100, 2, batch_first=True)

        self.inp1 = SlimFC(
            SS_AGENT,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FULL,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v1 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v2 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v3 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v4 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v5 = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(1).to('cuda')]
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        self._v5 = torch.cat((self._v1, self._v2, self._v3, self._v4), dim=1)


        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)


        y = torch.cat((self.v1(self._v1),self.v2(self._v2), self.v3(self._v3), self.v4(self._v4)),dim=1)
        y_full = self.v5(self._v5)
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
    

class CCEscAtt(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp4 = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None
        self._v5 = None
        self._val = None

        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 4, batch_first=True)
        self.att_val = nn.MultiheadAttention(100, 4, batch_first=True)

        self.inp1 = SlimFC(
            6,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            16,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            8,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp4 = SlimFC(
            30,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v1 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v2 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v3 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v4 = SlimFC(
            SS_FULL+ACTION_DIM,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v5 = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        #return [torch.zeros(1).to('cuda')]
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:6]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,6:22]
        self._inp3 = input_dict["obs"]["obs_1_own"][:, 22:]
        self._inp4 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        self._v5 = torch.cat((self._v1, self._v2, self._v3, self._v4), dim=1)


        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x_full = self.inp4(self._inp4)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)


        y = torch.cat((self.v1(self._v1),self.v2(self._v2), self.v3(self._v3), self.v4(self._v4)),dim=1)
        y_full = self.v5(self._v5)
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
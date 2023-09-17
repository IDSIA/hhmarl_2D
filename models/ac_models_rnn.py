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

SHARED_LAYER_CONC = SlimFC(
    600,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.xavier_uniform_,
)
    
CONCAT = False

class CCRocketLstm(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None

        self._inp_val = None
        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None
        self._val = None

        self.shared_layer = SHARED_LAYER
        self.rnn_act_ag = nn.LSTM(SS_AGENT, 200, batch_first=True)
        self.rnn_act_opp = nn.LSTM(SS_FULL-SS_AGENT, 200, batch_first=True)
        self.rnn_val = nn.LSTM(104, 400, batch_first=True)

        self.act_out = SlimFC( #400
            400,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC( #400
            400,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def get_initial_state(self):
        return [torch.zeros(200).to('cuda'), torch.zeros(200).to('cuda'), torch.zeros(200).to('cuda'), torch.zeros(200).to('cuda'), torch.zeros(400).to('cuda'), torch.zeros(400).to('cuda')]
        #return [torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(200), torch.zeros(200)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:]
        self._inp_val = input_dict["obs_flat"]

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x1, [h1, c1] = self.rnn_act_ag(add_time_dimension(self._inp1, seq_lens=seq_lens, framework="torch", time_major=False), [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        x2, [h2 ,c2] = self.rnn_act_opp(add_time_dimension(self._inp2, seq_lens=seq_lens, framework="torch", time_major=False), [torch.unsqueeze(state[2], 0), torch.unsqueeze(state[3], 0)])
        x = torch.cat((x1, x2), dim=2)
        x = self.shared_layer(x)
        x = self.act_out(x)

        x3, [h3, c3] = self.rnn_val(add_time_dimension(self._inp_val, seq_lens=seq_lens, framework="torch", time_major=False), [torch.unsqueeze(state[4], 0), torch.unsqueeze(state[5], 0)])
        y = self.shared_layer(x3)
        self.val = self.val_out(y)

        return x, [torch.squeeze(h1,0), torch.squeeze(c1, 0), torch.squeeze(h2, 0), torch.squeeze(c2, 0), torch.squeeze(h3, 0), torch.squeeze(c3, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self.val is not None, "must call forward first!"
        return torch.reshape(self.val, [-1])


class MsgPassBi(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None

        self._inp_flat = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None

        self.rnn_dim = 128

        self.shared_layer = SHARED_LAYER
        self.rnn_msg = nn.GRU(104, self.rnn_dim, batch_first=True, bidirectional=True)

        self.inp1 = SlimFC(
            SS_AGENT,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
            180,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            self.rnn_dim*2,
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

        self.inp1_val = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp4_val = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def get_initial_state(self):
        return [torch.zeros((2,self.rnn_dim)).to('cuda')]
        
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        self._inp_flat = input_dict["obs_flat"]

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):

        m, h = self.rnn_msg(add_time_dimension(self._inp_flat, seq_lens=seq_lens, framework="torch", time_major=False), state[0].transpose(0,1).contiguous())

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(m.reshape(-1, self.rnn_dim*2))), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        return x, [h]
    
    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        y = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        y = self.shared_layer(y)
        y = self.val_out(y)
        return torch.reshape(y, [-1])
    

class CCGRUSkip(RecurrentNetwork, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp_full = None
        self._inp_flat = None
        self.val = None

        self.rnn_dim = 128
        self.gru_act = nn.GRU(256, self.rnn_dim, batch_first=True)
        self.gru_val = nn.GRU(256, self.rnn_dim, batch_first=True)

        self.inp1 = SlimFC(
            SS_FULL,
            256,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL,
            256,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_mid = SlimFC(
            256+128,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1_val = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            256,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            256,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_mid = SlimFC(
            256+128,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def get_initial_state(self):
        #return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        return [torch.zeros(self.rnn_dim), torch.zeros(self.rnn_dim)]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp_full = input_dict["obs"]["obs_1_own"]
        self._inp_flat = input_dict["obs_flat"]

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state
    
    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x = self.inp1(self._inp_full)
        x, h = self.gru_act(add_time_dimension(x, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        y = self.inp2(self._inp_full)
        x = self.act_mid(torch.cat((x.reshape(-1, self.rnn_dim),y), dim=1))
        x = self.act_out(x)

        z = self.inp1_val(self._inp_flat)
        z, k = self.gru_val(add_time_dimension(z, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        w = self.inp2_val(self._inp_flat)
        z = self.val_mid(torch.cat((z.reshape(-1, self.rnn_dim),w), dim=1))
        self.val = self.val_out(z)
        return x, [torch.squeeze(h,0), torch.squeeze(k,0)]


    @override(ModelV2)
    def value_function(self):
        assert self.val is not None, "must call forward first!"
        return torch.reshape(self.val, [-1])


class CCGru(RecurrentNetwork, nn.Module):

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

        self.rnn_dim = 100

        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(200, self.rnn_dim, batch_first=True)
        self.rnn_val = nn.GRU(200, self.rnn_dim, batch_first=True)

        self.inp1 = SlimFC(
            SS_AGENT,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT-SS_FRI,
            180,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FULL,
            200,
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
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v2 = SlimFC(
            3*(SS_FULL+ACTION_DIM),
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v3 = SlimFC(
            4*(SS_FULL+ACTION_DIM),
            200,
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
        #return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        return [torch.zeros(self.rnn_dim), torch.zeros(self.rnn_dim)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x = self.shared_layer(x)
        y = self.inp3(self._inp3)
        y, h = self.rnn_act(add_time_dimension(y, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        x = torch.cat((x, y.reshape(-1, self.rnn_dim)), dim=1)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2)),dim=1) 
        z = self.shared_layer(z)
        w = self.v3(self._v3)
        w, k = self.rnn_val(add_time_dimension(w, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        z = torch.cat((z, w.reshape(-1, self.rnn_dim)), dim=1)
        self._val = self.val_out(z)

        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])
    

class CCGruFull(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp_gru_a = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._inp_gru_v = None
        self._val = None

        self.rnn_dim = 100 if CONCAT else 500
        self.batch_len = 0

        self.shared_layer = SHARED_LAYER if not CONCAT else SHARED_LAYER_CONC
        self.rnn_act = nn.GRU(30, self.rnn_dim, batch_first=True)
        self.rnn_val = nn.GRU(4*(SS_FULL+ACTION_DIM), self.rnn_dim, batch_first=True)

        self.rnn_act.weight_ih_l0.data.fill_(0)
        self.rnn_act.weight_hh_l0.data.fill_(0)
        self.rnn_act.bias_ih_l0.data.fill_(0)
        self.rnn_act.bias_hh_l0.data.fill_(0)

        self.rnn_val.weight_ih_l0.data.fill_(0)
        self.rnn_val.weight_hh_l0.data.fill_(0)
        self.rnn_val.bias_ih_l0.data.fill_(0)
        self.rnn_val.bias_hh_l0.data.fill_(0)

        self.inp1 = SlimFC(
            SS_AGENT,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT-SS_FRI,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FRI,
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
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v2 = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v3 = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.v4 = SlimFC(
            SS_FULL+ACTION_DIM,
            125,
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
        return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        #return [torch.zeros(self.rnn_dim), torch.zeros(self.rnn_dim)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:, SS_FULL-SS_FRI:]
        self.batch_len = self._inp1.shape[0]
        #self._inp_gru_a = torch.zeros((self.batch_len, 30)).to('cuda')
        self._inp_gru_a = input_dict["obs"]["obs_1_own"]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        #self._inp_gru_v = torch.zeros((self.batch_len, 4*(SS_FULL+ACTION_DIM))).to('cuda')
        self._inp_gru_v = torch.cat((self._v1, self._v2, self._v3, self._v4), dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        ga, h = self.rnn_act(add_time_dimension(self._inp_gru_a, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        ga = torch.reshape(ga, (self.batch_len, -1))

        if CONCAT:
            x = torch.cat((x, ga), dim=1)
        else:
            #x = x + ga
            x = nn.functional.normalize(x + ga)

        x = self.shared_layer(x)
        x = self.act_out(x)


        y = torch.cat((self.v1(self._v1),self.v2(self._v2), self.v3(self._v3), self.v4(self._v4)),dim=1)
        gv, k = self.rnn_val(add_time_dimension(self._inp_gru_v, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        gv = torch.reshape(gv, (self.batch_len, -1))

        if CONCAT:
            y = torch.cat((y, gv), dim=1)
        else:
            #y = y + gv
            y = nn.functional.normalize(y + gv)

        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

class CCGruFullSkip(RecurrentNetwork, nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        #self._inp_gru_a = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None
        self._v5 = None
        #self._inp_gru_v = None
        self._val = None

        self.rnn_dim = 100
        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        self.rnn_act = nn.GRU(self.rnn_dim, self.rnn_dim, batch_first=True)
        self.rnn_val = nn.GRU(self.rnn_dim, self.rnn_dim, batch_first=True)

        # self.rnn_act.weight_ih_l0.data.fill_(0)
        # self.rnn_act.weight_hh_l0.data.fill_(0)
        # self.rnn_act.bias_ih_l0.data.fill_(0)
        # self.rnn_act.bias_hh_l0.data.fill_(0)

        # self.rnn_val.weight_ih_l0.data.fill_(0)
        # self.rnn_val.weight_hh_l0.data.fill_(0)
        # self.rnn_val.bias_ih_l0.data.fill_(0)
        # self.rnn_val.bias_hh_l0.data.fill_(0)

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
        return [torch.zeros(self.rnn_dim).to('cuda'), torch.zeros(self.rnn_dim).to('cuda')]
        #return [torch.zeros(self.rnn_dim), torch.zeros(self.rnn_dim)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]
        #self._inp_gru_a = torch.zeros((self.batch_len, 30)).to('cuda')
        #self._inp_gru_a = input_dict["obs"]["obs_1_own"]

        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        self._v5 = torch.cat((self._v1, self._v2, self._v3, self._v4), dim=1)
        #self._inp_gru_v = torch.zeros((self.batch_len, 4*(SS_FULL+ACTION_DIM))).to('cuda')
        #self._inp_gru_v = torch.cat((self._v1, self._v2, self._v3, self._v4), dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        ga, h = self.rnn_act(add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        ga = torch.reshape(ga, (self.batch_len, -1))
        x_full = nn.functional.normalize(x_full + ga)
        x = torch.cat((x, x_full), dim=1)

        x = self.shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._v1),self.v2(self._v2), self.v3(self._v3), self.v4(self._v4)),dim=1)
        y_full = self.v5(self._v5)
        gv, k = self.rnn_val(add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        gv = torch.reshape(gv, (self.batch_len, -1))
        y_full = nn.functional.normalize(y_full + gv)
        y = torch.cat((y, y_full), dim=1)

        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])

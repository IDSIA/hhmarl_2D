from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

SS_FULL = 21 #21 ac1, 16 ac2
SS_AGENT = 12 #12 ac1, 7 ac2

ACTION_DIM = 4

ESC_SS_AGENT = 5
ESC_SS_FULL = 24

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.xavier_uniform_,
)


class CCRocket(TorchModelV2, nn.Module):
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

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1 = SlimFC(
            SS_AGENT,
            280,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
            220,
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
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:]
        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2)),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])
    
class CCRocketFri(TorchModelV2, nn.Module):
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

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

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
            7,
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
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:SS_FULL]
        self._inp3 = input_dict["obs"]["own_obs"][:, SS_FULL:]
        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2), self.inp3(self._inp3)),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketFriPred(TorchModelV2, nn.Module):
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

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1 = SlimFC(
            SS_AGENT+2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT+2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp3 = SlimFC(
            10,
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
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL+4,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL+4,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT+2]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT+2:SS_FULL+4]
        self._inp3 = input_dict["obs"]["own_obs"][:, SS_FULL+4:]
        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL+4]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL+4]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2), self.inp3(self._inp3)),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketMsg(TorchModelV2, nn.Module):
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

        self._inp_msg = None

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1_msg = SlimFC(
            16,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp2_msg = SlimFC(
            200,
            50,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1 = SlimFC(
            SS_AGENT,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
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

        self.inp1_val = SlimFC(
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:SS_FULL]
        self._inp_msg = input_dict["obs"]["own_obs"][:, SS_FULL:]

        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL]

        m = self.inp1_msg(self._inp_msg)
        m = self.inp2_msg(m)

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2), m),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None and self._inp_msg is not None, "must call forward first!"

        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketMsgAll(TorchModelV2, nn.Module):
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

        self._inp_msg1 = None
        self._inp_msg2 = None

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1_msg = SlimFC(
            8,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp2_msg = SlimFC(
            100,
            25,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp3_msg = SlimFC(
            16,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp4_msg = SlimFC(
            100,
            25,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1 = SlimFC(
            SS_AGENT,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
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

        self.inp1_val = SlimFC(
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:SS_FULL]
        self._inp_msg1 = input_dict["obs"]["own_obs"][:, SS_FULL:SS_FULL+8]
        self._inp_msg2 = input_dict["obs"]["own_obs"][:, SS_FULL+8:]

        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL]

        m1 = self.inp1_msg(self._inp_msg1)
        m1 = self.inp2_msg(m1)

        m2 = self.inp3_msg(self._inp_msg2)
        m2 = self.inp4_msg(m2)

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2), m1, m2),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None and self._inp_msg1 is not None, "must call forward first!"

        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketPred(TorchModelV2, nn.Module):
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

        self._inp_pred = None

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1_pred = SlimFC(
            12,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp2_pred = SlimFC(
            200,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1 = SlimFC(
            SS_AGENT,
            270,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
            230,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.sh_out = SlimFC(
            500,
            300,
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
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:SS_FULL]
        self._inp_pred = input_dict["obs"]["own_obs"][:, SS_FULL:]

        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2)),dim=1)
        x = self.shared_layer(x)
        x = self.sh_out(x)

        y = self.inp1_pred(self._inp_pred)
        y = self.inp2_pred(y)

        z = self.act_out(torch.cat((x,y), dim=1))

        return z, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None and self._inp_pred is not None, "must call forward first!"

        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketPredAll(TorchModelV2, nn.Module):
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

        self._inp_pred = None

        self._vf_inp_act = None
        self._vf_inp_agent = None
        self._vf_inp_opp = None

        self.inp1_pred = SlimFC(
            24,
            300,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp2_pred = SlimFC(
            300,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1 = SlimFC(
            SS_AGENT,
            270,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT,
            230,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.sh_out = SlimFC(
            500,
            300,
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
            ACTION_DIM,
            60,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_val = SlimFC(
            SS_FULL,
            220,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_val = SlimFC(
            SS_FULL,
            220,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, SS_AGENT:SS_FULL]
        self._inp_pred = input_dict["obs"]["own_obs"][:, SS_FULL:]

        self._vf_inp_act = input_dict["obs"]["opponent_action"]
        self._vf_inp_agent = input_dict["obs"]["own_obs"][:, :SS_FULL]
        self._vf_inp_opp = input_dict["obs"]["opponent_obs"][:, :SS_FULL]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2)),dim=1)
        x = self.shared_layer(x)
        x = self.sh_out(x)

        y = self.inp1_pred(self._inp_pred)
        y = self.inp2_pred(y)

        z = self.act_out(torch.cat((x,y), dim=1))

        return z, []

    @override(ModelV2)
    def value_function(self):
        assert self._vf_inp_act is not None and self._vf_inp_agent is not None and self._inp_pred is not None, "must call forward first!"

        x = torch.cat((self.inp1_val(self._vf_inp_act),self.inp2_val(self._vf_inp_agent), self.inp3_val(self._vf_inp_opp)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCEscape(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._inp1 = None
        self._inp2 = None

        self._inp_act = None
        self._inp_agent = None
        self._inp_opp = None

        self.inp1 = SlimFC(
            ESC_SS_AGENT,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            ESC_SS_FULL-ESC_SS_AGENT,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp3 = SlimFC(
            500,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.out_act = SlimFC(
            500,
            self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp1_v = SlimFC(
            ESC_SS_FULL,
            180,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2_v = SlimFC(
            ESC_SS_FULL,
            180,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3_v = SlimFC(
            ACTION_DIM,
            140,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.inp4_v = SlimFC(
            500,
            400,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.out_val = SlimFC(
            400,
            1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["own_obs"][:, :ESC_SS_AGENT]
        self._inp2 = input_dict["obs"]["own_obs"][:, ESC_SS_AGENT:]
        self._inp_act = input_dict["obs"]["opponent_action"]
        self._inp_agent = input_dict["obs"]["own_obs"]
        self._inp_opp = input_dict["obs"]["opponent_obs"]

        x = torch.cat((self.inp1(self._inp1),self.inp2(self._inp2)),dim=1)
        x = self.inp3(x)
        x = self.out_act(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None and self._inp2 is not None, "must call forward first!"
        x = torch.cat((self.inp1_v(self._inp_agent),self.inp2_v(self._inp_opp), self.inp3_v(self._inp_act)),dim=1)
        x = self.inp4_v(x)
        x = self.out_val(x)
        return torch.reshape(x, [-1])
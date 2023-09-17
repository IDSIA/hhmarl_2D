from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import torch.nn.functional as F
torch, nn = try_import_torch()

SS_FULL = 30 # 22 ss1, 30 ss2, 39 ss3
SS_AGENT = 12
SS_FRI = 8 # 8 ss2, 17 ss3


ACTION_DIM = 4

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

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None

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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCRocketDummy(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._inp1 = None

        self.act_out = SlimFC(
            SS_FULL,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.val_out = SlimFC(
            SS_FULL,
            1,
            activation_fn= None,
            initializer=torch.nn.init.xavier_uniform_,
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"]
        return self.act_out(self._inp1), []

    @override(ModelV2)
    def value_function(self):
        assert self._inp1 is not None, "must call forward first!"
        return torch.reshape(self.val_out(self._inp1), [-1])

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

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None

        self.inp1 = SlimFC(
            SS_AGENT,
            200, #220 / 200
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT-SS_FRI,
            200, #190 / 200
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FRI,
            100, #90 / 100
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,SS_FULL-SS_FRI:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])


class CCRocketEsc(TorchModelV2, nn.Module):
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
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            16,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            8,
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
            4*(SS_FULL+ACTION_DIM),
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:6]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,6:22]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,22:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"], input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)
        # self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        # self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        # self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        #assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        assert self._v1 is not None, "must call forward first!"
        #x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        x = self.inp1_val(self._v1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

# FOR BINARY CLASSIFICATION
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class CCIf(TorchModelV2, nn.Module):
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
        self._inp_full = None

        self._v1 = None
        self._v2 = None

        self.decide = StraightThroughEstimator()
        #self.norm = nn.LayerNorm(500)

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
        self.inp_full = SlimFC(
            SS_FULL,
            1,
            activation_fn= None,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,SS_FULL-SS_FRI:]
        self._inp_full = input_dict["obs"]["obs_1_own"]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        z = self.decide(self.inp_full(self._inp_full))

        # res = []
        # for i in range(len(z)):
        #     res.append( torch.cat((self.inp1_wf(self._inp1[i]).unsqueeze(0), self.inp2_wf(self._inp2[i]).unsqueeze(0), self.inp3_wf(self._inp3[i]).unsqueeze(0)),dim=1) if z[i] else torch.cat((self.inp1(self._inp1[i]).unsqueeze(0), self.inp2(self._inp2[i]).unsqueeze(0)),dim=1) )
        # x = torch.cat(res, dim=0)

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3*z)),dim=1)
        #x = self.norm(torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3*z)),dim=1))
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        #x = self.norm(torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1))
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

class CCMask(TorchModelV2, nn.Module):
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
        self._inp_full = None

        self._v1 = None
        self._v2 = None

        self.decide = StraightThroughEstimator()
        #self.norm = nn.LayerNorm(500)

        self.inp1 = SlimFC(
            SS_AGENT,
            270,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp2 = SlimFC(
            SS_FULL-SS_AGENT-SS_FRI,
            230,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp3 = SlimFC(
            SS_FRI,
            230,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp_full = SlimFC(
            SS_FULL,
            1,
            activation_fn= None,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,SS_FULL-SS_FRI:]
        self._inp_full = input_dict["obs"]["obs_1_own"]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        z = self.decide(self.inp_full(self._inp_full))
        f = self.inp3(self._inp3)*z + self.inp2(self._inp2)
        #f = F.normalize(self.inp3(self._inp3)*z + self.inp2(self._inp2))
        #f = self.norm(self.inp3(self._inp3)*z + self.inp2(self._inp2))
        x = torch.cat((self.inp1(self._inp1), f),dim=1)
        #x = self.norm(torch.cat((self.inp1(self._inp1), f),dim=1))
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        #x = self.norm(torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1))
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])
    
class CCFO(TorchModelV2, nn.Module):
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
        self._v2 = None

        self.decide = StraightThroughEstimator()

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
            SS_FRI,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.inp_fo = SlimFC(
            SS_FRI,
            1,
            activation_fn= None,
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
    def forward(self, input_dict, state, seq_lens):
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT:SS_FULL-SS_FRI]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,SS_FULL-SS_FRI:]
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((input_dict["obs"]["obs_4"], input_dict["obs"]["act_4"]),dim=1)

        z = self.decide(self.inp_fo(self._inp3))

        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3*z)),dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None and self._v2 is not None and self._v3 is not None and self._v4 is not None, "must call forward first!"
        x = torch.cat((self.inp1_val(self._v1),self.inp2_val(self._v2), self.inp3_val(self._v3), self.inp4_val(self._v4)),dim=1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])
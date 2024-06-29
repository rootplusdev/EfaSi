import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import MODELS
# from .blocks import Conv2dBlock, LinearBlock, ChannelWiseLeakyReLU, QuantPReLU, \
#     SwitchGate, SwitchLinearBlock, SwitchPReLU, SequentialWithExtraArguments


def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True, floor=False):
    """Fake quantization while keep float gradient."""
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x = torch.clamp(x, qmin / scale, qmax / scale)
    x_quant = (x.detach() * scale + zero_point)
    x_quant = x_quant.floor() if floor else x_quant.round()
    x_dequant = (x_quant - zero_point) / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x


def build_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation.startswith('lrelu/'):  # custom slope
        neg_slope = 1.0 / int(activation[6:])
        return nn.LeakyReLU(neg_slope, inplace=True)
    elif activation == 'crelu':
        return ClippedReLU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'mish':
        return nn.Mish(inplace=True)
    elif activation == 'none':
        return None
    else:
        assert 0, f"Unsupported activation: {activation}"


def build_norm1d_layer(norm, norm_dim=None):
    if norm == 'bn':
        return nn.BatchNorm1d(norm_dim)
    elif norm == 'in':
        return nn.InstanceNorm1d(norm_dim)
    elif norm == 'none':
        return None
    else:
        assert 0, f"Unsupported normalization: {norm}"


def build_norm2d_layer(norm, norm_dim=None):
    if norm == 'bn':
        assert norm_dim is not None
        return nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
        assert norm_dim is not None
        return nn.InstanceNorm2d(norm_dim)
    elif norm == 'none':
        return None
    else:
        assert 0, f"Unsupported normalization: {norm}"


# ---------------------------------------------------------
# Activation Layers


class ClippedReLU(nn.Module):
    def __init__(self, inplace=False, max=1):
        super().__init__()
        self.inplace = inplace
        self.max = max

    def forward(self, x: torch.Tensor):
        if self.inplace:
            return x.clamp_(0, self.max)
        else:
            return x.clamp(0, self.max)


class ChannelWiseLeakyReLU(nn.Module):
    def __init__(self, dim, bias=True, bound=0):
        super().__init__()
        self.neg_slope = nn.Parameter(torch.ones(dim) * 0.5)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.bound = bound

    def forward(self, x):
        assert x.ndim >= 2
        shape = [1, -1] + [1] * (x.ndim - 2)

        slope = self.neg_slope.view(shape)
        # limit slope to range [-bound, bound]
        if self.bound != 0:
            slope = torch.tanh(slope / self.bound) * self.bound

        x += -torch.relu(-x) * (slope - 1)
        if self.bias is not None:
            x += self.bias.view(shape)
        return x


class QuantPReLU(nn.PReLU):
    def __init__(self,
                 num_parameters: int = 1,
                 init: float = 0.25,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 weight_signed=True):
        super().__init__(num_parameters, init)
        self.input_quant_scale = input_quant_scale
        self.input_quant_bits = input_quant_bits
        self.weight_quant_scale = weight_quant_scale
        self.weight_quant_bits = weight_quant_bits
        self.weight_signed = weight_signed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = fake_quant(
            input,
            self.input_quant_scale,
            num_bits=self.input_quant_bits,
        )
        weight = fake_quant(
            self.weight,
            self.weight_quant_scale,
            num_bits=self.weight_quant_bits,
            signed=self.weight_signed,
        )
        return F.prelu(input, weight)


class SwitchPReLU(nn.Module):
    def __init__(self,
                 num_parameters: int,
                 num_experts: int,
                 init: float = 0.25,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.zeros((num_experts, num_parameters), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.full((1, num_parameters), init, **factory_kwargs))

    def get_weight(self, route_index: torch.Tensor) -> torch.Tensor:
        return F.embedding(route_index, self.weight) + self.weight_fact

    def get_weight_at_idx(self, index: int) -> torch.Tensor:
        return self.weight[index] + self.weight_fact[0]

    def forward(self, input: torch.Tensor, route_index: torch.Tensor) -> torch.Tensor:
        neg_slope = self.get_weight(route_index)
        neg_slope = neg_slope.view(neg_slope.shape[0], neg_slope.shape[1],
                                   *([1] * (input.ndim - neg_slope.ndim)))
        return torch.where(input >= 0, input, neg_slope * input)


# ---------------------------------------------------------
# Neural Network Layers


class LinearBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='none',
                 activation='relu',
                 bias=True,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=32):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        self.norm = build_norm1d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)

    def forward(self, x):
        if self.quant:
            # Using floor for inputs leads to closer results to the actual inference code
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits, floor=True)
            w = fake_quant(self.fc.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if self.fc.bias is not None:
                b = fake_quant(self.fc.bias,
                               self.weight_quant_scale * self.input_quant_scale,
                               num_bits=self.bias_quant_bits)
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x)

        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 bias=True,
                 dilation=1,
                 groups=1,
                 activation_first=False,
                 use_spectral_norm=False,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_scale=None,
                 bias_quant_bits=32):
        super(Conv2dBlock, self).__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == 'zeros', "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(self.conv.weight,
                           self.weight_quant_scale,
                           num_bits=self.weight_quant_bits)
            b = self.conv.bias
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)
            if self.quant == 'pixel-dwconv' or self.quant == 'pixel-dwconv-floor':  # pixel-wise quantization in depthwise conv
                assert self.conv.groups == x.size(1), "must be dwconv in pixel-dwconv quant mode!"
                x_ = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
                x = F.unfold(x, self.conv.kernel_size, self.conv.dilation, self.conv.padding, self.conv.stride)
                x = fake_quant(x * w.view(-1)[None, :, None],
                               self.bias_quant_scale,
                               num_bits=self.bias_quant_bits,
                               floor=(self.quant == 'pixel-dwconv-floor'))
                x = x.reshape(x_.shape[0], x_.shape[1], -1, x_.size(2) * x_.size(3)).sum(2)
                x = F.fold(x, (x_.size(2), x_.size(3)), (1, 1))
                x = x + b[None, :, None, None]
            else:
                x = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        else:
            x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Conv1dLine4Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
                 bias=True,
                 dilation=1,
                 groups=1,
                 activation_first=False,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_scale=None,
                 bias_quant_bits=32):
        super().__init__()
        assert in_dim % groups == 0, f"in_dim({in_dim}) should be divisible by groups({groups})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = ks
        self.stride = st
        self.padding = padding  # zeros padding
        self.dilation = dilation
        self.groups = groups
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.weight = nn.Parameter(torch.empty((4 * ks - 3, out_dim, in_dim // groups)))
        self.bias = nn.Parameter(torch.zeros((out_dim,))) if bias else None
        nn.init.kaiming_normal_(self.weight)

        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def make_kernel(self):
        kernel = []
        weight_index = 0
        zero = torch.zeros_like(self.weight[0])
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == j or i + j == self.kernel_size - 1 or i == self.kernel_size // 2 or j == self.kernel_size // 2:
                    kernel.append(self.weight[weight_index])
                    weight_index += 1
                else:
                    kernel.append(zero)

        assert weight_index == self.weight.size(0), f"{weight_index} != {self.weight.size(0)}"
        kernel = torch.stack(kernel, dim=2)
        kernel = kernel.reshape(self.out_dim, self.in_dim // self.groups, self.kernel_size,
                                self.kernel_size)
        return kernel

    def conv(self, x):
        w = self.make_kernel()
        b = self.bias

        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(w, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 dim_out=None,
                 dim_hidden=None,
                 activation_first=False):
        super(ResBlock, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or min(dim_in, dim_out)

        self.learned_shortcut = dim_in != dim_out
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(dim_in,
                        dim_hidden,
                        ks,
                        st,
                        padding,
                        norm,
                        activation,
                        pad_type,
                        activation_first=activation_first),
            Conv2dBlock(dim_hidden,
                        dim_out,
                        ks,
                        st,
                        padding,
                        norm,
                        activation if activation_first else 'none',
                        pad_type,
                        activation_first=activation_first),
        )
        if self.learned_shortcut:
            self.conv_shortcut = Conv2dBlock(dim_in,
                                             dim_out,
                                             1,
                                             1,
                                             norm=norm,
                                             activation=activation,
                                             activation_first=activation_first)

    def forward(self, x):
        residual = self.conv_shortcut(x) if self.learned_shortcut else x
        out = self.conv(x)
        out += residual
        if not self.activation_first and self.activation:
            out = self.activation(out)
        return out


class HashLayer(nn.Module):
    """Maps an input space of size=(input_level**input_size) to a hashed feature space of size=(2**hash_logsize)."""

    def __init__(self,
                 input_size,
                 input_level=2,
                 hash_logsize=20,
                 dim_feature=32,
                 quant_int8=True,
                 sub_features=1,
                 sub_divisor=2,
                 scale_grad_by_freq=False):
        super().__init__()
        self.input_size = input_size
        self.input_level = input_level
        self.hash_logsize = hash_logsize
        self.dim_feature = dim_feature
        self.quant_int8 = quant_int8
        self.sub_features = sub_features
        self.sub_divisor = sub_divisor

        self.perfect_hash = 2 ** hash_logsize >= input_level ** input_size
        if self.perfect_hash:  # Do perfect hashing
            n_features = input_level ** input_size
            self.register_buffer('idx_stride',
                                 input_level ** torch.arange(input_size, dtype=torch.int64))
        else:
            n_features = 2 ** hash_logsize
            ii32 = torch.iinfo(torch.int32)
            self.hashs = nn.Parameter(torch.randint(ii32.min,
                                                    ii32.max,
                                                    size=(input_size, input_level),
                                                    dtype=torch.int32),
                                      requires_grad=False)

        n_features = [math.ceil(n_features / sub_divisor ** i) for i in range(sub_features)]
        self.offsets = [sum(n_features[:i]) for i in range(sub_features + 1)]
        level_dim = max(dim_feature // sub_features, 1)
        self.features = nn.Embedding(sum(n_features),
                                     level_dim,
                                     scale_grad_by_freq=scale_grad_by_freq)
        if self.quant_int8:
            nn.init.trunc_normal_(self.features.weight.data, std=0.5, a=-1, b=127 / 128)
        if self.sub_features > 1:
            self.feature_mapping = nn.Linear(level_dim * sub_features, dim_feature)

    def forward(self, x, x_long=None):
        """
        Args:
            x: float tensor of (batch_size, input_size), in range [0, 1].
            x_long: long tensor of (batch_size, input_size), in range [0, input_level-1].
                If not None, x_long will be used and x will be ignored.
        Returns:
            x_features: float tensor of (batch_size, dim_feature).
        """
        # Quantize input to [0, input_level-1] level
        if x_long is None:
            assert torch.all((x >= 0) & (x <= 1)), f"Input x should be in range [0, 1], but got {x}"
            x_long = torch.round((self.input_level - 1) * x).long()  # (batch_size, input_size)

        if self.perfect_hash:
            x_indices = torch.sum(x_long * self.idx_stride, dim=1)  # (batch_size,)
        else:
            x_onthot = torch.zeros((x_long.shape[0], self.input_size, self.input_level),
                                   dtype=torch.bool,
                                   device=x_long.device)
            x_onthot.scatter_(2, x_long.unsqueeze(-1), 1)  # (batch_size, input_size, input_level)
            x_hash = x_onthot * self.hashs  # (batch_size, input_size, input_level)
            x_hash = torch.sum(x_hash, dim=(1, 2))  # (batch_size,)

            x_indices = x_hash % (2 ** self.hash_logsize)  # (batch_size,)

        if self.sub_features > 1:
            x_features = []
            for i in range(self.sub_features):
                assert torch.all(
                    x_indices < self.offsets[i + 1] - self.offsets[i]
                ), f"indices overflow: {i}, {(x_indices.min(), x_indices.max())}, {(self.offsets[i], self.offsets[i + 1])}"
                x_features.append(self.features(x_indices + self.offsets[i]))
                x_indices = torch.floor_divide(x_indices, self.sub_divisor)
            x_features = torch.cat(x_features, dim=1)  # (batch_size, level_dim * sub_features)
            x_features = self.feature_mapping(x_features)  # (batch_size, dim_feature)
        else:
            x_features = self.features(x_indices)  # (batch_size, dim_feature)

        if self.quant_int8:
            x_features = fake_quant(torch.clamp(x_features, min=-1, max=127 / 128), scale=128)
        return x_features


# ---------------------------------------------------------
# Switch Variant of Networks

class SwitchGate(nn.Module):
    """Switch Gating for MoE networks"""

    def __init__(self, num_experts: int, jitter_eps=0.0, no_scaling=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.no_scaling = no_scaling

    def forward(self, route_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply switch gating to routing logits.
        Args:
            route_logits: float tensor of (batch_size, num_experts).
        Returns:
            route_idx: routing index, long tensor of (batch_size,).
            route_multiplier: multipler for expert outputs, float tensor of (batch_size,).
            load_balancing_loss: load balancing loss, float tensor of ().
            aux_outputs: auxiliary outputs, dict.
        """
        # add random jittering when training
        if self.training and self.jitter_eps > 0:
            noise = torch.rand_like(route_logits)
            noise = noise * 2 * self.jitter_eps + 1 - self.jitter_eps
            route_logits = route_logits * noise

        # get routing probabilities and index
        route_probs = torch.softmax(route_logits, dim=1)  # [B, num_experts]
        route_prob_max, route_idx = torch.max(route_probs, dim=1)  # [B]

        # calc load balancing loss
        inv_batch_size = 1.0 / route_logits.shape[0]
        route_frac = torch.tensor([(route_idx == i).sum() * inv_batch_size
                                   for i in range(self.num_experts)],
                                  dtype=route_probs.dtype,
                                  device=route_probs.device)  # [num_experts]
        route_prob_mean = route_probs.mean(0)  # [num_experts]
        load_balancing_loss = self.num_experts * torch.dot(route_frac, route_prob_mean)
        load_balancing_loss = load_balancing_loss - 1.0

        if self.no_scaling:
            route_multiplier = route_prob_max / route_prob_max.detach()
        else:
            route_multiplier = route_prob_max

        aux_outputs = {
            'route_prob_max': route_prob_max,
            'route_frac_min': route_frac.min(),
            'route_frac_max': route_frac.max(),
            'route_frac_std': route_frac.std(),
        }
        return route_idx, route_multiplier, load_balancing_loss, aux_outputs


class SwitchLinear(nn.Module):
    '''Switchable linear layer for MoE networks'''

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_experts: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert num_experts > 0, "Number of experts must be at least 1"
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.empty((num_experts, out_features * in_features), **factory_kwargs))
        self.weight_fact = nn.Parameter(
            torch.empty((1, out_features * in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_features), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_weight = self.weight[index] + self.weight_fact[0]
        expert_weight = expert_weight.view(self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: torch.Tensor, route_index: torch.Tensor) -> torch.Tensor:
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        output = torch.einsum('bmn,bn->bm', expert_weight, input)
        if expert_bias is not None:
            output = output + expert_bias
        return output


class SwitchLinearBlock(nn.Module):
    '''LinearBlock with switchable linear layer for MoE networks'''

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_experts,
                 activation='relu',
                 bias=True,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=32) -> None:
        super().__init__()
        self.fc = SwitchLinear(in_dim, out_dim, num_experts, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        # initialize activation
        self.activation = build_activation_layer(activation)

    def forward(self, x: torch.Tensor, route_index: torch.Tensor):
        if self.quant:
            weight, bias = self.fc.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if bias is not None:
                b = fake_quant(bias,
                               self.weight_quant_scale * self.input_quant_scale,
                               num_bits=self.bias_quant_bits)
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x, route_index)

        if self.activation:
            out = self.activation(out)
        return out


class SwitchConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_experts: int,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_experts = num_experts
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        num_weight_params = math.prod(self.weight_shape)
        self.weight = nn.Parameter(torch.empty((num_experts, num_weight_params), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.empty((1, num_weight_params), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_channels), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_channels), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.weight_shape)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, *self.weight_shape)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_weight = (self.weight[index] + self.weight_fact[0]).view(self.weight_shape)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: torch.Tensor, route_index: torch.Tensor):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        batch_size = route_index.shape[0]
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        expert_weight = expert_weight.reshape(-1, *self.weight_shape[1:])
        if expert_bias is not None:
            expert_bias = expert_bias.view(-1)

        input = input.view(1, -1, input.size(-2), input.size(-1))
        output = F.conv2d(input, expert_weight, expert_bias, self.stride, padding, self.dilation,
                          self.groups * batch_size)
        output = output.view(batch_size, self.weight_shape[0], output.size(-2), output.size(-1))
        return output


class SwitchConv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_experts,
                 ks,
                 st=1,
                 padding=0,
                 activation='relu',
                 pad_type='zeros',
                 bias=True,
                 dilation=1,
                 groups=1,
                 activation_first=False,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_scale=None,
                 bias_quant_bits=32):
        super().__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = SwitchConv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            num_experts=num_experts,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == 'zeros', "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x: torch.Tensor, route_index: torch.Tensor):
        if self.activation and self.activation_first:
            x = self.activation(x)
        if self.quant:
            batch_size = x.shape[0]
            weight, bias = self.conv.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            x = x.view(1, -1, *x.shape[2:])
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            w = w.view(-1, *w.shape[2:])
            if bias is not None:
                bias = fake_quant(bias, self.bias_quant_scale, num_bits=self.bias_quant_bits)
                bias = bias.reshape(-1)
            x = F.conv2d(x, w, bias, self.conv.stride, self.conv.padding, self.conv.dilation,
                         self.conv.groups * batch_size)
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
        else:
            x = self.conv(x, route_index)
        if not self.activation_first and self.activation:
            x = self.activation(x)
        return x


# ---------------------------------------------------------
# Custom Containers


class SequentialWithExtraArguments(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x


def tuple_op(f, x):
    return tuple((f(xi) if xi is not None else None) for xi in x)

def add_op(t):
    return None if (t[0] is None and t[1] is None) else t[0] + t[1]


class DirectionalConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((3, dim_out, dim_in)))
        self.bias = nn.Parameter(torch.zeros((dim_out, )))
        nn.init.kaiming_normal_(self.weight)

    def _conv1d_direction(self, x, dir):
        kernel_size, dim_out, dim_in = self.weight.shape
        zero = torch.zeros((dim_out, dim_in), dtype=self.weight.dtype, device=self.weight.device)

        weight_func_map = [
            lambda w: (zero, zero, zero, w[0], w[1], w[2], zero, zero, zero),
            lambda w: (zero, w[0], zero, zero, w[1], zero, zero, w[2], zero),
            lambda w: (w[0], zero, zero, zero, w[1], zero, zero, zero, w[2]),
            lambda w: (zero, zero, w[2], zero, w[1], zero, w[0], zero, zero),
        ]
        weight = torch.stack(weight_func_map[dir](self.weight), dim=2)
        weight = weight.reshape(dim_out, dim_in, kernel_size, kernel_size)
        return torch.conv2d(x, weight, self.bias, padding=kernel_size // 2)

    def forward(self, x):
        assert len(x) == 4, f"must be 4 directions, got {len(x)}"
        return tuple((self._conv1d_direction(xi, i) if xi is not None else None) for i, xi in enumerate(x))


class DirectionalConvResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim, dim)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.d_conv(x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv1x1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
        return x


class Conv0dResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = tuple_op(self.conv1, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(self.conv2, x)
        x = tuple_op(self.activation, x)
        x = tuple_op(add_op, zip(x, residual))
        return x


class Mapping(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()
        self.d_conv = DirectionalConvLayer(dim_in, dim_middle)
        self.convs = nn.Sequential(
            *[DirectionalConvResBlock(dim_middle) for _ in range(4)],
            Conv0dResBlock(dim_middle),
        )
        self.final_conv = nn.Conv2d(dim_middle, dim_out, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x, dirs=[0, 1, 2, 3]):
        x = tuple((x if i in dirs else None) for i in range(4))
        x = self.d_conv(x)
        x = tuple_op(self.activation, x)
        x = self.convs(x)
        x = tuple_op(self.final_conv, x)
        x = tuple(xi for xi in x if xi is not None)
        x = torch.stack(x, dim=1)  # [B, <=4, dim_out, H, W]
        return x


class RotatedConv2d3x3(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((9, dim_out, dim_in)))
        self.bias = nn.Parameter(torch.zeros((dim_out, )))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        w = self.weight
        _, dim_out, dim_in = w.shape
        zero = torch.zeros((dim_out, dim_in), dtype=w.dtype, device=w.device)
        weight7x7 = [
            zero, zero, zero, zero, w[0], zero, zero, \
            zero, zero, w[1], zero, zero, zero, zero, \
            w[2], zero, zero, zero, zero, w[3], zero, \
            zero, zero, zero, w[4], zero, zero, zero, \
            zero, w[5], zero, zero, zero, zero, w[6], \
            zero, zero, zero, zero, w[7], zero, zero, \
            zero, zero, w[8], zero, zero, zero, zero,
        ]
        weight7x7 = torch.stack(weight7x7, dim=2)
        weight7x7 = weight7x7.reshape(dim_out, dim_in, 7, 7)
        return torch.conv2d(x, weight7x7, self.bias, padding=7 // 2)


class Mix9Net(nn.Module):
    def __init__(self,
                 model_name="Mix9Net",
                 dim_middle=128,
                 dim_feature=64,
                 dim_policy=32,
                 dim_value=64,
                 dim_dwconv=32):
        super().__init__()
        self.model_size = (dim_middle, dim_feature, dim_policy, dim_value, dim_dwconv)
        self.model_name = model_name
        assert dim_dwconv <= dim_feature, f"Invalid dim_dwconv {dim_dwconv}"
        assert dim_dwconv >= dim_policy, "dim_dwconv must be not less than dim_policy"

        self.mapping1 = Mapping(2, dim_middle, dim_feature)
        self.mapping2 = Mapping(2, dim_middle, dim_feature)

        # feature depth-wise conv
        self.feature_dwconv = Conv2dBlock(
            dim_dwconv,
            dim_dwconv,
            ks=3,
            st=1,
            padding=3 // 2,
            groups=dim_dwconv,
            activation='relu',
            quant='pixel-dwconv-floor',
            input_quant_scale=128,
            input_quant_bits=16,
            weight_quant_scale=65536,
            weight_quant_bits=16,
            bias_quant_scale=128,
            bias_quant_bits=16,
        )

        # policy head (point-wise conv)
        dim_pm = self.policy_middle_dim = 16
        self.policy_pwconv_weight_linear = nn.Sequential(
            LinearBlock(dim_feature, dim_policy * 2, activation='relu', quant=True),
            LinearBlock(dim_policy * 2, dim_pm * dim_policy + dim_pm, activation='none', quant=True),
        )
        self.policy_output = nn.Conv2d(dim_pm, 1, 1)

        # value head
        self.value_corner_linear = LinearBlock(dim_feature, dim_value, activation='relu', quant=True)
        self.value_edge_linear = LinearBlock(dim_feature, dim_value, activation='relu', quant=True)
        self.value_center_linear = LinearBlock(dim_feature, dim_value, activation='relu', quant=True)
        self.value_quad_linear = LinearBlock(dim_value, dim_value, activation='relu', quant=True)
        self.value_linear = nn.Sequential(
            LinearBlock(dim_feature + 4 * dim_value, dim_value, activation='relu', quant=True),
            LinearBlock(dim_value, dim_value, activation='relu', quant=True),
            LinearBlock(dim_value, 3, activation='none', quant=True),
        )

    def custom_init(self):
        self.feature_dwconv.conv.weight.data.mul_(0.25)

    def get_feature(self, data, inv_side=False):
        # get per-point 4-direction cell features
        feature1 = self.mapping1(data, dirs=[0, 1])  # [B, 2, dim_feature, H, W]
        feature2 = self.mapping2(data, dirs=[2, 3])  # [B, 2, dim_feature, H, W]
        feature = torch.cat([feature1, feature2], dim=1)  # [B, 4, dim_feature, H, W]

        # clamp feature for int quantization
        feature = torch.clamp(feature, min=-16, max=16)  # int16, scale=32, [-16,16]
        feature = fake_quant(feature, scale=32, num_bits=16)
        # sum (and rescale) feature across four directions
        feature = torch.mean(feature, dim=1)  # [B, dim_feature, H, W] int16, scale=128, [-16,16]
        # apply relu activation
        feature = F.relu(feature)  # [B, dim_feature, H, W] int16, scale=128, [0,16]

        # apply feature depth-wise conv
        _, _, _, _, dim_dwconv = self.model_size
        feat_dwconv = feature[:, :dim_dwconv]  # int16, scale=128, [0,16]
        feat_dwconv = self.feature_dwconv(feat_dwconv * 4)  # [B, dwconv, H, W] relu
        feat_dwconv = fake_quant(feat_dwconv, scale=128, num_bits=16)  # int16, scale=128, [0,9/2*16*4]

        # apply activation for direct feature
        feat_direct = feature[:, dim_dwconv:]  # [B, dim_feature-dwconv, H, W] int16, scale=128, [0,16]
        feat_direct = fake_quant(feat_direct, scale=128, num_bits=16)  # int16, scale=128, [0,16]

        feature = torch.cat([feat_dwconv, feat_direct], dim=1)  # [B, dim_feature, H, W]

        return feature

    def forward(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]

        # value feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8

        # value feature accumulator of nine groups
        B, _, H, W = feature.shape
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        pwconv_weight = pwconv_output[:, :dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128*128, num_bits=16, floor=True)
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(dim_pm)
        ], 1)
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy:].reshape(B, dim_pm, 1, 1)  # int32, scale=128*128*128
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        policy = self.policy_output(policy)  # [B, 1, H, W]

        # value head
        value_00 = fake_quant(self.value_corner_linear(feature_00), floor=True)
        value_01 = fake_quant(self.value_edge_linear(feature_01), floor=True)
        value_02 = fake_quant(self.value_corner_linear(feature_02), floor=True)
        value_10 = fake_quant(self.value_edge_linear(feature_10), floor=True)
        value_11 = fake_quant(self.value_center_linear(feature_11), floor=True)
        value_12 = fake_quant(self.value_edge_linear(feature_12), floor=True)
        value_20 = fake_quant(self.value_corner_linear(feature_20), floor=True)
        value_21 = fake_quant(self.value_edge_linear(feature_21), floor=True)
        value_22 = fake_quant(self.value_corner_linear(feature_22), floor=True)

        def avg4(a, b, c, d):
            ab = fake_quant((a + b + 1/128) / 2, floor=True)
            cd = fake_quant((c + d + 1/128) / 2, floor=True)
            return fake_quant((ab + cd + 1/128) / 2, floor=True)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        value_q00 = self.value_quad_linear(value_q00)
        value_q01 = self.value_quad_linear(value_q01)
        value_q10 = self.value_quad_linear(value_q10)
        value_q11 = self.value_quad_linear(value_q11)

        value = torch.cat([
            feature_sum,
            value_q00, value_q01, value_q10, value_q11,
        ], 1)  # [B, dim_feature + dim_value * 4]
        value = self.value_linear(value)

        policy = torch.where(data.sum(dim=1, keepdim=True) > 0, -torch.inf, policy)

        return policy, value

    def forward_debug_print(self, data):
        _, _, dim_policy, _, _ = self.model_size

        # get feature from single side
        feature = self.get_feature(data, False)  # [B, dim_feature, H, W]
        print(f"feature after dwconv at (0,0): \n{(feature[..., 0, 0]*128).int()}")

        # value feature accumulator
        feature_sum = torch.sum(feature, dim=(2, 3))  # [B, dim_feature]
        print(f"feature sum before scale: \n{(feature_sum*128).int()}")
        feature_sum = fake_quant(feature_sum / 256, scale=128, num_bits=32, floor=True)  # srai 8
        print(f"feature sum: \n{(feature_sum*128).int()}")

        # value feature accumulator of nine groups
        B, _, H, W = feature.shape
        H0, W0 = 0, 0
        H1, W1 = (H // 3) + (H % 3 == 2), (W // 3) + (W % 3 == 2)
        H2, W2 = (H // 3) * 2 + (H % 3 > 0), (W // 3) * 2 + (W % 3 > 0)
        H3, W3 = H, W
        feature_00 = torch.sum(feature[:, :, H0:H1, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_01 = torch.sum(feature[:, :, H0:H1, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_02 = torch.sum(feature[:, :, H0:H1, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_10 = torch.sum(feature[:, :, H1:H2, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_11 = torch.sum(feature[:, :, H1:H2, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_12 = torch.sum(feature[:, :, H1:H2, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_20 = torch.sum(feature[:, :, H2:H3, W0:W1], dim=(2, 3))  # [B, dim_feature]
        feature_21 = torch.sum(feature[:, :, H2:H3, W1:W2], dim=(2, 3))  # [B, dim_feature]
        feature_22 = torch.sum(feature[:, :, H2:H3, W2:W3], dim=(2, 3))  # [B, dim_feature]
        feature_00 = fake_quant(feature_00 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_01 = fake_quant(feature_01 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_02 = fake_quant(feature_02 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_10 = fake_quant(feature_10 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_11 = fake_quant(feature_11 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_12 = fake_quant(feature_12 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_20 = fake_quant(feature_20 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_21 = fake_quant(feature_21 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        feature_22 = fake_quant(feature_22 / 32, scale=128, num_bits=32, floor=True)  # srai 5
        print(f"feature 00 sum: \n{(feature_00*128).int()}")
        print(f"feature 01 sum: \n{(feature_01*128).int()}")
        print(f"feature 02 sum: \n{(feature_02*128).int()}")
        print(f"feature 10 sum: \n{(feature_10*128).int()}")
        print(f"feature 11 sum: \n{(feature_11*128).int()}")
        print(f"feature 12 sum: \n{(feature_12*128).int()}")
        print(f"feature 20 sum: \n{(feature_20*128).int()}")
        print(f"feature 21 sum: \n{(feature_21*128).int()}")
        print(f"feature 22 sum: \n{(feature_22*128).int()}")

        # policy head
        dim_pm = self.policy_middle_dim
        pwconv_output = self.policy_pwconv_weight_linear(feature_sum)
        print(f"policy pwconv output: \n{(pwconv_output*128*128).int()}")
        pwconv_weight = pwconv_output[:, :dim_pm * dim_policy].reshape(B, dim_pm * dim_policy, 1, 1)
        pwconv_weight = fake_quant(pwconv_weight, scale=128*128, num_bits=16, floor=True)
        print(f"policy pwconv weight: \n{(pwconv_weight.flatten(1, -1)*128*128).int()}")
        policy = fake_quant(feature[:, :dim_policy], scale=128, num_bits=16)  # [B, dim_policy, H, W]
        print(f"policy after dwconv at (0,0): \n{(policy[..., 0, 0]*128).int()}")
        policy = torch.cat([
            F.conv2d(input=policy.reshape(1, B * dim_policy, H, W),
                     weight=pwconv_weight[:, dim_policy * i:dim_policy * (i + 1)],
                     groups=B).reshape(B, 1, H, W) for i in range(dim_pm)
        ], 1)
        print(f"policy after dynamic pwconv at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        pwconv_bias = pwconv_output[:, dim_pm * dim_policy:].reshape(B, dim_pm, 1, 1)  # int32, scale=128*128*128
        print(f"policy pwconv bias: \n{(pwconv_bias.flatten(1, -1)*128*128*128).int()}")
        policy = torch.clamp(policy + pwconv_bias, min=0)  # [B, dim_pm, H, W] int32, scale=128*128*128, relu
        print(f"policy pwconv output at (0,0): \n{(policy[..., 0, 0]*128*128*128).int()}")
        policy = self.policy_output(policy)  # [B, 1, H, W]
        print(f"policy output at (0,0): \n{policy[..., 0, 0]}")

        # value head
        value_00 = fake_quant(self.value_corner_linear(feature_00), floor=True)
        value_01 = fake_quant(self.value_edge_linear(feature_01), floor=True)
        value_02 = fake_quant(self.value_corner_linear(feature_02), floor=True)
        value_10 = fake_quant(self.value_edge_linear(feature_10), floor=True)
        value_11 = fake_quant(self.value_center_linear(feature_11), floor=True)
        value_12 = fake_quant(self.value_edge_linear(feature_12), floor=True)
        value_20 = fake_quant(self.value_corner_linear(feature_20), floor=True)
        value_21 = fake_quant(self.value_edge_linear(feature_21), floor=True)
        value_22 = fake_quant(self.value_corner_linear(feature_22), floor=True)
        print(f"value_00: \n{(value_00*128).int()}")
        print(f"value_01: \n{(value_01*128).int()}")
        print(f"value_02: \n{(value_02*128).int()}")
        print(f"value_10: \n{(value_10*128).int()}")
        print(f"value_11: \n{(value_11*128).int()}")
        print(f"value_12: \n{(value_12*128).int()}")
        print(f"value_20: \n{(value_20*128).int()}")
        print(f"value_21: \n{(value_21*128).int()}")
        print(f"value_22: \n{(value_22*128).int()}")

        def avg4(a, b, c, d):
            ab = fake_quant((a + b + 1/128) / 2, floor=True)
            cd = fake_quant((c + d + 1/128) / 2, floor=True)
            return fake_quant((ab + cd + 1/128) / 2, floor=True)

        value_q00 = avg4(value_00, value_01, value_10, value_11)
        value_q01 = avg4(value_01, value_02, value_11, value_12)
        value_q10 = avg4(value_10, value_11, value_20, value_21)
        value_q11 = avg4(value_11, value_12, value_21, value_22)
        print(f"value_q00 avg: \n{(value_q00*128).int()}")
        print(f"value_q01 avg: \n{(value_q01*128).int()}")
        print(f"value_q10 avg: \n{(value_q10*128).int()}")
        print(f"value_q11 avg: \n{(value_q11*128).int()}")
        value_q00 = self.value_quad_linear(value_q00)
        value_q01 = self.value_quad_linear(value_q01)
        value_q10 = self.value_quad_linear(value_q10)
        value_q11 = self.value_quad_linear(value_q11)
        print(f"value_q00: \n{(value_q00*128).int()}")
        print(f"value_q01: \n{(value_q01*128).int()}")
        print(f"value_q10: \n{(value_q10*128).int()}")
        print(f"value_q11: \n{(value_q11*128).int()}")

        value = torch.cat([
            feature_sum,
            value_q00, value_q01, value_q10, value_q11,
        ], 1)  # [B, dim_feature + dim_value * 4]
        print(f"value feature input: \n{(value*128).int()}")
        for i, linear in enumerate(self.value_linear):
            value = linear(value)
            print(f"value feature after layer {i}: \n{(value*128).int()}")

        return policy, value

    @property
    def weight_clipping(self):
        # Clip prelu weight of mapping activation to [-1,1] to avoid overflow
        # In this range, prelu is the same as `max(x, ax)`.
        return [{
            'params': ['feature_dwconv.conv.weight'],
            'min_weight': -32768 / 65536,
            'max_weight': 32767 / 65536
        },
        {
            'params': ['value_corner_linear.fc.weight',
                       'value_edge_linear.fc.weight',
                       'value_center_linear.fc.weight',
                       'value_quad_linear.fc.weight',
                       'value_linear.0.fc.weight',
                       'value_linear.1.fc.weight',
                       'value_linear.2.fc.weight',
                       'policy_pwconv_weight_linear.0.fc.weight',
                       'policy_pwconv_weight_linear.1.fc.weight'],
            'min_weight': -128 / 128,
            'max_weight': 127 / 128
        }]

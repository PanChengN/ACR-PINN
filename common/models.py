import torch
import torch.nn as nn


def _initialize_linear(linear, gain=5 / 3):
    nn.init.xavier_normal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MLP(nn.Module):
    """Shared baseline MLP used by every PDE experiment."""

    def __init__(self, layers):
        super().__init__()
        modules = []
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            linear = nn.Linear(in_dim, out_dim)
            _initialize_linear(linear)
            modules.extend([linear, nn.Tanh()])
        output = nn.Linear(layers[-2], layers[-1])
        _initialize_linear(output, gain=1.0)
        modules.append(output)
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)


class ModifiedMLP(nn.Module):
    """The gated fully-connected architecture of Wang--Teng--Perdikaris.

    Two input encoders construct global candidate features U and V.  Each
    hidden activation H is mixed feature-wise as H*U + (1-H)*V.  This is kept
    distinct from LDA: its encoders are computed once, it has no per-layer
    coordinate re-encoding, and no sample-wise two-branch softmax gate.
    """

    def __init__(self, layers):
        super().__init__()
        if len(layers) < 3 or len(set(layers[1:-1])) != 1:
            raise ValueError('ModifiedMLP requires equal-width hidden layers.')
        width = layers[1]
        self.activation = nn.Tanh()
        self.encoder_u = nn.Linear(layers[0], width)
        self.encoder_v = nn.Linear(layers[0], width)
        self.hidden = nn.ModuleList([
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(layers[:-2], layers[1:-1])
        ])
        self.output = nn.Linear(layers[-2], layers[-1])
        for layer in [self.encoder_u, self.encoder_v, *self.hidden]:
            _initialize_linear(layer)
        _initialize_linear(self.output, gain=1.0)

    def forward(self, inputs):
        encoded_u = self.activation(self.encoder_u(inputs))
        encoded_v = self.activation(self.encoder_v(inputs))
        activations = inputs
        for layer in self.hidden:
            candidate = self.activation(layer(activations))
            activations = candidate * encoded_u + (1.0 - candidate) * encoded_v
        return self.output(activations)


class FourierFeatureMLP(nn.Module):
    """MLP preceded by a fixed, seeded random Fourier embedding."""

    def __init__(self, layers, feature_count=50, scale=1.0):
        super().__init__()
        if feature_count < 1 or scale <= 0:
            raise ValueError('Fourier features require positive count and scale.')
        self.feature_count = int(feature_count)
        self.scale = float(scale)
        projection = torch.randn(layers[0], self.feature_count) * self.scale
        self.register_buffer('projection', projection)
        effective_layers = [2 * self.feature_count, *layers[1:]]
        self.backbone = MLP(effective_layers)

    def forward(self, inputs):
        phases = 2.0 * torch.pi * inputs @ self.projection
        return self.backbone(torch.cat([torch.sin(phases), torch.cos(phases)], dim=1))


class LDA(nn.Module):
    """Layer-wise dynamic attention with optional implementation-level stabilization.

    The stabilized form preserves the submitted LDA topology: each hidden layer
    re-encodes the inputs twice, dynamically fuses the encodings with a softmax
    gate, and injects the result residually.  It only calibrates the tanh
    initialization, starts every gate at the symmetric 0.5/0.5 solution, and
    scales the residual branch by one learned scalar per hidden layer.
    """

    def __init__(self, layers, activation=None, stabilized=True, residual_scale=0.1,
                 encoder_input_dimensions=None):
        super().__init__()
        if len(layers) < 3:
            raise ValueError('LDA requires at least one hidden layer.')
        self.activation = activation or nn.Tanh()
        self.stabilized = stabilized
        self.encoder_input_dimensions = tuple(
            encoder_input_dimensions or (layers[0], layers[0])
        )
        if len(self.encoder_input_dimensions) != 2:
            raise ValueError('LDA requires exactly two encoder input dimensions.')
        self.linear = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])
        hidden_dims = layers[1:-1]
        self.encoder1 = nn.ModuleList([
            nn.Linear(self.encoder_input_dimensions[0], width) for width in hidden_dims
        ])
        self.encoder2 = nn.ModuleList([
            nn.Linear(self.encoder_input_dimensions[1], width) for width in hidden_dims
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * width, width),
                nn.Tanh(),
                nn.Linear(width, 2 * width),
            )
            for width in hidden_dims
        ])
        if stabilized:
            self.residual_scales = nn.Parameter(torch.full((len(hidden_dims),), residual_scale))
        else:
            self.register_buffer('residual_scales', torch.ones(len(hidden_dims)))
        for linear in self.linear:
            gain = 1.0 if linear is self.linear[-1] else 5 / 3
            _initialize_linear(linear, gain=gain)
        for encoder in list(self.encoder1) + list(self.encoder2):
            _initialize_linear(encoder, gain=5 / 3 if stabilized else 1.0)
        if stabilized:
            for gate in self.gates:
                nn.init.xavier_normal_(gate[0].weight, gain=5 / 3)
                nn.init.zeros_(gate[0].bias)
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)

    @staticmethod
    def _intervene_on_gate(weights, mode, layer_mean=None, generator=None):
        if mode == 'native':
            return weights
        if mode == 'uniform':
            return torch.full_like(weights, 0.5)
        if mode == 'swap':
            return weights.flip(1)
        if mode == 'sample_permute':
            permutation = torch.randperm(weights.size(0), device=weights.device, generator=generator)
            return weights.index_select(0, permutation)
        if mode == 'layer_mean':
            if layer_mean is None:
                raise ValueError('layer_mean intervention requires one mean gate tensor per layer.')
            mean = torch.as_tensor(layer_mean, dtype=weights.dtype, device=weights.device)
            if mean.ndim == 2:
                mean = mean.unsqueeze(0)
            if mean.shape != (1, 2, weights.size(2)):
                raise ValueError(
                    f'Expected layer mean shape (2, {weights.size(2)}) or '
                    f'(1, 2, {weights.size(2)}), got {tuple(mean.shape)}.'
                )
            return mean.expand_as(weights)
        raise ValueError(f'Unknown gate intervention: {mode}')

    def _forward(self, inputs, return_gates=False, gate_mode='native',
                 layer_means=None, generator=None):
        activations = inputs
        collected_gates = []
        encoder_input_1, encoder_input_2 = self._encoder_inputs(inputs)
        for index in range(len(self.linear) - 1):
            activations = self.activation(self.linear[index](activations))
            encoded1 = self.activation(self.encoder1[index](encoder_input_1))
            encoded2 = self.activation(self.encoder2[index](encoder_input_2))
            logits = self.gates[index](torch.cat([activations, encoded1, encoded2], dim=1))
            weights = torch.softmax(logits.view(activations.size(0), 2, activations.size(1)), dim=1)
            weights = self._intervene_on_gate(
                weights, gate_mode,
                layer_mean=None if layer_means is None else layer_means[index],
                generator=generator,
            )
            if return_gates:
                collected_gates.append(weights)
            attention = weights[:, 0, :] * encoded1 + weights[:, 1, :] * encoded2
            activations = activations + self.residual_scales[index] * attention
        output = self.linear[-1](activations)
        return (output, collected_gates) if return_gates else output

    def _encoder_inputs(self, inputs):
        return inputs, inputs

    def forward(self, inputs):
        return self._forward(inputs, return_gates=False)

    def forward_with_gates(self, inputs):
        return self._forward(inputs, return_gates=True)

    def forward_with_gate_intervention(self, inputs, mode, layer_means=None, generator=None,
                                       return_gates=False):
        return self._forward(
            inputs, return_gates=return_gates, gate_mode=mode,
            layer_means=layer_means, generator=generator,
        )


class SeparateSpaceTimeLDA(LDA):
    """LDA with one spatial and one temporal encoder at every hidden layer.

    The backbone and feature-wise dynamic gate are identical to joint LDA.  Only
    the two candidate encodings differ: branch 1 receives ``x`` and branch 2
    receives ``t``.  For the 2D [x, t] Klein--Gordon input this is within one
    percent of the joint LDA parameter count, so it is a direct controlled
    comparison rather than a capacity increase.
    """

    def __init__(self, layers, activation=None, stabilized=True, residual_scale=0.1):
        if layers[0] != 2:
            raise ValueError('SeparateSpaceTimeLDA requires a two-coordinate [x, t] input.')
        super().__init__(
            layers,
            activation=activation,
            stabilized=stabilized,
            residual_scale=residual_scale,
            encoder_input_dimensions=(1, 1),
        )

    def _encoder_inputs(self, inputs):
        return inputs[:, 0:1], inputs[:, 1:2]


class _LayerwiseCoordinateModel(nn.Module):
    """Shared initialization and output path for controlled LDA ablations."""

    def __init__(self, layers, residual_scale=0.1):
        super().__init__()
        if len(layers) < 3 or len(set(layers[1:-1])) != 1:
            raise ValueError('Controlled ablations require at least one equal-width hidden layer.')
        self.input_dim = layers[0]
        self.hidden_dims = layers[1:-1]
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([
            nn.Linear(layers[index], layers[index + 1])
            for index in range(len(layers) - 1)
        ])
        for index, linear in enumerate(self.linear):
            _initialize_linear(linear, gain=1.0 if index == len(self.linear) - 1 else 5 / 3)
        self._initial_residual_scale = residual_scale

    def _output(self, activations):
        return self.linear[-1](activations)


class DirectCoordinateInjection(_LayerwiseCoordinateModel):
    """Repeated raw-coordinate injection inside each hidden pre-activation."""

    def __init__(self, layers, residual_scale=0.1):
        super().__init__(layers, residual_scale=residual_scale)
        self.coordinate_projection = nn.ModuleList([
            nn.Linear(self.input_dim, width, bias=False) for width in self.hidden_dims
        ])
        for projection in self.coordinate_projection:
            _initialize_linear(projection)

    def forward(self, inputs):
        activations = inputs
        for index in range(len(self.hidden_dims)):
            activations = self.activation(
                self.linear[index](activations) + self.coordinate_projection[index](inputs)
            )
        return self._output(activations)


class SingleEncoderResidual(_LayerwiseCoordinateModel):
    """One learned coordinate encoder injected through a scaled residual path."""

    def __init__(self, layers, residual_scale=0.1):
        super().__init__(layers, residual_scale=residual_scale)
        self.encoder = nn.ModuleList([
            nn.Linear(self.input_dim, width) for width in self.hidden_dims
        ])
        self.residual_scales = nn.Parameter(
            torch.full((len(self.hidden_dims),), residual_scale)
        )
        for encoder in self.encoder:
            _initialize_linear(encoder)

    def forward(self, inputs):
        activations = inputs
        for index in range(len(self.hidden_dims)):
            activations = self.activation(self.linear[index](activations))
            encoded = self.activation(self.encoder[index](inputs))
            activations = activations + self.residual_scales[index] * encoded
        return self._output(activations)


class _TwoEncoderResidual(_LayerwiseCoordinateModel):
    def __init__(self, layers, residual_scale=0.1):
        super().__init__(layers, residual_scale=residual_scale)
        self.encoder1 = nn.ModuleList([
            nn.Linear(self.input_dim, width) for width in self.hidden_dims
        ])
        self.encoder2 = nn.ModuleList([
            nn.Linear(self.input_dim, width) for width in self.hidden_dims
        ])
        self.residual_scales = nn.Parameter(
            torch.full((len(self.hidden_dims),), residual_scale)
        )
        for encoder in list(self.encoder1) + list(self.encoder2):
            _initialize_linear(encoder)

    def _weights(self, index, encoded1):
        raise NotImplementedError

    def forward(self, inputs):
        activations = inputs
        for index in range(len(self.hidden_dims)):
            activations = self.activation(self.linear[index](activations))
            encoded1 = self.activation(self.encoder1[index](inputs))
            encoded2 = self.activation(self.encoder2[index](inputs))
            weights = self._weights(index, encoded1)
            fused = weights[:, 0, :] * encoded1 + weights[:, 1, :] * encoded2
            activations = activations + self.residual_scales[index] * fused
        return self._output(activations)


class FixedAverageResidual(_TwoEncoderResidual):
    """Two coordinate encoders with a non-learned 0.5/0.5 fusion."""

    def _weights(self, index, encoded1):
        return encoded1.new_full((encoded1.size(0), 2, encoded1.size(1)), 0.5)


class StaticFeatureGateResidual(_TwoEncoderResidual):
    """Feature-wise softmax fusion that is learned but independent of each sample."""

    def __init__(self, layers, residual_scale=0.1):
        super().__init__(layers, residual_scale=residual_scale)
        self.static_gate_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(2, width)) for width in self.hidden_dims
        ])

    def _weights(self, index, encoded1):
        weights = torch.softmax(self.static_gate_logits[index], dim=0)
        return weights.unsqueeze(0).expand(encoded1.size(0), -1, -1)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _mlp_parameter_count(layers):
    return sum((input_width + 1) * output_width
               for input_width, output_width in zip(layers[:-1], layers[1:]))


def _stabilized_lda_parameter_count(layers):
    input_width = layers[0]
    hidden_dims = layers[1:-1]
    main_path = _mlp_parameter_count(layers)
    encoders = 2 * sum((input_width + 1) * width for width in hidden_dims)
    gates = sum((3 * width + 1) * width + (width + 1) * (2 * width)
                for width in hidden_dims)
    residual_scales = len(hidden_dims)
    return main_path + encoders + gates + residual_scales


def parameter_matched_mlp_layers(layers):
    """Find an equal hidden width that most closely matches full stabilized LDA."""
    hidden_count = len(layers) - 2
    if hidden_count < 1:
        raise ValueError('Parameter matching requires at least one hidden layer.')
    target = _stabilized_lda_parameter_count(layers)
    maximum_width = max(layers[1:-1]) * 10
    candidates = ([layers[0]] + [width] * hidden_count + [layers[-1]]
                  for width in range(1, maximum_width + 1))
    return min(candidates, key=lambda candidate: abs(_mlp_parameter_count(candidate) - target))


def canonical_architecture(name):
    normalized = name.strip().lower()
    aliases = {
        'mlp': 'mlp',
        'modified_mlp': 'modified_mlp',
        'wang_modified_mlp': 'modified_mlp',
        'fourier_mlp': 'fourier_mlp',
        'fourier_features': 'fourier_mlp',
        'mlp_param_matched': 'mlp_param_matched',
        'parameter_matched_mlp': 'mlp_param_matched',
        'direct_coordinate': 'direct_coordinate',
        'single_encoder_residual': 'single_encoder_residual',
        'fixed_average': 'fixed_average',
        'static_gate': 'static_gate',
        'lda': 'lda',
        'attention': 'lda',
        'separate_st_lda': 'separate_st_lda',
        'separate_space_time_lda': 'separate_st_lda',
    }
    if normalized not in aliases:
        raise ValueError(f'Unknown architecture: {name}')
    return aliases[normalized]


def build_model(architecture, layers, lda_stabilized=True, lda_residual_scale=0.1,
                fourier_feature_count=50, fourier_scale=1.0):
    architecture = canonical_architecture(architecture)
    effective_layers = list(layers)
    if architecture == 'mlp':
        model = MLP(layers)
    elif architecture == 'modified_mlp':
        model = ModifiedMLP(layers)
    elif architecture == 'fourier_mlp':
        model = FourierFeatureMLP(layers, fourier_feature_count, fourier_scale)
    elif architecture == 'mlp_param_matched':
        effective_layers = parameter_matched_mlp_layers(layers)
        model = MLP(effective_layers)
    elif architecture == 'direct_coordinate':
        model = DirectCoordinateInjection(layers, residual_scale=lda_residual_scale)
    elif architecture == 'single_encoder_residual':
        model = SingleEncoderResidual(layers, residual_scale=lda_residual_scale)
    elif architecture == 'fixed_average':
        model = FixedAverageResidual(layers, residual_scale=lda_residual_scale)
    elif architecture == 'static_gate':
        model = StaticFeatureGateResidual(layers, residual_scale=lda_residual_scale)
    elif architecture == 'separate_st_lda':
        model = SeparateSpaceTimeLDA(
            layers, stabilized=lda_stabilized, residual_scale=lda_residual_scale
        )
    else:
        model = LDA(layers, stabilized=lda_stabilized, residual_scale=lda_residual_scale)
    model.architecture = architecture
    model.requested_layers = list(layers)
    model.effective_layers = effective_layers
    return model

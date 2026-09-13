import torch


EPSILON = 1e-12


def _gradient_dot(first, second):
    return sum((left * right).sum() for left, right in zip(first, second))


def task_gradient_gram(task_gradients):
    """Return the task-by-task Gram matrix without materializing a flat vector."""
    if not task_gradients:
        raise ValueError('at least one task gradient is required')
    task_count = len(task_gradients)
    device = task_gradients[0][0].device
    dtype = task_gradients[0][0].dtype
    gram = torch.empty((task_count, task_count), dtype=dtype, device=device)
    for first in range(task_count):
        for second in range(first, task_count):
            value = _gradient_dot(task_gradients[first], task_gradients[second])
            gram[first, second] = value
            gram[second, first] = value
    return gram


def _weighted_sum(task_gradients, weights):
    return [
        sum(weights[task_index] * task_gradients[task_index][parameter_index]
            for task_index in range(len(task_gradients)))
        for parameter_index in range(len(task_gradients[0]))
    ]


def _project_simplex(vector):
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    sorted_vector, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_vector, dim=0) - 1.0
    indices = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    support = sorted_vector - cumulative / indices > 0
    rho = int(support.nonzero()[-1].item())
    theta = cumulative[rho] / (rho + 1)
    return torch.clamp(vector - theta, min=0.0)


def mgda_weights(task_gradients, max_iterations=100, tolerance=1e-8):
    """Solve the standard MGDA minimum-norm convex-combination problem.

    Frank--Wolfe is deterministic for a fixed Gram matrix and avoids adding a
    numerical-optimization dependency to the formal experiment environment.
    """
    gram = task_gradient_gram(task_gradients)
    task_count = gram.size(0)
    weights = torch.full((task_count,), 1.0 / task_count, dtype=gram.dtype, device=gram.device)
    for _ in range(max_iterations):
        gradient = 2.0 * gram.mv(weights)
        vertex = torch.zeros_like(weights)
        vertex[torch.argmin(gradient)] = 1.0
        direction = vertex - weights
        directional_derivative = torch.dot(direction, gradient)
        if abs(float(directional_derivative)) <= tolerance:
            break
        curvature = 2.0 * torch.dot(direction, gram.mv(direction))
        if float(curvature) <= EPSILON:
            step = 1.0
        else:
            step = torch.clamp(-directional_derivative / curvature, 0.0, 1.0)
        updated = weights + step * direction
        if torch.max(torch.abs(updated - weights)) <= tolerance:
            weights = updated
            break
        weights = updated
    return weights


def mgda_combined_gradients(task_gradients):
    weights = mgda_weights(task_gradients)
    return _weighted_sum(task_gradients, weights), weights


class GradNormBalancer:
    """Stateful GradNorm loss weighting with manual constrained-weight updates."""

    def __init__(self, task_count, alpha=1.5, learning_rate=0.025):
        if task_count < 2:
            raise ValueError('GradNorm requires at least two tasks.')
        self.task_count = task_count
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.initial_losses = None
        self.weights = None

    def update_and_combine(self, task_gradients, losses):
        if len(task_gradients) != self.task_count or len(losses) != self.task_count:
            raise ValueError('GradNorm task count must match losses and gradients.')
        device = task_gradients[0][0].device
        dtype = task_gradients[0][0].dtype
        if self.weights is None:
            self.weights = torch.ones(self.task_count, device=device, dtype=dtype)
        loss_values = torch.stack([loss.detach().to(device=device, dtype=dtype) for loss in losses])
        if self.initial_losses is None:
            self.initial_losses = loss_values.clamp_min(EPSILON)
        base_norms = torch.stack([
            torch.sqrt(_gradient_dot(gradients, gradients).clamp_min(EPSILON))
            for gradients in task_gradients
        ])
        weighted_norms = self.weights * base_norms
        relative_rates = loss_values / self.initial_losses
        inverse_training_rates = relative_rates / relative_rates.mean().clamp_min(EPSILON)
        targets = weighted_norms.mean().detach() * inverse_training_rates.pow(self.alpha)
        # This is d ||w_i * ||g_i|| - target_i|| / d w_i, with targets detached
        # as prescribed by GradNorm's alternating update.
        weight_gradient = torch.sign(weighted_norms - targets) * base_norms
        self.weights = torch.clamp(
            self.weights - self.learning_rate * weight_gradient, min=EPSILON
        )
        self.weights = self.task_count * self.weights / self.weights.sum().clamp_min(EPSILON)
        return _weighted_sum(task_gradients, self.weights), self.weights.detach().clone()


class GradientStatisticsLossBalancer:
    """Wang--Teng--Perdikaris learning-rate-annealing task weighting.

    The PDE task stays at unit weight.  Each auxiliary task receives the EMA
    of max(|grad PDE|) / mean(|grad auxiliary|), updated at a configured
    interval by the training script.
    """

    def __init__(self, task_count, ema=0.9):
        if task_count < 2 or not 0.0 <= ema < 1.0:
            raise ValueError('LRA requires at least two tasks and EMA in [0, 1).')
        self.task_count, self.ema, self.weights = task_count, float(ema), None

    def update(self, task_gradients):
        if len(task_gradients) != self.task_count:
            raise ValueError('LRA task count mismatch.')
        reference = torch.cat([gradient.detach().abs().reshape(-1) for gradient in task_gradients[0]]).max()
        targets = [reference]
        for gradients in task_gradients[1:]:
            values = torch.cat([gradient.detach().abs().reshape(-1) for gradient in gradients])
            targets.append(reference / values.mean().clamp_min(EPSILON))
        target = torch.stack(targets)
        target[0] = 1.0
        self.weights = target if self.weights is None else self.ema * self.weights + (1.0 - self.ema) * target
        self.weights[0] = 1.0
        return self.weights.detach().clone()


class ResidualBasedAttention:
    """Pointwise residual-based attention (RBA) from Anagnostopoulos et al.

    The persistent detached attention field is only meaningful while the
    collocation points retain their ordering. Training scripts therefore
    reject RBA together with epoch-wise resampling.
    """

    def __init__(self, eta=0.001, gamma=0.999):
        if eta <= 0.0 or not 0.0 <= gamma < 1.0:
            raise ValueError('RBA requires eta > 0 and gamma in [0, 1).')
        self.eta, self.gamma, self.weights = float(eta), float(gamma), None

    def update_and_weight(self, residual):
        magnitude = residual.detach().abs()
        normalized = self.eta * magnitude / magnitude.max().clamp_min(EPSILON)
        if self.weights is None:
            self.weights = normalized
        else:
            if self.weights.shape != normalized.shape:
                raise ValueError('RBA residual shape changed; use fixed collocation points.')
            self.weights = self.gamma * self.weights + normalized
        self.weights = self.weights.detach()
        # The reference implementation uses mean((w * r)**2), rather than
        # mean(w * r**2); preserve that exact convention here.
        return torch.mean((self.weights * residual) ** 2)


class BalancedResidualDecayRate:
    """Full-batch BRDR weighting and adaptive scaling from Zhang et al.

    The state implements the paper's Algorithm 1 without mini-batching:
    fourth-moment EMA with bias correction, pointwise IRDR weights normalized
    jointly across every loss component, and the post-backward scaling update.
    Like RBA, it requires persistent collocation-point ordering.
    """

    def __init__(self, beta_c=0.999, beta_w=0.999):
        if not 0.0 <= beta_c < 1.0 or not 0.0 <= beta_w < 1.0:
            raise ValueError('BRDR beta_c and beta_w must lie in [0, 1).')
        self.beta_c, self.beta_w = float(beta_c), float(beta_w)
        self.step, self.scale = 0, None
        self.moments, self.weights = None, None

    def update_and_weight(self, residuals):
        if not residuals:
            raise ValueError('BRDR requires at least one non-empty residual component.')
        detached = [residual.detach() for residual in residuals]
        if any(residual.numel() == 0 for residual in detached):
            raise ValueError('BRDR residual components must be non-empty.')
        if self.moments is None:
            self.moments = [torch.zeros_like(residual) for residual in detached]
            self.weights = [torch.ones_like(residual) for residual in detached]
            self.scale = detached[0].new_tensor(1.0)
        if len(detached) != len(self.moments) or any(
            residual.shape != moment.shape for residual, moment in zip(detached, self.moments)
        ):
            raise ValueError('BRDR residual shape changed; use fixed collocation points.')
        self.step += 1
        corrected_moments, inverse_decay_rates = [], []
        correction = 1.0 - self.beta_c ** self.step
        for residual, moment in zip(detached, self.moments):
            updated = self.beta_c * moment + (1.0 - self.beta_c) * residual.pow(4)
            corrected_moments.append(updated.detach())
            inverse_decay_rates.append(residual.square() / torch.sqrt(updated / correction + EPSILON))
        mean_rate = torch.cat([rate.reshape(-1) for rate in inverse_decay_rates]).mean().clamp_min(EPSILON)
        self.moments = corrected_moments
        self.weights = [
            (self.beta_w * weight + (1.0 - self.beta_w) * rate / mean_rate).detach()
            for weight, rate in zip(self.weights, inverse_decay_rates)
        ]
        return self.scale * sum(
            torch.mean(weight * residual.square())
            for weight, residual in zip(self.weights, residuals)
        )

    def correct_gradients(self, loss, parameters, learning_rate):
        """Apply the paper's adaptive scaling after ``loss.backward()``.

        ``loss`` includes the current scale. The resulting gradient correction
        is applied before the existing optimizer's ``step``.
        """
        if self.scale is None:
            raise RuntimeError('BRDR weights must be initialized before gradient correction.')
        if learning_rate <= 0.0:
            raise ValueError('BRDR requires a positive current learning rate.')
        squared_norm = sum(
            parameter.grad.detach().square().sum()
            for parameter in parameters if parameter.grad is not None
        )
        previous = self.scale
        updated = (1.0 - learning_rate) * previous + 2.0 * previous * loss.detach() / squared_norm.clamp_min(EPSILON)
        self.scale = updated.detach()
        ratio = self.scale / previous.clamp_min(EPSILON)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(ratio)
        return self.scale.detach().clone()


def hutchinson_ntk_traces(task_outputs, parameters, generator, max_outputs=128):
    """Estimate each output Jacobian NTK trace with one seeded Rademacher probe."""
    if max_outputs < 1:
        raise ValueError('max_outputs must be positive.')
    traces = []
    for outputs in task_outputs:
        flattened = outputs.reshape(-1)
        if flattened.numel() == 0:
            raise ValueError('NTK task outputs must be non-empty.')
        stride = max(1, flattened.numel() // max_outputs)
        selected = flattened[::stride][:max_outputs]
        signs = torch.randint(
            0, 2, (selected.numel(),), generator=generator, device='cpu', dtype=torch.int64
        ).to(selected.device, dtype=selected.dtype)
        signs = 2.0 * signs - 1.0
        gradients = torch.autograd.grad(
            torch.sum(signs * selected), parameters, retain_graph=True, allow_unused=True
        )
        squared_norm = sum(
            gradient.square().sum() if gradient is not None else torch.zeros((), device=selected.device)
            for gradient in gradients
        )
        traces.append(squared_norm * (flattened.numel() / selected.numel()))
    return torch.stack(traces)


def ntk_loss_weights(ntk_traces):
    """Trace-ratio NTK weighting: lambda_i = sum_j Tr(K_j) / Tr(K_i)."""
    if ntk_traces.ndim != 1 or ntk_traces.numel() < 2:
        raise ValueError('NTK weighting requires one trace per at least two tasks.')
    return ntk_traces.sum() / ntk_traces.clamp_min(EPSILON)


def project_conflicting_gradients(task_gradients, order_generator):
    """Shared PCGrad projection for a list of per-objective parameter gradients."""
    projected = [[gradient.clone() for gradient in task] for task in task_gradients]
    task_count = len(projected)
    for task_index in range(task_count):
        for other_index_tensor in torch.randperm(task_count, generator=order_generator):
            other_index = other_index_tensor.item()
            if task_index == other_index:
                continue
            dot_product = sum(
                (current * other).sum()
                for current, other in zip(projected[task_index], task_gradients[other_index])
            )
            if dot_product < 0:
                squared_norm = sum((other ** 2).sum() for other in task_gradients[other_index])
                if squared_norm > 0:
                    projected[task_index] = [
                        current - dot_product / squared_norm * other
                        for current, other in zip(projected[task_index], task_gradients[other_index])
                    ]
    return [sum(per_parameter) for per_parameter in zip(*projected)]


def canonical_optimizer(name):
    normalized = name.strip().lower()
    aliases = {
        'sum': 'sum', 'pinn': 'sum', 'pcgrad': 'pcgrad',
        'mgda': 'mgda', 'gradnorm': 'gradnorm',
        'ntk': 'ntk', 'ntk_weighting': 'ntk', 'lra': 'lra', 'rba': 'rba', 'brdr': 'brdr',
    }
    if normalized not in aliases:
        raise ValueError(f'Unknown optimizer strategy: {name}')
    return aliases[normalized]

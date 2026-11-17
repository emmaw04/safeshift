"""Unlabelled, online test-time adaptation for CIFAR-100-C experiments.

Classes: SourceOnly, BatchNormAdapt, Tent, CoTTA, ETA, EATA, SAR.

Every ``forward(images)`` returns DETACHED, PRE-UPDATE logits. A returned
prediction is never recomputed after adapting on the same target batch.
Target labels must NEVER be supplied to an adapter. ``predict_only`` performs
read-only inference for post-adaptation clean-set evaluation.

Research caveat: these are independently implemented, tested method variants,
not verified bitwise clones of the authors' official code/checkpoints. Log
checkpoint, preprocessing, protocol and hyperparameters with every experiment.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_NORM = {"bn": _BN, "gn": (nn.GroupNorm,), "ln": (nn.LayerNorm,)}


def prediction_entropy(logits: Tensor) -> Tensor:
    """Stable per-example predictive entropy [B], without labels."""
    z = logits.float()
    log_p = F.log_softmax(z, dim=1)
    return -(log_p.exp() * log_p).sum(dim=1)


def _norm_types(names: Sequence[str]) -> Tuple[type, ...]:
    if not names or any(name not in _NORM for name in names):
        raise ValueError("norm_types must be a nonempty sequence from ('bn','gn','ln')")
    return tuple(cls for n in names for cls in _NORM[n])


def _set_batch_stats(model: nn.Module) -> None:
    """Use current BN batch statistics, never maintain running statistics.

    Keep all non-BN modules in evaluation mode (including Dropout). This
    mutates the *adapter's private clone*, not the caller's source model.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, _BN):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
            m.train()


def _collect_affines(model: nn.Module, norm_types: Sequence[str],
                     excluded_prefixes: Sequence[str] = ()) -> List[nn.Parameter]:
    allowed = _norm_types(norm_types)
    for p in model.parameters():
        p.requires_grad_(False)
    result = []
    for name, module in model.named_modules():
        if isinstance(module, allowed) and not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in excluded_prefixes):
            for param in (module.weight, module.bias):
                if param is not None:
                    param.requires_grad_(True)
                    result.append(param)
    if not result:
        raise ValueError(f"No trainable normalisation affine parameters found for {norm_types}")
    return result


def _optimizer(params: Iterable[nn.Parameter], name: str, lr: float,
               momentum: float = 0.9) -> torch.optim.Optimizer:
    if lr <= 0:
        raise ValueError("lr must be positive")
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError("optimizer_name must be 'adam' or 'sgd'")


def _check_images(x: Tensor) -> None:
    if not torch.is_tensor(x) or x.ndim != 4 or x.shape[0] == 0:
        raise ValueError("Expected a nonempty image tensor [B,C,H,W]")


def _check_steps(steps: int) -> None:
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")


class BaseAdapter(nn.Module):
    """Owns independent student/source copies; reset and stream-state snapshots."""

    def __init__(self, backbone: nn.Module, *, episodic: bool = False) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        self._source = copy.deepcopy(backbone).eval()
        self._source.requires_grad_(False)
        self.model = copy.deepcopy(backbone)
        self.episodic = bool(episodic)
        self.n_batches = 0
        self.n_images = 0
        self.n_updates = 0
        self.last_diagnostics: Dict[str, Any] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def _setup(self) -> None:
        """Override to configure each new student and construct optimiser."""
        self.model.eval()

    def _restore(self) -> None:
        self.model = copy.deepcopy(self._source)
        self._setup()
        self.n_batches = self.n_images = self.n_updates = 0
        self.last_diagnostics = {}

    def reset(self) -> None:
        """Restore source weights, source buffers, optimizer and method state."""
        self._restore()

    def _adapt_predict(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        _check_images(x)
        if self.episodic:
            self.reset()
        z = self._adapt_predict(x)
        if z.ndim != 2 or z.shape[0] != x.shape[0] or not torch.isfinite(z).all():
            raise RuntimeError("Adapter produced invalid logits")
        self.n_batches += 1
        self.n_images += int(x.shape[0])
        return z.detach()

    def adapt_and_predict(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def predict_only(self, x: Tensor) -> Tensor:
        """Forward without adaptation (uses the method's normalisation policy).

        This operation never mutates weights/BN buffers; EMA-BN variant
        overrides it to read running statistics instead of updating them.
        """
        _check_images(x)
        with torch.no_grad():
            return self.model(x).detach()

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"batches": self.n_batches, "images": self.n_images,
                "optimizer_steps": self.n_updates, **self.last_diagnostics}

    def _extra_state(self) -> Dict[str, Any]:
        return {}

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        pass

    def adaptation_state_dict(self, *, include_rng: bool = False) -> Dict[str, Any]:
        """Save independently from nn.Module.state_dict (includes optimiser)."""
        state: Dict[str, Any] = {
            "method": self.__class__.__name__,
            "source": copy.deepcopy(self._source.state_dict()),
            "student": copy.deepcopy(self.model.state_dict()),
            "optimizer": (copy.deepcopy(self.optimizer.state_dict())
                          if self.optimizer is not None else None),
            "counters": (self.n_batches, self.n_images, self.n_updates),
            "diagnostics": copy.deepcopy(self.last_diagnostics),
            "extra": self._extra_state(),
        }
        if include_rng:
            state["cpu_rng"] = torch.get_rng_state().clone()
            if torch.cuda.is_available():
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
        return state

    def load_adaptation_state_dict(self, state: Mapping[str, Any],
                                   *, restore_rng: bool = False) -> None:
        if state["method"] != self.__class__.__name__:
            raise ValueError("Adaptation checkpoint was created by a different method")
        self._source.load_state_dict(state["source"], strict=True)
        self.reset()
        self.model.load_state_dict(state["student"], strict=True)
        if self.optimizer is not None:
            if state["optimizer"] is None:
                raise ValueError("Checkpoint is missing the optimizer state")
            self.optimizer.load_state_dict(state["optimizer"])
        self.n_batches, self.n_images, self.n_updates = state["counters"]
        self.last_diagnostics = copy.deepcopy(state["diagnostics"])
        self._load_extra_state(state["extra"])
        if restore_rng:
            if "cpu_rng" not in state:
                raise ValueError("Checkpoint has no saved RNG state")
            torch.set_rng_state(state["cpu_rng"].cpu())
            if "cuda_rng" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])


class SourceOnly(BaseAdapter):
    def __init__(self, backbone: nn.Module, *, episodic: bool = False) -> None:
        super().__init__(backbone, episodic=episodic)
        self.reset()

    def _setup(self) -> None:
        self.model.eval().requires_grad_(False)
        self.optimizer = None

    def _adapt_predict(self, x: Tensor) -> Tensor:
        return self.predict_only(x)

    def predict_only(self, x: Tensor) -> Tensor:
        self.model.eval()  # source BN buffers must never change, even after .train()
        return super().predict_only(x)


class BatchNormAdapt(BaseAdapter):
    """BN test-batch (default) or momentum-EMA statistics, no gradient updates."""
    def __init__(self, backbone: nn.Module, *, mode: str = "batch",
                 momentum: float = 0.1, episodic: bool = False) -> None:
        super().__init__(backbone, episodic=episodic)
        if mode not in {"batch", "ema"} or not 0 < momentum <= 1:
            raise ValueError("mode must be 'batch'/'ema', momentum in (0,1]")
        self.mode, self.momentum = mode, momentum
        self.reset()

    def _setup(self) -> None:
        if not any(isinstance(m, _BN) for m in self.model.modules()):
            raise ValueError("BatchNormAdapt requires a model with BatchNorm layers")
        self.model.eval().requires_grad_(False)
        if self.mode == "batch":
            _set_batch_stats(self.model)
        else:
            for m in self.model.modules():
                if isinstance(m, _BN):
                    m.track_running_stats = True
                    m.momentum = self.momentum
                    m.train()  # updates BN buffers, not weights; dropout stays eval
        self.optimizer = None

    def _adapt_predict(self, x: Tensor) -> Tensor:
        return self.predict_only(x) if self.mode == "batch" else self._ema_forward(x)

    def _ema_forward(self, x: Tensor) -> Tensor:
        # Ensure BN buffer updates even if evaluator called adapter.eval().
        for m in self.model.modules():
            if isinstance(m, _BN):
                m.train()
        with torch.no_grad():
            return self.model(x).detach()

    def predict_only(self, x: Tensor) -> Tensor:
        _check_images(x)
        if self.mode == "batch":
            return super().predict_only(x)
        layers = [m for m in self.model.modules() if isinstance(m, _BN)]
        was_training = [m.training for m in layers]
        try:
            for m in layers:
                m.eval()
            with torch.no_grad():
                return self.model(x).detach()
        finally:
            for m, prev in zip(layers, was_training):
                m.train(prev)


class Tent(BaseAdapter):
    """Wang et al. ICLR'21: test entropy minimisation of BN affine weights."""
    def __init__(self, backbone: nn.Module, *, lr: float = 1e-3,
                 optimizer_name: str = "adam", steps: int = 1,
                 episodic: bool = False, norm_types: Sequence[str] = ("bn",)) -> None:
        super().__init__(backbone, episodic=episodic)
        _check_steps(steps)
        self.lr, self.optimizer_name, self.steps = lr, optimizer_name, steps
        self.norm_types = tuple(norm_types)
        self.reset()

    def _setup(self) -> None:
        self.model.eval()
        if "bn" in self.norm_types:
            _set_batch_stats(self.model)
        params = _collect_affines(self.model, self.norm_types)
        self.optimizer = _optimizer(params, self.optimizer_name, self.lr)

    def _adapt_predict(self, x: Tensor) -> Tensor:
        pre = None
        with torch.enable_grad():
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                z = self.model(x)
                if pre is None:
                    pre = z.detach()
                loss = prediction_entropy(z).mean()
                loss.backward()
                self.optimizer.step()
                self.n_updates += 1
        self.last_diagnostics = {"entropy_loss": float(loss.detach())}
        return pre


class CoTTA(BaseAdapter):
    """Wang et al. CVPR'22: student, EMA teacher, source anchor, restoration.

    CIFAR reference defaults: EMA=.999; restore=.01; gate=.92;
    augmentations=32; Adam lr=.001. The built-in augmentation magnitudes
    are a documented reimplementation, not an exact official torchvision
    Compose clone. Input must be raw [0,1] or explicitly normalised.
    """
    def __init__(self, backbone: nn.Module, *, lr: float = 1e-3,
                 optimizer_name: str = "adam", ema_alpha: float = 0.999,
                 restore_prob: float = 0.01, aug_confidence_threshold: float = 0.92,
                 num_augmentations: int = 32, steps: int = 1,
                 episodic: bool = False, input_space: str = "zero_one",
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 augment_fn: Optional[Any] = None,
                 use_test_batch_stats: bool = True) -> None:
        super().__init__(backbone, episodic=episodic)
        _check_steps(steps)
        if not 0 <= ema_alpha < 1 or not 0 <= restore_prob <= 1:
            raise ValueError("ema_alpha must be in [0,1); restore_prob in [0,1]")
        if not 0 <= aug_confidence_threshold <= 1 or num_augmentations < 1:
            raise ValueError("Invalid augmentation confidence threshold/count")
        if input_space not in ("zero_one", "normalized"):
            raise ValueError("input_space must be 'zero_one' or 'normalized'")
        if input_space == "normalized" and (mean is None or std is None):
            raise ValueError("Specify mean and std to augment normalised images")
        if std is not None and any(s <= 0 for s in std):
            raise ValueError("All std entries must be positive")
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.ema_alpha = ema_alpha
        self.restore_prob = restore_prob
        self.aug_confidence_threshold = aug_confidence_threshold
        self.num_augmentations = num_augmentations
        self.steps = steps
        self.input_space = input_space
        self.mean = tuple(mean) if mean is not None else None
        self.std = tuple(std) if std is not None else None
        self.augment_fn = augment_fn
        self.use_test_batch_stats = use_test_batch_stats
        self.teacher: nn.Module
        self.reset()

    @property
    def anchor(self) -> nn.Module:
        return self._source

    def _setup(self) -> None:
        self.model.eval().requires_grad_(True)
        self.teacher = copy.deepcopy(self._source).eval().requires_grad_(False)
        if self.use_test_batch_stats:
            _set_batch_stats(self.model)
            _set_batch_stats(self.teacher)
        self.optimizer = _optimizer(self.model.parameters(), self.optimizer_name, self.lr)

    def _to_raw(self, x: Tensor) -> Tensor:
        if self.input_space == "zero_one":
            return x.clamp(0, 1)
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std = x.new_tensor(self.std).view(1, -1, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def _from_raw(self, x: Tensor) -> Tensor:
        if self.input_space == "zero_one":
            return x
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std = x.new_tensor(self.std).view(1, -1, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _default_augment(raw: Tensor) -> Tensor:
        """On-device differentiability-free stochastic CIFAR augmentation.

        Approximate CoTTA color jitter, affine, Gaussian blur, flip, noise.
        Applies the same sampled transform to a batch for efficiency; each
        of the independently sampled augmented views differs.
        """
        b, c, _, _ = raw.shape
        t = raw
        def uniform(low: float, high: float) -> float:
            return float(torch.empty(()).uniform_(low, high))
        # brightness, contrast, saturation, gamma
        t = t * uniform(0.8, 1.2)
        channel_mean = t.mean(dim=(2, 3), keepdim=True)
        t = (t - channel_mean) * uniform(0.8, 1.2) + channel_mean
        if c == 3:
            grey = (0.2989 * t[:, 0:1] + 0.587 * t[:, 1:2] +
                    0.114 * t[:, 2:3])
            t = (t - grey) * uniform(0.8, 1.2) + grey
        t = t.clamp(0, 1).pow(uniform(0.8, 1.2))
        if torch.rand(()) < 0.5:
            t = torch.flip(t, dims=(-1,))
        # small rotation, scale and translation through affine_grid/grid_sample
        angle = uniform(-8, 8) * math.pi / 180
        scale = uniform(0.9, 1.1)
        dx, dy = uniform(-0.08, 0.08), uniform(-0.08, 0.08)
        matrix = t.new_tensor([[scale*math.cos(angle), -scale*math.sin(angle), dx],
                               [scale*math.sin(angle), scale*math.cos(angle), dy]])
        grid = F.affine_grid(matrix.unsqueeze(0).expand(b, -1, -1), t.shape,
                             align_corners=False)
        t = F.grid_sample(t, grid, align_corners=False,
                          padding_mode="reflection", mode="bilinear")
        if torch.rand(()) < 0.5:
            sigma = uniform(0.1, 1.2)
            coords = torch.arange(-2, 3, device=t.device, dtype=t.dtype)
            k = torch.exp(-0.5 * (coords / sigma) ** 2)
            k = k / k.sum()
            kernel = torch.outer(k, k).expand(c, 1, 5, 5)
            t = F.conv2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), kernel,
                         groups=c)
        t = t + torch.randn_like(t) * uniform(0.0, 0.03)
        return t.clamp(0, 1)

    def _augment(self, x: Tensor) -> Tensor:
        raw = self._to_raw(x)
        if self.augment_fn is not None:
            aug_raw = self.augment_fn(raw)
        else:
            aug_raw = self._default_augment(raw)
        if aug_raw.shape != x.shape:
            raise ValueError("augment_fn must preserve [B,C,H,W] shape")
        return self._from_raw(aug_raw.clamp(0, 1))

    def _ema_update(self) -> None:
        with torch.no_grad():
            for teacher_p, student_p in zip(self.teacher.parameters(),
                                            self.model.parameters()):
                teacher_p.lerp_(student_p, 1.0 - self.ema_alpha)
            # If running buffers exist, copy the student counters/statistics.
            teacher_buffers = dict(self.teacher.named_buffers())
            for name, student_buf in self.model.named_buffers():
                if name in teacher_buffers:
                    teacher_buffers[name].copy_(student_buf)

    def _stochastic_restore(self) -> None:
        with torch.no_grad():
            for param, source_param in zip(self.model.parameters(),
                                           self._source.parameters()):
                if param.requires_grad:
                    mask = torch.rand(param.shape, device=param.device) < self.restore_prob
                    param.copy_(torch.where(mask, source_param, param))

    def _adapt_predict(self, x: Tensor) -> Tensor:
        first_teacher = None
        used_augs = False
        for _ in range(self.steps):
            with torch.enable_grad():
                self.optimizer.zero_grad(set_to_none=True)
                student_logits = self.model(x)
            with torch.no_grad():
                anchor_probs = self.anchor(x).float().softmax(dim=1)
                anchor_conf = float(anchor_probs.max(1).values.mean())
                if anchor_conf < self.aug_confidence_threshold:
                    logits_teacher = torch.stack([
                        self.teacher(self._augment(x)).detach()
                        for _ in range(self.num_augmentations)], dim=0).mean(dim=0)
                    used_augs = True
                else:
                    logits_teacher = self.teacher(x).detach()
            if first_teacher is None:
                first_teacher = logits_teacher.detach()
            teacher_probs = logits_teacher.float().softmax(dim=1).detach()
            with torch.enable_grad():
                loss = -(teacher_probs * F.log_softmax(student_logits.float(), 1)).sum(1).mean()
                loss.backward()
                self.optimizer.step()
            self._ema_update()
            self._stochastic_restore()
            self.n_updates += 1
        self.last_diagnostics = {"loss": float(loss.detach()),
                                 "anchor_confidence": anchor_conf,
                                 "used_augmentations": used_augs,
                                 "prediction_source": "pre_update_teacher"}
        return first_teacher

    def predict_only(self, x: Tensor) -> Tensor:
        _check_images(x)
        with torch.no_grad():
            return self.teacher(x).detach()

    def _extra_state(self) -> Dict[str, Any]:
        return {"teacher": copy.deepcopy(self.teacher.state_dict())}

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        self.teacher.load_state_dict(state["teacher"], strict=True)


class ETA(BaseAdapter):
    """Efficient entropy adaptation (EATA without Fisher penalty)."""
    def __init__(self, backbone: nn.Module, *, lr: float = 1e-3,
                 optimizer_name: str = "adam", e_margin: Optional[float] = None,
                 d_margin: float = 0.05, prob_ema_momentum: float = 0.9,
                 steps: int = 1, episodic: bool = False,
                 norm_types: Sequence[str] = ("bn",)) -> None:
        super().__init__(backbone, episodic=episodic)
        _check_steps(steps)
        if not 0 <= prob_ema_momentum < 1 or not 0 <= d_margin <= 1:
            raise ValueError("Invalid probability EMA momentum or diversity margin")
        self.lr, self.optimizer_name, self.steps = lr, optimizer_name, steps
        self.e_margin, self.d_margin = e_margin, d_margin
        self.prob_ema_momentum = prob_ema_momentum
        self.norm_types = tuple(norm_types)
        self.prob_ema: Optional[Tensor] = None
        self.reset()

    def _setup(self) -> None:
        self.model.eval()
        if "bn" in self.norm_types:
            _set_batch_stats(self.model)
        params = _collect_affines(self.model, self.norm_types)
        self.optimizer = _optimizer(params, self.optimizer_name, self.lr)
        self.prob_ema = None

    def _regularizer(self) -> Tensor:
        return next(self.model.parameters()).new_zeros(())

    def _adapt_predict(self, x: Tensor) -> Tensor:
        first = None
        selected_count = 0
        reliable_count = 0
        last_loss = None
        for _ in range(self.steps):
            with torch.enable_grad():
                self.optimizer.zero_grad(set_to_none=True)
                z = self.model(x)
                if first is None:
                    first = z.detach()
                entropy = prediction_entropy(z)
                limit = (self.e_margin if self.e_margin is not None
                         else 0.4 * math.log(z.shape[1]))
                reliable = (entropy.detach() < limit)
                reliable_count = int(reliable.sum())
                selected = reliable.clone()
                probs = z.detach().float().softmax(1)
                if self.prob_ema is not None:
                    similarity = F.cosine_similarity(
                        probs, self.prob_ema.to(probs.device).unsqueeze(0), dim=1).abs()
                    selected &= similarity < self.d_margin
                selected_count = int(selected.sum())
                if selected_count == 0:
                    continue
                weights = torch.exp(limit - entropy[selected].detach())
                loss = (weights * entropy[selected]).mean() + self._regularizer()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite ETA/EATA adaptation loss")
                loss.backward()
                self.optimizer.step()
                self.n_updates += 1
                last_loss = float(loss.detach())
                new_mean = probs[selected].mean(dim=0)
                if self.prob_ema is None:
                    self.prob_ema = new_mean.detach().clone()
                else:
                    self.prob_ema = self.prob_ema.to(new_mean.device)
                    self.prob_ema.mul_(self.prob_ema_momentum).add_(
                        new_mean, alpha=1.0 - self.prob_ema_momentum)
        self.last_diagnostics = {"reliable": reliable_count,
                                 "selected": selected_count, "loss": last_loss}
        return first

    def _extra_state(self) -> Dict[str, Any]:
        return {"prob_ema": None if self.prob_ema is None else self.prob_ema.clone()}

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        self.prob_ema = (None if state["prob_ema"] is None else
                         state["prob_ema"].clone())


class EATA(ETA):
    """ETA with Fisher-weighted source parameter anchoring.

    Must receive a diagonal Fisher estimate computed on a separate, UNLABELLED
    in-distribution calibration set. Never uses CIFAR-100-C ground-truth labels.
    The coefficient is intentionally tunable for CIFAR-100 research extensions.
    """
    def __init__(self, backbone: nn.Module, *, fisher: Mapping[str, Tensor],
                 fisher_alpha: float = 2000.0,
                 source_parameters: Optional[Mapping[str, Tensor]] = None,
                 **kwargs: Any) -> None:
        if not fisher:
            raise ValueError("EATA requires a nonempty Fisher estimate; use ETA otherwise")
        if fisher_alpha < 0:
            raise ValueError("fisher_alpha must be non-negative")
        self.fisher_alpha = fisher_alpha
        self._incoming_fisher = fisher
        self._incoming_sources = source_parameters
        super().__init__(backbone, **kwargs)

    def _setup(self) -> None:
        super()._setup()
        self.fisher: Dict[str, Tensor] = {}
        self.source_parameters: Dict[str, Tensor] = {}
        param_dict = dict(self.model.named_parameters())
        originals = (self._incoming_sources if self._incoming_sources is not None else
                     dict(self._source.named_parameters()))
        for name, param in param_dict.items():
            if param.requires_grad:
                if name not in self._incoming_fisher or name not in originals:
                    raise ValueError(f"Missing Fisher or source parameter for {name}")
                f = self._incoming_fisher[name].detach().to(param.device, param.dtype)
                source = originals[name].detach().to(param.device, param.dtype)
                if f.shape != param.shape or source.shape != param.shape or (f < 0).any():
                    raise ValueError(f"Invalid Fisher/source tensor for {name}")
                self.fisher[name] = f.clone()
                self.source_parameters[name] = source.clone()

    def _regularizer(self) -> Tensor:
        total = next(self.model.parameters()).new_zeros(())
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name].to(device=param.device, dtype=param.dtype)
                source = self.source_parameters[name].to(device=param.device,
                                                         dtype=param.dtype)
                total = total + (fisher * (param - source).square()).sum()
        return self.fisher_alpha * total

    def _extra_state(self) -> Dict[str, Any]:
        return {**super()._extra_state(),
                "fisher": copy.deepcopy(self.fisher),
                "source_parameters": copy.deepcopy(self.source_parameters)}

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        super()._load_extra_state(state)
        self.fisher = copy.deepcopy(state["fisher"])
        self.source_parameters = copy.deepcopy(state["source_parameters"])


def estimate_fisher_from_unlabeled(
        model: nn.Module, calibration_loader: Iterable[Any], *,
        max_samples: int = 2000, norm_types: Sequence[str] = ("bn",),
        norm_policy: str = "batch") -> Dict[str, Tensor]:
    """Average squared pseudo-label batch gradients (EATA-style approximation).

    Ignores every label in calibration_loader; only its image tensors are read.
    Provide *unlabelled in-distribution calibration images* and record their
    provenance. This is not a per-example Fisher estimator. No model updates.
    """
    if max_samples < 1 or norm_policy not in ("batch", "source"):
        raise ValueError("max_samples >= 1 and norm_policy='batch'/'source' required")
    clone = copy.deepcopy(model).eval()
    if norm_policy == "batch" and "bn" in norm_types:
        _set_batch_stats(clone)
    params = _collect_affines(clone, norm_types)
    named_params = {name: p for name, p in clone.named_parameters() if p.requires_grad}
    sums = {name: torch.zeros_like(p) for name, p in named_params.items()}
    n_batches, n_seen = 0, 0
    device = next(clone.parameters()).device
    for batch in calibration_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not torch.is_tensor(x):
            raise TypeError("Calibration loader must yield image tensors")
        if n_seen >= max_samples:
            break
        x = x[:max_samples-n_seen].to(device)
        if len(x) == 0:
            continue
        with torch.no_grad():
            pseudo_y = clone(x).argmax(dim=1)
        with torch.enable_grad():
            clone.zero_grad(set_to_none=True)
            loss = F.cross_entropy(clone(x), pseudo_y)
            grads = torch.autograd.grad(loss, tuple(named_params.values()))
        for (name, _), gradient in zip(named_params.items(), grads):
            sums[name] += gradient.detach().square()
        n_seen += x.shape[0]
        n_batches += 1
    if n_batches == 0:
        raise ValueError("Calibration loader returned no images")
    return {name: value / n_batches for name, value in sums.items()}


class SAR(BaseAdapter):
    """Niu et al. ICLR'23: reliable entropy + two-pass SAM + recovery.

    Defaults to GroupNorm/LayerNorm, with final ResNet stage excluded.
    On CIFAR-100 these hyperparameters are research extensions of the
    authors' ImageNet experiments, not claimed paper-reproduction values.
    """
    def __init__(self, backbone: nn.Module, *, lr: float = 2.5e-4,
                 rho: float = 0.05, e_margin: Optional[float] = None,
                 recovery_entropy_threshold: float = 0.2,
                 entropy_ema_decay: float = 0.9, momentum: float = 0.9,
                 norm_types: Sequence[str] = ("gn", "ln"),
                 freeze_final_stage: bool = True, steps: int = 1,
                 episodic: bool = False) -> None:
        super().__init__(backbone, episodic=episodic)
        _check_steps(steps)
        if rho <= 0 or not 0 <= entropy_ema_decay < 1 or momentum < 0:
            raise ValueError("rho must be positive; invalid EMA decay/momentum")
        if recovery_entropy_threshold < 0:
            raise ValueError("recovery_entropy_threshold must be >= 0")
        self.lr, self.rho, self.e_margin = lr, rho, e_margin
        self.recovery_entropy_threshold = recovery_entropy_threshold
        self.entropy_ema_decay, self.momentum = entropy_ema_decay, momentum
        self.norm_types, self.freeze_final_stage = tuple(norm_types), freeze_final_stage
        self.steps = steps
        self.entropy_ema: Optional[float] = None
        self.n_recoveries = 0
        self.reset()

    def _setup(self) -> None:
        self.model.eval()
        if "bn" in self.norm_types:
            _set_batch_stats(self.model)
        excluded: List[str] = []
        if self.freeze_final_stage:
            candidates = [i for i in range(1, 9) if hasattr(self.model, f"layer{i}")]
            if candidates:
                excluded = [f"layer{max(candidates)}"]
        params = _collect_affines(self.model, self.norm_types, excluded)
        self._adapt_params = params
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        self.entropy_ema = None
        self.n_recoveries = 0

    def _recover(self) -> None:
        """Reset model/optimiser but preserve stream accounting."""
        batches, images, updates, recoveries = (self.n_batches, self.n_images,
                                                self.n_updates, self.n_recoveries)
        self.model = copy.deepcopy(self._source)
        self._setup()
        self.n_batches, self.n_images, self.n_updates = batches, images, updates
        self.n_recoveries = recoveries + 1

    def _adapt_predict(self, x: Tensor) -> Tensor:
        pre = None
        selected1 = selected2 = 0
        did_recover = False
        last_loss = None
        for _ in range(self.steps):
            with torch.enable_grad():
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                if pre is None:
                    pre = logits.detach()
                h = prediction_entropy(logits)
                limit = (self.e_margin if self.e_margin is not None
                         else 0.4 * math.log(logits.shape[1]))
                mask1 = h.detach() < limit
                selected1 = int(mask1.sum())
                if selected1 == 0:
                    continue
                loss1 = h[mask1].mean()
                loss1.backward()
                grads = [p.grad for p in self._adapt_params if p.grad is not None]
                if not grads:
                    continue
                squared_norm = sum(g.detach().float().square().sum() for g in grads)
                grad_norm = torch.sqrt(squared_norm)
                if not bool(torch.isfinite(grad_norm)):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("Nonfinite SAM gradient norm")
                originals: List[Tensor] = [p.detach().clone() for p in self._adapt_params]
                with torch.no_grad():
                    scale = self.rho / (float(grad_norm) + 1e-12)
                    for p in self._adapt_params:
                        if p.grad is not None:
                            p.add_(p.grad, alpha=scale)
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    logits2 = self.model(x)
                    h2 = prediction_entropy(logits2)
                    mask2 = mask1 & (h2.detach() < limit)
                    selected2 = int(mask2.sum())
                    if selected2:
                        loss2 = h2[mask2].mean()
                        if not torch.isfinite(loss2):
                            raise RuntimeError("Nonfinite second-pass SAR loss")
                        loss2.backward()
                        last_loss = float(loss2.detach())
                finally:
                    with torch.no_grad():
                        for p, original in zip(self._adapt_params, originals):
                            p.copy_(original)
                if selected2 == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                self.n_updates += 1
                self.entropy_ema = (last_loss if self.entropy_ema is None else
                                    self.entropy_ema_decay * self.entropy_ema +
                                    (1.0 - self.entropy_ema_decay) * last_loss)
                if self.entropy_ema < self.recovery_entropy_threshold:
                    self._recover()
                    did_recover = True
                    break
        self.last_diagnostics = {"reliable_first": selected1,
                                 "reliable_second": selected2,
                                 "entropy_loss": last_loss,
                                 "entropy_ema": self.entropy_ema,
                                 "recoveries": self.n_recoveries,
                                 "recovered_on_batch": did_recover}
        return pre

    def _extra_state(self) -> Dict[str, Any]:
        return {"entropy_ema": self.entropy_ema, "n_recoveries": self.n_recoveries}

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        self.entropy_ema = state["entropy_ema"]
        self.n_recoveries = state["n_recoveries"]


ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "source": SourceOnly, "bn_adapt": BatchNormAdapt, "tent": Tent,
    "cotta": CoTTA, "eta": ETA, "eata": EATA, "sar": SAR,
}


def list_adapters() -> List[str]:
    return sorted(ADAPTER_REGISTRY)


def create_adapter(name: str, backbone: nn.Module, **kwargs: Any) -> BaseAdapter:
    try:
        constructor = ADAPTER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {name!r}; available: {', '.join(list_adapters())}") from exc
    return constructor(backbone, **kwargs)


__all__ = ["BaseAdapter", "SourceOnly", "BatchNormAdapt", "Tent", "CoTTA",
           "ETA", "EATA", "SAR", "prediction_entropy", "estimate_fisher_from_unlabeled",
           "ADAPTER_REGISTRY", "list_adapters", "create_adapter"]

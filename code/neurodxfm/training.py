import json
import logging
import math
import os
import random
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .data import VolumeBatch
from .losses import FiveHeadObjective, LossBatch, LossValues
from .model import NeuroDxFM
from .settings import ExperimentSettings

LOGGER = logging.getLogger("neurodxfm")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_ready() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_ready() else 1


def initialize_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    process_rank = int(os.environ.get("RANK", "0"))
    processes = int(os.environ.get("WORLD_SIZE", "1"))
    if processes > 1 and not distributed_ready():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return local_rank, process_rank, processes


def reduce_mean(value: Tensor) -> Tensor:
    if not distributed_ready():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result / world_size()


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, minimum_ratio: float = 0.0) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.minimum_ratio = minimum_ratio
        self.step_index = 0
        self.base = [float(group["lr"]) for group in optimizer.param_groups]

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine

    def step(self) -> None:
        factor = self.factor(self.step_index)
        for base, group in zip(self.base, self.optimizer.param_groups, strict=True):
            group["lr"] = base * factor
        self.step_index += 1

    def state_dict(self) -> dict[str, object]:
        return {"step_index": self.step_index, "base": self.base}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.step_index = int(state["step_index"])
        self.base = [float(item) for item in state["base"]]


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, momentum: float) -> None:
        self.momentum = momentum
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.momentum).add_(value.detach(), alpha=1.0 - self.momentum)
            else:
                self.shadow[name].copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, object]:
        return {"momentum": self.momentum, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.momentum = float(state["momentum"])
        self.shadow = dict(state["shadow"])


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    best_validation_auc: float = 0.0
    stale_epochs: int = 0


def atomic_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def capture_random_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosine,
    scaler: torch.amp.GradScaler,
    ema: ExponentialMovingAverage,
    state: TrainingState,
    settings: ExperimentSettings,
) -> None:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    payload: dict[str, object] = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "training_state": asdict(state),
        "settings": asdict(settings),
        "seed": settings.runtime.seed,
        "random_state": capture_random_state(),
    }
    atomic_save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosine,
    scaler: torch.amp.GradScaler,
    ema: ExponentialMovingAverage,
) -> TrainingState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    ema.load_state_dict(payload["ema"])
    restore_random_state(payload["random_state"])
    return TrainingState(**payload["training_state"])


def build_optimizer(model: nn.Module, settings: ExperimentSettings) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings.optimizer.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=settings.optimizer.learning_rate,
        betas=(settings.optimizer.beta1, settings.optimizer.beta2),
    )


class Trainer:
    def __init__(self, model: NeuroDxFM, objective: FiveHeadObjective, settings: ExperimentSettings, device: torch.device) -> None:
        self.settings = settings
        self.device = device
        self.model: nn.Module = model.to(device)
        if distributed_ready():
            self.model = DistributedDataParallel(self.model, device_ids=[device.index], find_unused_parameters=True)
        self.objective = objective.to(device)
        self.optimizer = build_optimizer(self.model, settings)
        total = settings.schedule.epochs * settings.schedule.steps_per_epoch // settings.runtime.gradient_accumulation
        warmup = settings.schedule.warmup_epochs * settings.schedule.steps_per_epoch // settings.runtime.gradient_accumulation
        self.scheduler = WarmupCosine(self.optimizer, warmup, total)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        raw_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.ema = ExponentialMovingAverage(raw_model, settings.model.ema_momentum)
        self.state = TrainingState()

    def _objective(self, batch: VolumeBatch, epoch: int) -> LossValues:
        original = self.model(batch.volumes, batch.months, self.settings.loss.grl)
        counterfactual = self.model(batch.counterfactual, None, self.settings.loss.grl)
        future = counterfactual.representation.detach() if epoch > self.settings.schedule.phase_boundary else None
        loss_batch = LossBatch(
            batch.protocol,
            future,
            batch.reconstruction_target,
            batch.missing_modalities,
            batch.anatomy,
            batch.anatomy_valid,
            batch.amyloid,
            batch.amyloid_valid,
            batch.diagnosis,
        )
        return self.objective(original, counterfactual, loss_batch, epoch)

    def train_epoch(self, batches: Iterable[VolumeBatch], epoch: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.settings.runtime.gradient_accumulation
        for step, host_batch in enumerate(batches):
            batch = host_batch.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                losses = self._objective(batch, epoch)
                scaled_loss = losses.total / accumulation
            self.scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.optimizer.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                raw_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
                self.ema.update(raw_model)
                self.state.optimizer_step += 1
            for name, value in losses.detached().items():
                totals[name] = totals.get(name, 0.0) + value
            count += 1
            self.state.global_step += 1
        self.state.epoch = epoch
        return {name: value / max(count, 1) for name, value in totals.items()}

    def fit(self, batches: Iterable[VolumeBatch]) -> None:
        for epoch in range(self.state.epoch + 1, self.settings.schedule.epochs + 1):
            metrics = self.train_epoch(batches, epoch)
            if rank() == 0:
                LOGGER.info("epoch=%d metrics=%s", epoch, json.dumps(metrics, sort_keys=True))
                if epoch % self.settings.runtime.checkpoint_interval == 0:
                    save_checkpoint(
                        self.settings.runtime.output_directory / f"epoch_{epoch:04d}.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        self.ema,
                        self.state,
                        self.settings,
                    )

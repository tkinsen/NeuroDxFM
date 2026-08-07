import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .configuration import load_settings
from .data import NeuroVolumeDataset, collate_volume_batch, read_manifest
from .losses import FiveHeadObjective
from .model import NeuroDxFM
from .training import Trainer, initialize_distributed, load_checkpoint, set_seed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="neurodxfm")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--override", action="append", default=[])
    train.add_argument("--resume", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--override", action="append", default=[])
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    return root


def build_loader(configuration_path: Path, overrides: list[str], training: bool) -> tuple[object, DataLoader[object]]:
    settings = load_settings(configuration_path, overrides)
    records = read_manifest(settings.data.manifest)
    dataset = NeuroVolumeDataset(records, settings.data, training, settings.runtime.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.runtime.batch_size_per_gpu,
        shuffle=training,
        num_workers=settings.runtime.workers_per_gpu,
        pin_memory=True,
        drop_last=training,
        collate_fn=collate_volume_batch,
        persistent_workers=settings.runtime.workers_per_gpu > 0,
    )
    return settings, loader


def train_command(arguments: argparse.Namespace) -> int:
    local_rank, _, _ = initialize_distributed()
    settings, loader = build_loader(arguments.config, arguments.override, True)
    set_seed(settings.runtime.seed, settings.runtime.deterministic)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = NeuroDxFM(settings.model)
    objective = FiveHeadObjective(settings.loss, settings.model.temperature)
    trainer = Trainer(model, objective, settings, device)
    if arguments.resume is not None:
        trainer.state = load_checkpoint(
            arguments.resume,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.scaler,
            trainer.ema,
        )
    trainer.fit(loader)
    return 0


def evaluate_command(arguments: argparse.Namespace) -> int:
    settings, loader = build_loader(arguments.config, arguments.override, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroDxFM(settings.model).to(device)
    payload = torch.load(arguments.weights, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.volumes)
            probability = output.evidence / output.evidence.sum(dim=-1, keepdim=True)
            uncertainty = output.evidence.shape[-1] / output.evidence.sum(dim=-1)
            for index, subject in enumerate(batch.subject_ids):
                rows.append(
                    {
                        "subject_id": subject,
                        "target": int(batch.diagnosis[index]),
                        "probability": float(probability[index, 1]),
                        "uncertainty": float(uncertainty[index]),
                    }
                )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id", "target", "probability", "uncertainty"))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def validate_manifest_command(arguments: argparse.Namespace) -> int:
    records = read_manifest(arguments.manifest)
    summary = {
        "records": len(records),
        "subjects": len({record.subject_id for record in records}),
        "cohorts": sorted({record.cohort for record in records}),
        "sites": len({record.site for record in records}),
        "protocols": len({record.protocol for record in records}),
    }
    logging.getLogger("neurodxfm").info("%s", json.dumps(summary, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    arguments = parser().parse_args(argv)
    if arguments.command == "train":
        return train_command(arguments)
    if arguments.command == "evaluate":
        return evaluate_command(arguments)
    if arguments.command == "validate-manifest":
        return validate_manifest_command(arguments)
    raise ValueError(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())

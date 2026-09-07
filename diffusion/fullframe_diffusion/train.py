from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import MASAM2_DIR, OSISAF_DIR
from dataset import create_dataloaders

from fullframe_diffusion.config import FullFrameDiffusionConfig
from fullframe_diffusion.ema import ExponentialMovingAverage
from fullframe_diffusion.runtime import create_model


GIB = 1024**3
TRACKED_METRICS = (
    "loss",
    "velocity_loss",
    "x0_loss",
)

# These settings control only the duration, location, or reporting cadence of a
# resumed run. They may change without altering the model, optimizer trajectory,
# data split, or diffusion objective stored in the checkpoint.
RESUME_MUTABLE_TRAINING_FIELDS = frozenset(
    {
        "epochs",
        "milestone_epochs",
        "output_dir",
        "num_workers",
        "validation_batches",
        "sample_every_epochs",
        "checkpoint_every_epochs",
        "checkpoint_every_steps",
        "early_stopping_patience",
    }
)


def validate_resume_configuration(
    checkpoint_values: dict,
    requested_config: FullFrameDiffusionConfig,
) -> None:
    """Reject resume settings that would change the experiment itself."""
    if not isinstance(checkpoint_values, dict):
        raise ValueError("Resume checkpoint does not contain a valid configuration")

    try:
        checkpoint_config = FullFrameDiffusionConfig.from_dict(checkpoint_values)
    except (TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint configuration is invalid") from error

    checkpoint_dict = checkpoint_config.to_dict()
    requested_dict = requested_config.to_dict()
    differences: list[str] = []

    for section in ("geometry", "network", "diffusion"):
        for key, checkpoint_value in checkpoint_dict[section].items():
            requested_value = requested_dict[section].get(key)
            if requested_value != checkpoint_value:
                differences.append(
                    f"{section}.{key}: checkpoint={checkpoint_value!r}, "
                    f"requested={requested_value!r}"
                )

    for key, checkpoint_value in checkpoint_dict["training"].items():
        if key in RESUME_MUTABLE_TRAINING_FIELDS:
            continue
        requested_value = requested_dict["training"].get(key)
        if requested_value != checkpoint_value:
            differences.append(
                f"training.{key}: checkpoint={checkpoint_value!r}, "
                f"requested={requested_value!r}"
            )

    if differences:
        details = "\n  - ".join(differences)
        raise ValueError(
            "Resume configuration changes model or training-critical settings:\n"
            f"  - {details}"
        )


def append_epoch_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_early_stopping_state(
    validation_loss: float,
    best_validation_loss: float,
    counter: int,
) -> tuple[float, int, bool]:
    """Update strict min-loss early stopping state, matching the U-Net runs."""
    if validation_loss < best_validation_loss:
        return validation_loss, 0, True
    return best_validation_loss, counter + 1, False


def save_checkpoint(
    path: Path,
    model,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: FullFrameDiffusionConfig,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    early_stopping_counter: int = 0,
    epoch_complete: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "early_stopping_counter": early_stopping_counter,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


@torch.no_grad()
def validate(
    model,
    dataloader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in TRACKED_METRICS}
    count = 0
    amp_enabled = device.type == "cuda"
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        osisaf = batch["lr"].to(device, non_blocking=True)
        masam2 = batch["hr"].to(device, non_blocking=True)
        with autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            losses = model.training_loss(osisaf, masam2)
        for key in totals:
            totals[key] += float(losses[key].item())
        count += 1
    model.train()
    if count == 0:
        raise ValueError("Validation dataloader produced no batches")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def save_validation_sample(
    model,
    dataloader,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    batch = next(iter(dataloader))
    osisaf = batch["lr"][:1].to(device)
    prediction = model.sample(osisaf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction[0, 0].float().cpu().numpy())
    model.train()


def train(
    config: FullFrameDiffusionConfig,
    resume: Path | None = None,
    osisaf_dir: str | Path = OSISAF_DIR,
    masam2_dir: str | Path = MASAM2_DIR,
    missed_dates_path: str | Path | None = None,
) -> None:
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Full-frame diffusion training requires a CUDA GPU")
    device = torch.device("cuda")
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _, _ = create_dataloaders(
        osisaf_dir=osisaf_dir,
        masam2_dir=masam2_dir,
        train_date_range=config.training.train_date_range,
        val_date_range=config.training.val_date_range,
        test_date_range=config.training.test_date_range,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        with_missed=config.training.with_missed_dates,
        missed_dates_path=missed_dates_path,
    )
    model = create_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=config.training.use_amp)
    ema = ExponentialMovingAverage(model, config.training.ema_decay)

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    early_stopping_counter = 0
    if resume is not None:
        checkpoint = torch.load(
            resume,
            map_location="cpu",
            weights_only=False,
        )
        validate_resume_configuration(checkpoint.get("config"), config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        checkpoint_epoch = int(checkpoint["epoch"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = checkpoint_epoch + 1 if epoch_complete else checkpoint_epoch
        if not epoch_complete:
            print(
                f"Checkpoint was saved during epoch {checkpoint_epoch + 1}; "
                "that epoch will restart from its beginning."
            )
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))
        if early_stopping_counter < 0:
            raise ValueError("Checkpoint early_stopping_counter must be non-negative")
        if start_epoch >= config.training.epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch}; "
                f"--epochs must be greater than {start_epoch} to continue training"
            )

    output_dir = Path(config.training.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    epoch_log_path = output_dir / "training_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.training.epochs):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started_at = time.perf_counter()
        model.train()
        running = {key: 0.0 for key in TRACKED_METRICS}
        batch_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.training.epochs}",
        )
        for batch_index, batch in enumerate(progress):
            osisaf = batch["lr"].to(device, non_blocking=True)
            masam2 = batch["hr"].to(device, non_blocking=True)
            with autocast(
                "cuda",
                enabled=config.training.use_amp,
                dtype=torch.float16,
            ):
                losses = model.training_loss(osisaf, masam2)
                scaled_loss = losses["loss"] / config.training.gradient_accumulation
            scaler.scale(scaled_loss).backward()

            should_step = (
                (batch_index + 1) % config.training.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
                if global_step % config.training.checkpoint_every_steps == 0:
                    save_checkpoint(
                        checkpoint_dir / "progress.pth",
                        model,
                        ema,
                        optimizer,
                        scaler,
                        config,
                        epoch,
                        global_step,
                        best_validation_loss,
                        early_stopping_counter,
                        epoch_complete=False,
                    )

            for key in running:
                running[key] += float(losses[key].item())
            batch_count += 1
            progress.set_postfix(loss=f"{running['loss'] / (batch_index + 1):.5f}")

        validation = validate(
            model,
            val_loader,
            device,
            config.training.validation_batches,
        )
        print(
            f"Epoch {epoch + 1}: val_loss={validation['loss']:.6f}, "
            f"velocity={validation['velocity_loss']:.6f}, "
            f"x0={validation['x0_loss']:.6f}"
        )

        best_validation_loss, early_stopping_counter, validation_improved = (
            update_early_stopping_state(
                validation["loss"],
                best_validation_loss,
                early_stopping_counter,
            )
        )
        if validation_improved:
            save_checkpoint(
                checkpoint_dir / "best.pth",
                model,
                ema,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_validation_loss,
                early_stopping_counter,
            )
        else:
            print(
                "Early stopping: "
                f"{early_stopping_counter}/{config.training.early_stopping_patience} "
                "epochs without validation improvement"
            )

        should_early_stop = (
            early_stopping_counter >= config.training.early_stopping_patience
        )

        if (
            (epoch + 1) % config.training.checkpoint_every_epochs == 0
            or should_early_stop
        ):
            save_checkpoint(
                checkpoint_dir / "last.pth",
                model,
                ema,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_validation_loss,
                early_stopping_counter,
            )
        if epoch + 1 in config.training.milestone_epochs:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                model,
                ema,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_validation_loss,
                early_stopping_counter,
            )
        if (epoch + 1) % config.training.sample_every_epochs == 0:
            save_validation_sample(
                model,
                val_loader,
                device,
                sample_dir / f"epoch_{epoch + 1:04d}.npy",
            )
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started_at
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / GIB
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB
        train_average = {
            key: value / max(batch_count, 1) for key, value in running.items()
        }
        epoch_record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "epoch_seconds": epoch_seconds,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "train": train_average,
            "validation": validation,
            "validation_improved": validation_improved,
            "early_stopping_counter": early_stopping_counter,
            "early_stopping_patience": config.training.early_stopping_patience,
        }
        append_epoch_log(epoch_log_path, epoch_record)
        print(
            f"Epoch {epoch + 1} resources: time={epoch_seconds:.1f}s "
            f"({epoch_seconds / 60.0:.1f} min), "
            f"peak_allocated={peak_allocated_gib:.2f} GiB, "
            f"peak_reserved={peak_reserved_gib:.2f} GiB"
        )
        if should_early_stop:
            print("=" * 70)
            print("EARLY STOP")
            print(
                "Validation loss did not improve for "
                f"{config.training.early_stopping_patience} consecutive epochs"
            )
            print(f"Best validation loss: {best_validation_loss:.6f}")
            print(f"Stopped after epoch: {epoch + 1}")
            print("=" * 70)
            break


def main() -> None:
    defaults = FullFrameDiffusionConfig()
    parser = argparse.ArgumentParser(
        description="Train SR3-conditioned direct full-frame v-diffusion."
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--base-channels",
        type=int,
        default=defaults.network.base_channels,
    )
    parser.add_argument(
        "--shuffle-factor",
        type=int,
        choices=(1, 2, 4),
        default=defaults.geometry.shuffle_factor,
    )
    parser.add_argument("--epochs", type=int, default=defaults.training.epochs)
    parser.add_argument("--osisaf-dir", type=Path, default=OSISAF_DIR)
    parser.add_argument("--masam2-dir", type=Path, default=MASAM2_DIR)
    parser.add_argument("--missed-dates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-range", nargs=2, metavar=("START", "END"))
    parser.add_argument("--val-range", nargs=2, metavar=("START", "END"))
    parser.add_argument("--test-range", nargs=2, metavar=("START", "END"))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.training.num_workers,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=defaults.training.gradient_accumulation,
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=defaults.training.validation_batches,
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=defaults.training.checkpoint_every_steps,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.training.early_stopping_patience,
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=defaults.diffusion.sampling_steps,
    )
    parser.add_argument("--seed", type=int, default=defaults.training.seed)
    parser.add_argument("--disable-attention", action="store_true")
    args = parser.parse_args()

    config = defaults
    padded_shape = config.geometry.padded_shape
    config.geometry = replace(
        config.geometry,
        shuffle_factor=args.shuffle_factor,
    )
    if any(size % args.shuffle_factor for size in padded_shape):
        raise ValueError("Configured padded shape is incompatible with shuffle factor")
    config.network = replace(
        config.network,
        base_channels=args.base_channels,
        use_pooled_attention=not args.disable_attention,
    )
    config.diffusion = replace(
        config.diffusion,
        sampling_steps=args.sampling_steps,
    )
    config.training = replace(
        config.training,
        train_date_range=tuple(args.train_range or config.training.train_date_range),
        val_date_range=tuple(args.val_range or config.training.val_date_range),
        test_date_range=tuple(args.test_range or config.training.test_date_range),
        epochs=args.epochs,
        milestone_epochs=tuple(
            epoch for epoch in config.training.milestone_epochs if epoch <= args.epochs
        ),
        num_workers=args.num_workers,
        gradient_accumulation=args.gradient_accumulation,
        validation_batches=args.validation_batches,
        checkpoint_every_steps=args.checkpoint_every_steps,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        output_dir=str(args.output_dir or config.training.output_dir),
        with_missed_dates=args.missed_dates is None,
    )
    train(
        config,
        resume=args.resume,
        osisaf_dir=args.osisaf_dir,
        masam2_dir=args.masam2_dir,
        missed_dates_path=args.missed_dates,
    )


if __name__ == "__main__":
    main()

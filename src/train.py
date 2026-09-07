"""Train UNetLight / AttentionUNet.

Configurations differ in the architecture and in where the OSISAF -> MASAM2 map
projection happens; `sweep_*` vary the channel width. See TASKS below.

Reads the integer cache written by prepare_data.py.
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AttentionUNet, UNetLight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
from paths import CACHE

TRAIN_RANGE = ("20120701", "20201231")
VAL_RANGE = ("20210101", "20221231")
TEST_RANGE = ("20230101", "20260607")

TASKS = {
    # name: (input cache directory, model class, model kwargs, batch size)
    #
    # e2e        native OSISAF; the projection layer is applied to the INPUT, so
    #            the whole U-Net runs at 2100x2550. Structurally identical to
    #            `enhance` and differing only in who reprojected.
    # enhance    input already reprojected offline by `cdo remapnn`; no
    #            projection layer, the network only enhances.
    # e2e_attn   e2e with the attention architecture; paired with e2e it
    #            isolates the attention gates.
    # e2e_head   projection applied in the HEAD instead, so the U-Net runs on the
    #            native 432x432 grid. Cheaper and with a wider receptive field.
    "e2e":       (CACHE / "native", UNetLight,
                  dict(reproject_first=True, use_checkpoint=True), 1),
    "e2e_attn":  (CACHE / "native", AttentionUNet,
                  dict(reproject_first=True, use_checkpoint=True), 1),
    "e2e_head":  (CACHE / "native", UNetLight, dict(reproject=True), 2),
    "enhance":   (CACHE / "reproj", UNetLight,
                  dict(out_size=(2100, 2550), use_checkpoint=True), 1),

    # Capacity sweep: the head configuration with everything fixed except the
    # channel width.
    "sweep_w05": (CACHE / "native", UNetLight, dict(reproject=True, width=0.5), 2),
    "sweep_w10": (CACHE / "native", UNetLight, dict(reproject=True, width=1.0), 2),
    "sweep_w20": (CACHE / "native", UNetLight, dict(reproject=True, width=2.0), 2),
    "sweep_w25": (CACHE / "native", UNetLight,
                  dict(reproject=True, width=2.5, use_checkpoint=True), 2),
}



def missed_dates():
    import ast
    entries = ast.literal_eval((PROJECT_ROOT / "data" / "masam2_missed.txt").read_text(encoding="utf-8"))
    return {re.search(r"(\d{8})", str(e)).group(1) for e in entries}


def index_dir(d):
    out = {}
    for p in Path(d).glob("*.npy"):
        m = re.search(r"(\d{8})", p.name)
        if m:
            out[m.group(1)] = str(p)
    return out


class U8Pairs(Dataset):
    """Integer cache on disk -> float32 [0,1] tensors"""

    SCALE = {np.dtype(np.uint8): 100.0, np.dtype(np.uint16): 10000.0}

    def __init__(self, days, src_index, tgt_index):
        self.days = days
        self.src = src_index
        self.tgt = tgt_index

    def __len__(self):
        return len(self.days)

    def _load(self, path):
        a = np.load(path)
        scale = self.SCALE.get(a.dtype)
        if scale is None:
            raise ValueError(f"unexpected cache dtype {a.dtype} in {path}")
        return torch.from_numpy(a.astype(np.float32) / scale)[None]

    def __getitem__(self, i):
        day = self.days[i]
        return self._load(self.src[day]), self._load(self.tgt[day]), day


def build_splits(src_index, tgt_index):
    bad = missed_dates()
    common = sorted((set(src_index) & set(tgt_index)) - bad)
    if not common:
        raise SystemExit(
            "no overlapping days - has cache finished for both sources?"
        )
    def rng(a, b):
        return [d for d in common if a <= d <= b]
    return rng(*TRAIN_RANGE), rng(*VAL_RANGE), rng(*TEST_RANGE)


AMP_DTYPE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class NonFiniteLoss(RuntimeError):
    """Training produced NaN/Inf; weights past this point are worthless."""


def run_epoch(model, loader, crit, dev, opt=None, amp=None, log_every=200, tag="",
              save_every=0, save_fn=None, scaler=None, channels_last=False,
              accum_steps=1):
    """save_every>0 snapshots weights+optimizer mid-epoch. Resuming re-runs the epoch from its start but keeps the
    weights, which is the part that costs time to recreate."""
    train = opt is not None
    model.train(train)
    if train:
        opt.zero_grad(set_to_none=True)
    total, n = 0.0, 0
    t0 = time.time()
    for i, (x, y, _) in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", amp or torch.float16, enabled=amp is not None):
                pred = model(x)
                loss = crit(pred.float(), y)
            if train:
                scaled = loss / accum_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
        lv = loss.item()
        if not math.isfinite(lv):
            raise NonFiniteLoss(
                f"non-finite loss ({lv}) at step {i+1} of {tag or 'epoch'}; "
                f"weights are unrecoverable from here - resume from _best.pth"
            )
        total += lv * x.shape[0]
        n += x.shape[0]
        if train and log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    {tag} {i+1}/{len(loader)}  loss {total/n:.5f}  "
                  f"{el:.0f}s  eta {el/(i+1)*(len(loader)-i-1):.0f}s", flush=True)
        if train and save_every and save_fn and (i + 1) % save_every == 0:
            save_fn()
    return total / max(n, 1), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20, help="early stop patience, epochs")
    ap.add_argument("--sched-patience", type=int, default=8)
    ap.add_argument("--warm-restart-lr", type=float, default=None,
                    help="on resume, force this LR and restart the scheduler; "
                         "for a model that never reached the annealing phase")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    ap.add_argument("--channels-last", action="store_true",
                    help="NHWC layout; stays fp32, ~2x faster here, no precision change")
    ap.add_argument("--accum-steps", type=int, default=2,
                    help="gradient accumulation; effective batch = batch_size * accum_steps")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth day of each split (3 = a third of the data)")
    ap.add_argument("--save-every", type=int, default=400,
                    help="mid-epoch snapshot interval in steps; 0 disables")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.80,
                    help="cap on VRAM, leaving headroom for the desktop compositor")
    ap.add_argument("--nice", action="store_true", default=True,
                    help="run below normal priority so the UI stays responsive")
    ap.add_argument("--no-nice", dest="nice", action="store_false")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: cap train days")
    args = ap.parse_args()

    dev = "cuda"
    if args.nice:
        try:                                   # BELOW_NORMAL_PRIORITY_CLASS
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
            print("  process priority: below normal")
        except Exception as e:
            print(f"  could not lower priority: {e}")
    if args.gpu_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM cap: {args.gpu_mem_fraction:.0%} of {total:.1f} GB "
              f"= {total*args.gpu_mem_fraction:.1f} GB")

    src_dir, model_cls, kwargs, default_bs = TASKS[args.task]
    bs = args.batch_size or default_bs

    src_index = index_dir(src_dir)
    tgt_index = index_dir(CACHE / "masam2")
    tr, va, te = build_splits(src_index, tgt_index)
    if args.stride > 1:
        tr, va, te = tr[::args.stride], va[::args.stride], te[::args.stride]
        print(f"  stride {args.stride}: using every {args.stride}rd day")
    if args.limit_train:
        tr = tr[: args.limit_train]
        va = va[: max(8, args.limit_train // 4)]
    print(f"task={args.task}  input={src_dir}")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}  batch {bs}"
          f"  accum {args.accum_steps} (effective batch {bs*args.accum_steps})")

    mk = lambda days, shuffle: DataLoader(
        U8Pairs(days, src_index, tgt_index), batch_size=bs, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    train_loader, val_loader = mk(tr, True), mk(va, False)

    model = model_cls(1, 1, **kwargs).to(dev)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    amp = AMP_DTYPE[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    print(f"  precision={args.precision}  channels_last={args.channels_last}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {model_cls.__name__} {n_par/1e6:.2f}M params, kwargs={kwargs}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.sched_patience
    )
    crit = nn.L1Loss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / f"unet_{args.task}_best.pth"
    hist_path = MODELS_DIR / f"unet_{args.task}_history.json"
    start, best, history, stale = 0, float("inf"), [], 0
    state_path = MODELS_DIR / f"unet_{args.task}_last.pth"
    resume_from = state_path if state_path.exists() else None
    if resume_from is not None:
        probe = torch.load(resume_from, map_location="cpu")
        if any(not torch.isfinite(v).all() for v in probe["model_state_dict"].values()):
            print(f"  {state_path.name} contains non-finite weights - falling back to {ckpt.name}")
            resume_from = ckpt if ckpt.exists() else None
        del probe
    if resume_from is not None:
        st = torch.load(resume_from, map_location=dev)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        sched.load_state_dict(st["scheduler_state_dict"])
        start = st["epoch"] + 1
        best = st.get("best_val", st.get("val_loss", float("inf")))
        stale = st.get("stale", 0)
        history = [h for h in st.get("history", []) if h["epoch"] <= start]
        if args.warm_restart_lr is not None:
            for group in opt.param_groups:
                group["lr"] = args.warm_restart_lr
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=args.sched_patience
            )
            stale = 0
            print(f"  WARM RESTART: lr set to {args.warm_restart_lr:.2e}, fresh "
                  f"scheduler, stale counter cleared (best val kept at {best:.6f})")
        print(f"  resumed from {resume_from.name} at epoch {start}, "
              f"best val {best:.6f}, stale {stale}")

    def snapshot(epoch, best_val, stale, note=""):
        tmp = state_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "best_val": best_val, "stale": stale, "history": history,
                    "task": args.task, "kwargs": kwargs, "note": note}, tmp)
        os.replace(tmp, state_path)

    for ep in range(start, args.epochs):
        mid = lambda: snapshot(ep - 1, best, stale, note=f"mid-epoch {ep+1}")
        trl, tt = run_epoch(model, train_loader, crit, dev, opt, amp, tag=f"e{ep+1}",
                            save_every=args.save_every, save_fn=mid, scaler=scaler,
                            channels_last=args.channels_last, accum_steps=args.accum_steps)
        vl, vt = run_epoch(model, val_loader, crit, dev, None, amp,
                           channels_last=args.channels_last)
        sched.step(vl)
        history.append({"epoch": ep + 1, "train": trl, "val": vl,
                        "lr": opt.param_groups[0]["lr"], "sec": tt + vt})

        flag = ""
        if vl < best:
            best, stale, flag = vl, 0, "  <- best"
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "val_loss": vl, "train_loss": trl, "history": history,
                        "task": args.task, "kwargs": kwargs}, ckpt)
        else:
            stale += 1
            flag = f"  (stale {stale}/{args.patience})"
        snapshot(ep, best, stale)

        print(f"epoch {ep+1}/{args.epochs}  train {trl:.6f}  val {vl:.6f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({tt:.0f}s + {vt:.0f}s){flag}", flush=True)
        hist_path.write_text(json.dumps(history, indent=1))

        if stale >= args.patience:
            print(f"\nearly stop: val did not improve for {args.patience} epochs")
            break

    print(f"\ndone. best val {best:.6f}  ->  {ckpt}")


if __name__ == "__main__":
    main()

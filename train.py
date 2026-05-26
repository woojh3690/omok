import random
import time
from collections import deque
from pathlib import Path
from typing import Optional
import re

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader, TensorDataset

from omok.mcts import MCTS
from omok.model import ModelWrapper
from omok.selfplay import play_self_play_game
from omok.storage import ModelRegistry

app = typer.Typer(add_completion=False)


def make_grad_scaler():
    if not torch.cuda.is_available():
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _symmetries(
    planes: np.ndarray, policy: np.ndarray, board_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    policy_board = policy.reshape(board_size, board_size)
    augmented = []
    for k in range(4):
        rotated_planes = np.rot90(planes, k, axes=(1, 2))
        rotated_policy = np.rot90(policy_board, k)
        augmented.append(
            (
                np.ascontiguousarray(rotated_planes),
                np.ascontiguousarray(rotated_policy.reshape(-1)),
            )
        )
        augmented.append(
            (
                np.ascontiguousarray(np.flip(rotated_planes, axis=2)),
                np.ascontiguousarray(np.flip(rotated_policy, axis=1).reshape(-1)),
            )
        )
    return augmented


def tensorize(samples, board_size: int, augment: bool = True):
    planes_list = []
    policy_list = []
    value_list = []
    for planes, policy, value in samples:
        if augment:
            views = _symmetries(planes, policy, board_size)
        else:
            views = [(np.ascontiguousarray(planes), np.ascontiguousarray(policy))]
        for aug_planes, aug_policy in views:
            planes_list.append(aug_planes)
            policy_list.append(aug_policy)
            value_list.append(value)

    planes = torch.from_numpy(np.stack(planes_list).astype(np.float32))
    target_p = torch.from_numpy(np.stack(policy_list).astype(np.float32))
    target_v = torch.from_numpy(np.asarray(value_list, dtype=np.float32))
    return planes, target_p, target_v


def _infer_last_epoch(best_dir: Path, registry: ModelRegistry) -> int:
    candidates: list[int] = []
    if best_dir.exists():
        for p in best_dir.glob("model_*.pt"):
            m = re.search(r"model_(\d+)\.pt$", p.name)
            if m:
                candidates.append(int(m.group(1)))
    for info in registry.list_models():
        m = re.search(r"(\d+)$", info.id)
        if m:
            candidates.append(int(m.group(1)))
    return max(candidates, default=0)


def _find_latest_weights(latest_dir: Path, best_dir: Path, registry: ModelRegistry) -> Optional[Path]:
    candidates: list[Path] = []
    latest_pt = latest_dir / "latest.pt"
    if latest_pt.exists():
        candidates.append(latest_pt)
    if best_dir.exists():
        candidates.extend([p for p in best_dir.glob("*.pt") if p.is_file()])
    for info in registry.list_models():
        p = Path(info.path)
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_training_checkpoint(
    ckpt_path: Path,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> dict:
    state = torch.load(ckpt_path, map_location=model.device)
    model.net.load_state_dict(state["model_state"])
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if "scaler_state" in state and scaler is not None:
        try:
            scaler.load_state_dict(state["scaler_state"])
        except Exception:
            pass
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    return state


def _save_training_checkpoint(
    ckpt_path: Path,
    *,
    epoch: int,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    best_loss: Optional[float],
    loss_ema: Optional[float],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "board_size": model.board_size,
            "model_state": model.net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "loss_ema": loss_ema,
        },
        ckpt_path,
    )


@app.command()
def main(
    epochs: int = typer.Option(1000, help="Number of training epochs."),
    games_per_epoch: int = typer.Option(10, help="Self-play games per epoch."),
    visits: int = typer.Option(200, help="MCTS simulations per move."),
    batch_size: int = typer.Option(64, help="Training batch size."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    min_lr: float = typer.Option(1e-5, help="Minimum learning rate for plateau scheduling."),
    save_every: int = typer.Option(1, help="Save checkpoint every N epochs."),
    min_improvement: float = typer.Option(0.01, help="Loss delta to promote a model as best."),
    board_size: int = typer.Option(15, help="Board size (default 15)."),
    replay_buffer_size: int = typer.Option(20000, help="Maximum self-play samples kept for replay."),
    train_passes: int = typer.Option(2, help="Training passes over the replay buffer per epoch."),
    max_grad_norm: float = typer.Option(5.0, help="Gradient clipping norm. Set <=0 to disable."),
    loss_ema_alpha: float = typer.Option(0.3, help="EMA smoothing factor for best-model promotion."),
    augment: bool = typer.Option(True, help="Use 8-way board symmetry augmentation."),
    seed: Optional[int] = typer.Option(None, help="Optional random seed for reproducible runs."),
    checkpoint_root: Path = typer.Option(Path("checkpoints"), help="Checkpoint root directory."),
    resume: bool = typer.Option(True, help="Resume from latest saved checkpoint if available."),
) -> None:
    set_seed(seed)
    registry = ModelRegistry(root=checkpoint_root)
    model = ModelWrapper(board_size=board_size)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = make_grad_scaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=min_lr
    )
    best_loss: Optional[float] = None
    loss_ema: Optional[float] = None
    replay_buffer = deque(maxlen=max(1, replay_buffer_size))
    save_every = max(1, save_every)
    train_passes = max(1, train_passes)
    loss_ema_alpha = min(1.0, max(0.0, loss_ema_alpha))

    latest_dir = checkpoint_root / "latest"
    best_dir = checkpoint_root / "best"
    latest_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    ckpt_path = latest_dir / "latest.ckpt"
    if resume:
        if ckpt_path.exists():
            ckpt = _load_training_checkpoint(ckpt_path, model, optimizer, scaler, scheduler)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = ckpt.get("best_loss")
            loss_ema = ckpt.get("loss_ema")
            print(f"Resumed training from {ckpt_path} (epoch={start_epoch - 1}).")
        else:
            weights = _find_latest_weights(latest_dir, best_dir, registry)
            if weights is not None:
                model.load(weights)
                start_epoch = _infer_last_epoch(best_dir, registry) + 1
                losses = [m.loss for m in registry.list_models()]
                best_loss = min(losses) if losses else None
                print(f"Loaded latest weights from {weights} (starting epoch={start_epoch}).")

    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch ({start_epoch}) > epochs ({epochs}).")
        return

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        mcts = MCTS(model=model, board_size=board_size)
        new_samples = []

        for g in range(games_per_epoch):
            new_samples.extend(play_self_play_game(mcts, simulations=visits))

        replay_buffer.extend(new_samples)
        planes, target_p, target_v = tensorize(list(replay_buffer), board_size=board_size, augment=augment)
        dataset = TensorDataset(planes, target_p, target_v)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        policy_losses = []
        value_losses = []
        for _ in range(max(1, train_passes)):
            for batch in loader:
                p_loss, v_loss = model.train_step(
                    batch, optimizer, scaler=scaler, max_grad_norm=max_grad_norm
                )
                policy_losses.append(p_loss)
                value_losses.append(v_loss)

        epoch_p_loss = sum(policy_losses) / len(policy_losses)
        epoch_v_loss = sum(value_losses) / len(value_losses)
        epoch_loss = epoch_p_loss + epoch_v_loss
        loss_ema = epoch_loss if loss_ema is None else (
            loss_ema_alpha * epoch_loss + (1.0 - loss_ema_alpha) * loss_ema
        )
        scheduler.step(loss_ema)
        duration = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} loss={epoch_loss:.4f} ema={loss_ema:.4f} "
            f"(p={epoch_p_loss:.4f}, v={epoch_v_loss:.4f}) "
            f"samples={len(new_samples)} replay={len(replay_buffer)} lr={current_lr:.2e} time={duration:.1f}s"
        )

        metric_loss = loss_ema
        improved = best_loss is None or metric_loss < (best_loss - min_improvement)
        if improved:
            best_loss = metric_loss
            best_path = best_dir / f"model_{epoch:03d}.pt"
            model.save(best_path)
            info = registry.register(best_path, loss=epoch_loss, tag="best")
            print(f"Promoted new best model: {info.id} loss={epoch_loss:.4f} ema={metric_loss:.4f}")

        if epoch % save_every == 0:
            latest_path = latest_dir / "latest.pt"
            model.save(latest_path)
            _save_training_checkpoint(
                ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_loss=best_loss,
                loss_ema=loss_ema,
            )

    print("Training finished.")


if __name__ == "__main__":
    app()

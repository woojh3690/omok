from __future__ import annotations

import shutil
import time
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


def tensorize(samples):
    planes = torch.from_numpy(np.stack([s[0] for s in samples]).astype(np.float32))
    target_p = torch.from_numpy(np.stack([s[1] for s in samples]).astype(np.float32))
    target_v = torch.from_numpy(np.asarray([s[2] for s in samples], dtype=np.float32))
    return planes, target_p, target_v


def _infer_last_epoch(best_dir: Path, registry: ModelRegistry) -> int:
    candidates: list[int] = []
    if best_dir.exists():
        for p in best_dir.glob("model_*.pt"):
            m = re.search(r"model_(\\d+)\\.pt$", p.name)
            if m:
                candidates.append(int(m.group(1)))
    for info in registry.list_models():
        m = re.search(r"(\\d+)$", info.id)
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
    scaler: torch.amp.GradScaler,
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
    return state


def _save_training_checkpoint(
    ckpt_path: Path,
    *,
    epoch: int,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best_loss: Optional[float],
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
    save_every: int = typer.Option(1, help="Save checkpoint every N epochs."),
    min_improvement: float = typer.Option(0.01, help="Loss delta to promote a model as best."),
    board_size: int = typer.Option(15, help="Board size (default 15)."),
    resume: bool = typer.Option(True, help="Resume from latest saved checkpoint if available."),
) -> None:
    registry = ModelRegistry()
    model = ModelWrapper(board_size=board_size)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_loss: Optional[float] = None

    latest_dir = Path("checkpoints/latest")
    best_dir = Path("checkpoints/best")
    latest_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    ckpt_path = latest_dir / "latest.ckpt"
    if resume:
        if ckpt_path.exists():
            ckpt = _load_training_checkpoint(ckpt_path, model, optimizer, scaler)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = ckpt.get("best_loss")
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
        buffer = []

        for g in range(games_per_epoch):
            buffer.extend(play_self_play_game(mcts, simulations=visits))

        planes, target_p, target_v = tensorize(buffer)
        dataset = TensorDataset(planes, target_p, target_v)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        policy_losses = []
        value_losses = []
        for batch in loader:
            p_loss, v_loss = model.train_step(batch, optimizer, scaler=scaler)
            policy_losses.append(p_loss)
            value_losses.append(v_loss)

        epoch_p_loss = sum(policy_losses) / len(policy_losses)
        epoch_v_loss = sum(value_losses) / len(value_losses)
        epoch_loss = epoch_p_loss + epoch_v_loss
        duration = time.time() - start

        print(
            f"Epoch {epoch}/{epochs} loss={epoch_loss:.4f} "
            f"(p={epoch_p_loss:.4f}, v={epoch_v_loss:.4f}) games={len(buffer)} time={duration:.1f}s"
        )

        if epoch % save_every == 0:
            latest_path = latest_dir / "latest.pt"
            model.save(latest_path)
            _save_training_checkpoint(
                ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_loss=best_loss,
            )

        improved = best_loss is None or epoch_loss < (best_loss - min_improvement)
        if improved:
            best_loss = epoch_loss
            best_path = best_dir / f"model_{epoch:03d}.pt"
            shutil.copy(latest_dir / "latest.pt", best_path) if (latest_dir / "latest.pt").exists() else model.save(
                best_path
            )
            info = registry.register(best_path, loss=epoch_loss, tag="best")
            print(f"Promoted new best model: {info.id} loss={epoch_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    app()

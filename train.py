from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import typer
from torch.utils.data import DataLoader, TensorDataset

from omok.mcts import MCTS
from omok.model import ModelWrapper
from omok.selfplay import play_self_play_game
from omok.storage import ModelRegistry

app = typer.Typer(add_completion=False)


def tensorize(samples):
    planes = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    target_p = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    target_v = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    return planes, target_p, target_v


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

    for epoch in range(1, epochs + 1):
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

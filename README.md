# Omok RL (Gomoku, Long Pro)

Reinforcement-learning playground for Gomoku (15x15, Long Pro rule). The project trains an AlphaZero-style policy/value network with self-play MCTS and serves a web UI to play against saved checkpoints.

## Getting started

```bash
uv venv
uv sync
```

This project uses `pyproject.toml` and points `torch` at the official CUDA 12.8 PyTorch wheel index on Windows/Linux. This is intended for modern NVIDIA GPUs such as RTX 50-series cards.

Verify that PyTorch can see the GPU:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Training

Self-play + training loop (tweak hyper-parameters inside `train.py`):

```bash
uv run python train.py --epochs 50 --games-per-epoch 20 --visits 200
```

Checkpoints live in `checkpoints/latest` and promoted best models are copied to `checkpoints/best`. A lightweight `model_index.json` keeps metadata for the web UI.

For a CUDA machine, keep training on GPU and run parallel self-play inference on CPU. This avoids creating many CUDA contexts from spawned self-play processes:

```bash
uv run python train.py --epochs 150 --games-per-epoch 20 --visits 160 --mcts-batch-size 0 --batch-size 1024 --data-workers 2 --self-play-workers 0 --self-play-device auto --self-play-torch-threads 1
```

Use `--mcts-batch-size 0` to choose a default automatically: 64 when self-play uses CUDA, 16 when it uses CPU.
Use `--self-play-workers 0` to run self-play in parallel with all logical CPU cores. If you explicitly set `--self-play-device cuda`, keep `--self-play-workers` low to avoid CUDA compiler and context memory pressure.

Quick tactical verification for a trained checkpoint:

```bash
uv run python -m omok.evaluation --weights checkpoints/latest/latest.pt --mcts-batch-size 64
```

Assumptions about Long Pro: black must win with exactly five; making an overline (6+) is a foul that awards the win to white. White wins on five or more in a row.

## Web play

Serve the API + static files:

```bash
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and play in the browser. Use the model selector to switch among saved checkpoints; default is the latest model.

## Project layout

- `omok/game.py` – board state, Long Pro rule enforcement, winner detection.
- `omok/mcts.py` – Monte Carlo Tree Search with policy/value guidance.
- `omok/model.py` – Torch policy-value network.
- `omok/selfplay.py` – self-play episode generator.
- `omok/storage.py` – checkpoint loading, saving, and metadata for the UI.
- `train.py` – high-level training loop.
- `server.py` – FastAPI app exposing play + model selection; serves `web/`.
- `web/` – static frontend (vanilla JS) for in-browser play.

## Notes

- Board size is fixed at 15x15.
- The training code is intentionally lightweight; adjust network depth, MCTS visits, and replay buffer size for serious runs on powerful GPUs.

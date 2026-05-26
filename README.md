# Omok RL (Gomoku, Long Pro)

Reinforcement-learning playground for Gomoku (15x15, Long Pro rule). The project trains an AlphaZero-style policy/value network with self-play MCTS and serves a web UI to play against saved checkpoints.

## Getting started

```bash
uv venv
uv pip install -r requirements.txt
```

## Training

Self-play + training loop (tweak hyper-parameters inside `train.py`):

```bash
uv run python train.py --epochs 50 --games-per-epoch 20 --visits 200
```

Checkpoints live in `checkpoints/latest` and promoted best models are copied to `checkpoints/best`. A lightweight `model_index.json` keeps metadata for the web UI.

For a CUDA machine, keep MCTS on CPU but batch the neural-network leaf evaluations on GPU:

```bash
uv run python train.py --epochs 150 --games-per-epoch 20 --visits 160 --mcts-batch-size 64 --batch-size 256 --data-workers 2
```

Use `--mcts-batch-size 0` to choose a default automatically: 64 on CUDA, 16 on CPU.

Quick tactical verification for a trained checkpoint:

```bash
uv run python -m omok.evaluation --weights checkpoints/latest/latest.pt --mcts-batch-size 64
```

Assumptions about Long Pro: black must win with exactly five; making an overline (6+) is a foul that awards the win to white. White wins on five or more in a row.

## Web play

Serve the API + static files:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
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

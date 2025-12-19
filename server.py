from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omok.game import GomokuBoard
from omok.mcts import MCTS
from omok.model import ModelWrapper
from omok.storage import ModelRegistry

app = FastAPI(title="Omok RL", default_response_class=JSONResponse)

STATIC_DIR = Path("web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GameSession:
    def __init__(self, board_size: int = 15):
        self.board = GomokuBoard(size=board_size)

    def reset(self):
        self.board = GomokuBoard(size=self.board.size)


class ModelManager:
    def __init__(self, board_size: int = 15):
        self.registry = ModelRegistry()
        self.model = ModelWrapper(board_size=board_size)
        self.board_size = board_size
        self.active_id: Optional[str] = None
        self.active_path: Optional[Path] = None
        self._load_latest_or_default()

    def _find_latest_weights(self) -> Optional[Path]:
        candidates: list[Path] = []
        latest_path = Path("checkpoints/latest/latest.pt")
        if latest_path.exists():
            candidates.append(latest_path)
        best_dir = Path("checkpoints/best")
        if best_dir.exists():
            candidates.extend([p for p in best_dir.glob("*.pt") if p.is_file()])
        for info in self.registry.list_models():
            p = Path(info.path)
            if p.exists():
                candidates.append(p)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _resolve_active_id(self, path: Path) -> str:
        if path.name == "latest.pt":
            return "latest"
        for info in self.registry.list_models():
            if Path(info.path) == path:
                return info.id
        return path.stem

    def _latest_entry(self) -> Optional[dict]:
        latest_path = Path("checkpoints/latest/latest.pt")
        if not latest_path.exists():
            return None
        created_at = datetime.fromtimestamp(
            latest_path.stat().st_mtime, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
        return {
            "id": "latest",
            "path": str(latest_path),
            "loss": None,
            "created_at": created_at,
            "tag": "latest",
        }

    def _load_latest_or_default(self):
        latest_path = self._find_latest_weights()
        if latest_path is None:
            print("Using freshly initialized model (no checkpoint found).")
            return

        self.model.load(latest_path)
        self.active_path = latest_path
        self.active_id = self._resolve_active_id(latest_path)
        if self.active_id in self.registry.models:
            self.registry.set_active(self.active_id)
        print(f"Loaded latest model {self.active_id}")

    def list_models(self):
        items = [m.__dict__ for m in self.registry.list_models()]
        latest = self._latest_entry()
        if latest is not None:
            items = [latest, *items]
        return items

    def set_active(self, model_id: str):
        info = self.registry.set_active(model_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        self.model.load(info.path)
        return info


session = GameSession()
models = ModelManager()


def board_payload(board: GomokuBoard):
    return {
        "board": board.board.tolist(),
        "current_player": board.current_player,
        "winner": board.winner,
        "foul": board.foul,
        "last_move": board.last_move,
    }


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/api/models")
async def list_models():
    return {"models": models.list_models(), "active_id": models.active_id}


@app.post("/api/models/select")
async def select_model(payload: dict):
    raise HTTPException(status_code=403, detail="Model selection is disabled")


@app.post("/api/game/new")
async def new_game():
    session.reset()
    return board_payload(session.board)


@app.post("/api/game/move")
async def play_move(payload: dict):
    if session.board.winner is not None:
        return board_payload(session.board)

    x = payload.get("x")
    y = payload.get("y")
    if x is None or y is None:
        raise HTTPException(status_code=400, detail="x and y are required")

    if not session.board.is_valid_move(x, y):
        raise HTTPException(status_code=400, detail="Invalid move")

    session.board.play_move(x, y)
    if session.board.winner is None:
        ai_action = choose_ai_action(session.board, models.model)
        if ai_action is not None:
            ax, ay = session.board.from_flat_index(ai_action)
            session.board.play_move(ax, ay)

    return board_payload(session.board)


def choose_ai_action(board: GomokuBoard, model: ModelWrapper) -> Optional[int]:
    if board.winner is not None:
        return None
    mcts = MCTS(model=model, board_size=board.size, c_puct=2.2)
    pi = mcts.run(board.clone(), num_simulations=200, add_noise=False)
    return int(np.argmax(pi))


@app.get("/api/state")
async def state():
    return board_payload(session.board)

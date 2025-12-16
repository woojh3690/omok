from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class ModelInfo:
    id: str
    path: str
    loss: float
    created_at: str
    tag: str = "best"


class ModelRegistry:
    def __init__(self, root: str | Path = "checkpoints"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "model_index.json"
        self.models: Dict[str, ModelInfo] = {}
        self.active_id: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        data = json.loads(self.index_path.read_text())
        self.active_id = data.get("active_id")
        for item in data.get("models", []):
            info = ModelInfo(**item)
            self.models[info.id] = info

    def _persist(self) -> None:
        payload = {
            "active_id": self.active_id,
            "models": [asdict(m) for m in self.models.values()],
        }
        self.index_path.write_text(json.dumps(payload, indent=2))

    def register(self, path: Path, loss: float, tag: str = "best") -> ModelInfo:
        model_id = path.stem
        info = ModelInfo(id=model_id, path=str(path), loss=loss, created_at=now_ts(), tag=tag)
        self.models[model_id] = info
        self.active_id = model_id  # default new best to active
        self._persist()
        return info

    def list_models(self) -> List[ModelInfo]:
        return sorted(self.models.values(), key=lambda m: m.created_at, reverse=True)

    def set_active(self, model_id: str) -> Optional[ModelInfo]:
        if model_id not in self.models:
            return None
        self.active_id = model_id
        self._persist()
        return self.models[model_id]

    def active_model(self) -> Optional[ModelInfo]:
        if self.active_id is None:
            return None
        return self.models.get(self.active_id)

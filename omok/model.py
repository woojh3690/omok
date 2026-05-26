from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


def cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 기본 ResNet 블록
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return F.relu(out)


class GomokuNet(nn.Module):
    def __init__(self, board_size: int = 15, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            conv3(2, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            conv3(channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_linear = nn.Linear(32 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_head = nn.Sequential(
            conv3(channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_linear = nn.Sequential(
            nn.Linear(32 * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_linear(p)

        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = self.value_linear(v)
        return p, v.squeeze(-1)


@dataclass
class ModelWrapper:
    board_size: int = 15
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_amp: bool = True

    def __post_init__(self):
        self.net = GomokuNet(board_size=self.board_size)
        self.net.to(self.device)

    def predict(self, board) -> Tuple:
        policy, value = self.predict_batch([board])
        return policy[0], float(value[0])

    def predict_batch(self, boards) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized inference for multiple boards to amortize model overhead."""
        self.net.eval()
        use_amp = self.inference_amp and self.device.type == "cuda"
        with torch.inference_mode(), cuda_autocast(use_amp):
            planes = np.stack([b.canonical_board() for b in boards]).astype(np.float32)
            inp = torch.from_numpy(planes).to(self.device, non_blocking=self.device.type == "cuda")
            policy_logits, values = self.net(inp)
            policy = torch.softmax(policy_logits.float(), dim=1).cpu().numpy()
            # Keep `values` iterable even when batch size is 1.
            values = values.float().view(-1)
            return policy, values.cpu().numpy()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

    def train_step(self, batch, optimizer, scaler=None, max_grad_norm: float = 5.0) -> Tuple[float, float]:
        """Batch = (planes, target_policy (probabilities), target_value)."""
        planes, target_p, target_v = batch
        non_blocking = self.device.type == "cuda"
        planes = planes.to(self.device, non_blocking=non_blocking)
        target_p = target_p.to(self.device, non_blocking=non_blocking)
        target_v = target_v.to(self.device, non_blocking=non_blocking)

        # MCTS 방문수가 모두 0인 예외 배치도 학습을 깨지 않도록 확률분포로 보정한다.
        target_sum = target_p.sum(dim=1, keepdim=True)
        uniform = torch.full_like(target_p, 1.0 / target_p.size(1))
        target_p = torch.where(target_sum > 1e-8, target_p / target_sum.clamp_min(1e-8), uniform)

        self.net.train()
        optimizer.zero_grad()
        use_scaler = scaler is not None and scaler.is_enabled()

        def forward_pass():
            p_logits, v = self.net(planes)
            log_probs = F.log_softmax(p_logits, dim=1)
            # 정책: 크로스엔트로피, 가치: MSE
            policy_loss = -(target_p * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(v, target_v)
            return policy_loss, value_loss, policy_loss + value_loss

        if use_scaler:
            with cuda_autocast(True):
                p_loss, v_loss, loss = forward_pass()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            p_loss, v_loss, loss = forward_pass()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            optimizer.step()

        return float(p_loss.item()), float(v_loss.item())

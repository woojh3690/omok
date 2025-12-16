from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .game import GomokuBoard
from .mcts import MCTS


@dataclass
class EpisodeStep:
    planes: np.ndarray
    policy: np.ndarray
    player: int


def select_action(prob: np.ndarray, temperature: float, legal_actions: list[int] | None = None) -> int:
    """Sample an action; if prob is degenerate, fall back to uniform over legal moves."""
    if legal_actions is not None:
        mask = np.zeros_like(prob)
        mask[legal_actions] = prob[legal_actions]
        prob = mask
    if temperature <= 1e-6:
        return int(np.argmax(prob))
    # 온도 샘플링으로 초반 탐험 확대
    scaled = np.power(prob, 1.0 / temperature)
    scaled_sum = scaled.sum()
    if scaled_sum <= 1e-8:
        if legal_actions:
            scaled = np.zeros_like(prob)
            scaled[legal_actions] = 1.0 / len(legal_actions)
        else:
            scaled = np.ones_like(prob) / len(prob)
    else:
        scaled /= scaled_sum
    return int(np.random.choice(len(prob), p=scaled))


def play_self_play_game(
    mcts: MCTS,
    simulations: int = 200,
    temperature: float = 1.0,
    temp_decay_move: int = 20,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    board = GomokuBoard(size=mcts.board_size)
    history: List[EpisodeStep] = []

    while board.winner is None:
        planes = board.canonical_board()
        cur_player = board.current_player
        add_noise = True
        pi = mcts.run(board, num_simulations=simulations, add_noise=add_noise)
        move_temp = temperature if len(history) < temp_decay_move else 1e-3
        legal_actions = board.legal_actions_flat()
        action = select_action(pi, temperature=move_temp, legal_actions=legal_actions)
        x, y = board.from_flat_index(action)
        result = board.play_move(x, y)

        history.append(EpisodeStep(planes=planes, policy=pi, player=cur_player))

        if result.winner is not None:
            break

    winner = board.winner if board.winner is not None else 0
    data = []
    for step in history:
        if winner == 0:
            value = 0.0
        elif winner == step.player:
            value = 1.0
        else:
            value = -1.0
        # (상태, 정책 타겟, 최종 가치) 수집
        data.append((step.planes, step.policy, value))
    return data

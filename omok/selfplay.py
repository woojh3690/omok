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
    prob = np.asarray(prob, dtype=np.float64)
    if legal_actions is not None:
        legal_actions = list(legal_actions)
        if not legal_actions:
            raise ValueError("No legal actions to sample")
        legal_prob = np.zeros_like(prob, dtype=np.float64)
        legal_prob[legal_actions] = np.maximum(prob[legal_actions], 0.0)
        prob = legal_prob

    if temperature <= 0.05:
        if legal_actions is not None:
            legal_scores = prob[legal_actions]
            return int(legal_actions[int(np.argmax(legal_scores))])
        return int(np.argmax(prob))

    # 낮은 온도에서도 언더플로우가 나지 않도록 log 확률에서 softmax를 계산한다.
    positive = np.maximum(prob, 0.0)
    if positive.sum() <= 1e-12:
        if legal_actions is not None:
            scaled = np.zeros_like(prob, dtype=np.float64)
            scaled[legal_actions] = 1.0 / len(legal_actions)
        else:
            scaled = np.ones_like(prob, dtype=np.float64) / len(prob)
    else:
        logits = np.full_like(prob, -np.inf, dtype=np.float64)
        support = np.flatnonzero(positive > 0)
        logits[support] = np.log(positive[support]) / temperature
        max_logit = np.max(logits[support])
        scaled = np.zeros_like(prob, dtype=np.float64)
        scaled[support] = np.exp(logits[support] - max_logit)
        scaled_sum = scaled.sum()
        if scaled_sum <= 1e-12:
            if legal_actions is not None:
                scaled[legal_actions] = 1.0 / len(legal_actions)
            else:
                scaled.fill(1.0 / len(prob))
        else:
            scaled /= scaled_sum
    return int(np.random.choice(len(prob), p=scaled))


def play_self_play_game(
    mcts: MCTS,
    simulations: int = 200,
    temperature: float = 1.0,
    temp_decay_move: int = 20,
    clear_tree: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    if clear_tree:
        mcts.clear()

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

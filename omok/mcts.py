from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .game import GomokuBoard


def _state_key(board: GomokuBoard) -> bytes:
    return board.board.tobytes() + bytes([board.current_player])


@dataclass
class Node:
    player: int
    priors: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def select(self, c_puct: float) -> int:
        total_visits = sum(child.visits for child in self.children.values())
        best_score = -1e9
        best_action = None
        for action, child in self.children.items():
            prior = self.priors.get(action, 0.0)
            # UCB 점수: 탐험(prior) + 가치(Q)
            u = c_puct * prior * math.sqrt(total_visits + 1) / (1 + child.visits)
            score = child.q_value() + u
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            # Fallback to a random move if no children were expanded yet.
            best_action = random.choice(list(self.priors.keys()))
        return best_action

    def expand(self, valid_actions, policy: np.ndarray) -> None:
        self.expanded = True
        for a in valid_actions:
            self.priors[a] = float(policy[a])
            if a not in self.children:
                self.children[a] = Node(player=0)  # player filled lazily

    def add_dirichlet_noise(self, alpha: float, frac: float) -> None:
        actions = list(self.priors.keys())
        if not actions:
            return
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.priors[a] = (1 - frac) * self.priors[a] + frac * float(n)


class MCTS:
    def __init__(
        self,
        model,
        board_size: int = 15,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.03,
        dirichlet_frac: float = 0.25,
        eval_batch_size: int = 16,
    ):
        self.model = model
        self.board_size = board_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.eval_batch_size = eval_batch_size
        self.nodes: Dict[bytes, Node] = {}

    def _get_node(self, board: GomokuBoard) -> Node:
        key = _state_key(board)
        if key not in self.nodes:
            self.nodes[key] = Node(player=board.current_player)
        node = self.nodes[key]
        node.player = board.current_player
        return node

    def run(self, board: GomokuBoard, num_simulations: int, add_noise: bool = False) -> np.ndarray:
        root = self._get_node(board)

        # Expand root if unseen.
        if not root.expanded:
            policy, value = self._evaluate(board)
            valid_actions = [board.to_flat_index(x, y) for (x, y) in board.legal_moves()]
            masked_policy = self._mask_policy(policy, valid_actions)
            root.expand(valid_actions, masked_policy)
        if add_noise:
            # 자가 대국 다양성을 위해 루트에 Dirichlet noise 추가
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_frac)

        pending: list[Tuple[GomokuBoard, list[Tuple[Node, Node]], Node]] = []
        for _ in range(num_simulations):
            sim_board = board.clone()
            path, leaf_node, terminal_value = self._select_leaf(sim_board)
            if terminal_value is not None:
                self._backpropagate(path, terminal_value)
                continue
            if leaf_node is None:
                continue
            pending.append((sim_board, path, leaf_node))
            if len(pending) >= self.eval_batch_size:
                self._evaluate_pending(pending)
                pending.clear()

        if pending:
            self._evaluate_pending(pending)

        visits = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visits
        legal_actions = [board.to_flat_index(x, y) for (x, y) in board.legal_moves()]
        if legal_actions:
            visits = self._mask_policy(visits, legal_actions)
        return visits

    def _select_leaf(
        self, board: GomokuBoard
    ) -> Tuple[list[Tuple[Node, Node]], Optional[Node], Optional[float]]:
        path: list[Tuple[Node, Node]] = []
        node = self._get_node(board)

        while node.expanded and board.winner is None:
            action = node.select(self.c_puct)
            x, y = board.from_flat_index(action)
            result = board.play_move(x, y)
            child_node = self._get_node(board)
            node.children[action] = child_node
            path.append((node, child_node))

            if result.winner is not None:
                leaf_value = self._terminal_value(result.winner, node.player)
                return path, None, leaf_value

            node = child_node

        return path, node, None

    def _evaluate_pending(self, pending: list[Tuple[GomokuBoard, list[Tuple[Node, Node]], Node]]) -> None:
        boards = [b for (b, _, _) in pending]
        policies, values = self._evaluate_batch(boards)
        for (b, path, leaf_node), policy, value in zip(pending, policies, values):
            valid_actions = [b.to_flat_index(x, y) for (x, y) in b.legal_moves()]
            masked_policy = self._mask_policy(policy, valid_actions)
            leaf_node.expand(valid_actions, masked_policy)
            self._backpropagate(path, float(value))

    def _evaluate_batch(self, boards: list[GomokuBoard]) -> Tuple[np.ndarray, np.ndarray]:
        if not boards:
            return np.array([]), np.array([])
        if hasattr(self.model, "predict_batch"):
            policy, value = self.model.predict_batch(boards)
        else:
            policies = []
            values = []
            for b in boards:
                p, v = self.model.predict(b)
                policies.append(p)
                values.append(v)
            policy = np.stack(policies)
            value = np.array(values, dtype=np.float32)
        return policy, value

    def _mask_policy(self, policy: np.ndarray, valid_actions) -> np.ndarray:
        masked = np.zeros_like(policy, dtype=np.float32)
        if not valid_actions:
            return masked
        masked[valid_actions] = policy[valid_actions]
        s = masked.sum()
        if s <= 1e-6:
            masked[valid_actions] = 1.0 / len(valid_actions)
        else:
            masked /= s
        return masked

    def _evaluate(self, board: GomokuBoard) -> Tuple[np.ndarray, float]:
        policy, value = self.model.predict(board)
        return policy, float(value)

    def _terminal_value(self, winner: int, perspective_player: int) -> float:
        if winner == 0:
            return 0.0
        return 1.0 if winner == perspective_player else -1.0

    def _backpropagate(self, path: list[Tuple[Node, Node]], leaf_value: float) -> None:
        value = leaf_value
        for parent, child in reversed(path):
            child.visits += 1
            child.value_sum += value
            value = -value
            parent.visits += 1
            parent.value_sum += value

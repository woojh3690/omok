import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .game import BLACK, WHITE, GomokuBoard


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

    def q_value_for_parent(self, parent_player: int) -> float:
        if self.visits <= 0:
            return 0.0
        if self.player == parent_player:
            return self.q_value()
        return -self.q_value()

    def select(self, c_puct: float) -> int:
        total_visits = sum(child.visits for child in self.children.values())
        best_score = -1e9
        best_action = None
        for action, child in self.children.items():
            prior = self.priors.get(action, 0.0)
            # UCB 점수: 탐험(prior) + 가치(Q)
            u = c_puct * prior * math.sqrt(total_visits + 1) / (1 + child.visits)
            score = child.q_value_for_parent(self.player) + u
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            # 확장된 자식이 없으면 prior 중 하나를 임의로 선택
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

    def clear(self) -> None:
        self.nodes.clear()

    def _one_hot_policy(self, actions: list[int]) -> np.ndarray:
        policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        if actions:
            policy[actions] = 1.0 / len(actions)
        return policy

    def _winning_actions(self, board: GomokuBoard, player: int) -> list[int]:
        actions: list[int] = []
        for action in board.legal_actions_flat():
            x, y = board.from_flat_index(action)
            test_board = board.clone()
            test_board.current_player = player
            result = test_board.play_move(x, y)
            if result.winner == player:
                actions.append(action)
        return actions

    def _tactical_policy(self, board: GomokuBoard) -> Optional[np.ndarray]:
        current_wins = self._winning_actions(board, board.current_player)
        if current_wins:
            return self._one_hot_policy(current_wins)

        opponent = self._opponent(board.current_player)
        opponent_wins = self._winning_actions(board, opponent)
        legal_actions = set(board.legal_actions_flat())
        blocks = [a for a in opponent_wins if a in legal_actions]
        if blocks:
            return self._one_hot_policy(blocks)
        return None

    def _get_node(self, board: GomokuBoard) -> Node:
        key = _state_key(board)
        if key not in self.nodes:
            self.nodes[key] = Node(player=board.current_player)
        node = self.nodes[key]
        node.player = board.current_player
        return node

    def run(self, board: GomokuBoard, num_simulations: int, add_noise: bool = False) -> np.ndarray:
        tactical_policy = self._tactical_policy(board)
        if tactical_policy is not None:
            return tactical_policy

        root = self._get_node(board)

        # 처음 보는 루트만 신경망 prior로 확장
        if not root.expanded:
            policy, value = self._evaluate(board)
            valid_actions = board.legal_actions_flat()
            masked_policy = self._mask_policy(policy, valid_actions)
            root.expand(valid_actions, masked_policy)
        if add_noise:
            # 자가 대국 다양성을 위해 루트에 Dirichlet noise 추가
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_frac)

        start_visits = {action: child.visits for action, child in root.children.items()}
        pending: list[Tuple[GomokuBoard, list[Node], Node]] = []
        pending_keys: set[bytes] = set()
        for _ in range(num_simulations):
            while True:
                sim_board = board.clone()
                path, leaf_node, terminal_value = self._select_leaf(sim_board)
                if terminal_value is not None:
                    self._backpropagate(path, terminal_value)
                    break
                if leaf_node is None:
                    break

                leaf_key = _state_key(sim_board)
                if leaf_key in pending_keys and pending:
                    # 같은 미확장 리프가 중복 선택되면 먼저 평가해서 방문수를 반영
                    self._evaluate_pending(pending)
                    pending.clear()
                    pending_keys.clear()
                    continue

                pending.append((sim_board, path, leaf_node))
                pending_keys.add(leaf_key)
                if len(pending) >= self.eval_batch_size:
                    self._evaluate_pending(pending)
                    pending.clear()
                    pending_keys.clear()
                break

        if pending:
            self._evaluate_pending(pending)

        visits = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = max(0, child.visits - start_visits.get(action, 0))
        legal_actions = board.legal_actions_flat()
        if legal_actions:
            visits = self._mask_policy(visits, legal_actions)
        return visits

    def _select_leaf(
        self, board: GomokuBoard
    ) -> Tuple[list[Node], Optional[Node], Optional[float]]:
        node = self._get_node(board)
        path: list[Node] = [node]

        while node.expanded and board.winner is None:
            action = node.select(self.c_puct)
            x, y = board.from_flat_index(action)
            result = board.play_move(x, y)

            if result.winner is not None:
                child_node = node.children[action]
                child_node.player = self._opponent(node.player)
                path.append(child_node)
                leaf_value = self._terminal_value(result.winner, child_node.player)
                return path, None, leaf_value

            child_node = self._get_node(board)
            node.children[action] = child_node
            path.append(child_node)
            node = child_node

        return path, node, None

    def _evaluate_pending(self, pending: list[Tuple[GomokuBoard, list[Node], Node]]) -> None:
        boards = [b for (b, _, _) in pending]
        policies, values = self._evaluate_batch(boards)
        for (b, path, leaf_node), policy, value in zip(pending, policies, values):
            valid_actions = b.legal_actions_flat()
            masked_policy = self._mask_policy(policy, valid_actions)
            leaf_node.expand(valid_actions, masked_policy)
            # value는 리프에서 둘 차례인 플레이어 관점
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
        policy = np.asarray(policy)
        value = np.asarray(value)
        if policy.ndim == 1:
            policy = policy[None, :]
        if value.ndim == 0:
            value = value.reshape(1)
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

    def _opponent(self, player: int) -> int:
        return WHITE if player == BLACK else BLACK

    def _backpropagate(self, path: list[Node], leaf_value: float) -> None:
        if not path:
            return
        value = leaf_value
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

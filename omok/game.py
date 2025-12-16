from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


BLACK = 1
WHITE = 2
EMPTY = 0


@dataclass
class MoveResult:
    winner: Optional[int]
    foul: bool
    last_move: Tuple[int, int]


class GomokuBoard:
    """
    Long Pro rule:
    - Board is 15x15.
    - Black must win with exactly five. Making an overline (6+) is a foul that awards the win to White.
    - White wins with five or more.
    - Draw when the board is full without a winner.
    """

    def __init__(self, size: int = 15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.winner: Optional[int] = None
        self.foul: bool = False
        self.last_move: Optional[Tuple[int, int]] = None

    def clone(self) -> "GomokuBoard":
        other = GomokuBoard(self.size)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.winner = self.winner
        other.foul = self.foul
        other.last_move = self.last_move
        return other

    def reset(self) -> None:
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.winner = None
        self.foul = False
        self.last_move = None

    def is_valid_move(self, x: int, y: int) -> bool:
        if self.winner is not None:
            return False
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self.board[y, x] == EMPTY

    def legal_moves(self) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(self.size) for x in range(self.size) if self.board[y, x] == EMPTY]

    def _line_length(self, x: int, y: int, dx: int, dy: int) -> int:
        """Count contiguous stones for the player at (x,y) along (dx,dy) both directions."""
        # 양방향으로 같은 색 돌이 얼마나 이어지는지 계산
        player = self.board[y, x]
        count = 1
        cx, cy = x + dx, y + dy
        while 0 <= cx < self.size and 0 <= cy < self.size and self.board[cy, cx] == player:
            count += 1
            cx += dx
            cy += dy
        cx, cy = x - dx, y - dy
        while 0 <= cx < self.size and 0 <= cy < self.size and self.board[cy, cx] == player:
            count += 1
            cx -= dx
            cy -= dy
        return count

    def _max_line_after_move(self, x: int, y: int) -> int:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        return max(self._line_length(x, y, dx, dy) for dx, dy in directions)

    def play_move(self, x: int, y: int) -> MoveResult:
        if not self.is_valid_move(x, y):
            raise ValueError("Invalid move")

        self.board[y, x] = self.current_player
        self.last_move = (x, y)

        max_line = self._max_line_after_move(x, y)
        foul = False
        winner: Optional[int] = None

        # Long Pro: 흑이 6목 이상이면 반칙, 백 승리
        if self.current_player == BLACK and max_line >= 6:
            foul = True
            winner = WHITE
        # 흑은 정확히 5목, 백은 5목 이상이면 승리
        elif max_line == 5 or (self.current_player == WHITE and max_line > 5):
            winner = self.current_player
        elif not self.legal_moves():
            winner = 0  # Draw

        if winner is not None:
            self.winner = winner
            self.foul = foul
        else:
            self.current_player = WHITE if self.current_player == BLACK else BLACK

        return MoveResult(winner=winner, foul=foul, last_move=(x, y))

    def canonical_board(self) -> np.ndarray:
        """
        Return a 2-channel board from the perspective of current_player.
        Channel 0: current player's stones, Channel 1: opponent stones.
        """
        cur = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == (WHITE if self.current_player == BLACK else BLACK)).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    def to_flat_index(self, x: int, y: int) -> int:
        return y * self.size + x

    def from_flat_index(self, idx: int) -> Tuple[int, int]:
        return (idx % self.size, idx // self.size)

    def __repr__(self) -> str:
        mapping = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = [" ".join(mapping[val] for val in row) for row in self.board]
        return "\n".join(lines)

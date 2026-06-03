from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


BLACK = 1
WHITE = 2
EMPTY = 0

DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
FORBIDDEN_OVERLINE = "overline"
FORBIDDEN_DOUBLE_FOUR = "double_four"
FORBIDDEN_DOUBLE_THREE = "double_three"
FORBIDDEN_FOUR_THREE = "four_three"


@dataclass
class MoveResult:
    winner: Optional[int]
    foul: bool
    last_move: Tuple[int, int]


class GomokuBoard:
    """
    Chinese/Renju-style rule preset:
    - Board is 15x15.
    - Swap opening availability is tracked after Black's first move.
    - Black forbidden moves are excluded from legal moves: overline, double-four, double-three,
      and the requested four-three fork ban.
    - Black wins with exactly five unless the move also creates an overline. White wins with five
      or more.
    - Draw when the board is full without a winner.
    """

    def __init__(
        self,
        size: int = 15,
        forbid_black_forbidden_moves: bool = True,
        forbid_black_four_three: bool = True,
        enable_swap_rule: bool = True,
    ):
        self.size = size
        self.forbid_black_forbidden_moves = forbid_black_forbidden_moves
        self.forbid_black_four_three = forbid_black_four_three
        self.enable_swap_rule = enable_swap_rule
        self.swap_used = False
        self.board = np.zeros((size, size), dtype=np.int8)
        self.empty_count = size * size
        self.current_player = BLACK
        self.winner: Optional[int] = None
        self.foul: bool = False
        self.last_move: Optional[Tuple[int, int]] = None

    def clone(self) -> "GomokuBoard":
        other = GomokuBoard(
            self.size,
            forbid_black_forbidden_moves=self.forbid_black_forbidden_moves,
            forbid_black_four_three=self.forbid_black_four_three,
            enable_swap_rule=self.enable_swap_rule,
        )
        other.swap_used = self.swap_used
        other.board = self.board.copy()
        other.empty_count = self.empty_count
        other.current_player = self.current_player
        other.winner = self.winner
        other.foul = self.foul
        other.last_move = self.last_move
        return other

    def reset(self) -> None:
        self.board.fill(EMPTY)
        self.empty_count = self.size * self.size
        self.current_player = BLACK
        self.winner = None
        self.foul = False
        self.last_move = None
        self.swap_used = False

    @property
    def move_count(self) -> int:
        return self.size * self.size - self.empty_count

    def can_swap(self) -> bool:
        # 스왑 룰은 첫 흑 착수 직후 백 차례에서만 한 번 열어 둔다.
        return (
            self.enable_swap_rule
            and not self.swap_used
            and self.winner is None
            and self.move_count == 1
            and self.current_player == WHITE
        )

    def swap_opening(self) -> None:
        # 실제 돌 색은 유지되고, 외부 플레이어의 색 선택만 바뀌는 오프닝 절차를 기록한다.
        if not self.can_swap():
            raise ValueError("Swap is not available")
        self.swap_used = True

    def is_valid_move(self, x: int, y: int) -> bool:
        return self.is_valid_move_for_player(self.current_player, x, y)

    def is_valid_move_for_player(self, player: int, x: int, y: int) -> bool:
        if player not in (BLACK, WHITE):
            return False
        if self.winner is not None:
            return False
        if not self._is_on_board(x, y):
            return False
        if self.board[y, x] != EMPTY:
            return False
        return self.forbidden_reason(player, x, y) is None

    def legal_moves(self) -> List[Tuple[int, int]]:
        if self.empty_count <= 0:
            return []
        coords = np.argwhere(self.board == EMPTY)  # (y, x)
        return [
            (int(x), int(y))
            for y, x in coords
            if self.is_valid_move_for_player(self.current_player, int(x), int(y))
        ]

    def legal_actions_flat(self) -> List[int]:
        """Return legal moves as flat indices (y*size+x)."""
        return self.legal_actions_flat_for_player(self.current_player)

    def legal_actions_flat_for_player(self, player: int) -> List[int]:
        """Return legal moves for a specific player as flat indices."""
        if self.empty_count <= 0:
            return []
        actions: List[int] = []
        for idx in np.flatnonzero(self.board.ravel() == EMPTY):
            x, y = self.from_flat_index(int(idx))
            if self.is_valid_move_for_player(player, x, y):
                actions.append(int(idx))
        return actions

    def forbidden_reason(self, player: int, x: int, y: int) -> Optional[str]:
        if not self.forbid_black_forbidden_moves or player != BLACK:
            return None
        if self.winner is not None or not self._is_on_board(x, y) or self.board[y, x] != EMPTY:
            return None

        self.board[y, x] = player
        try:
            max_line = self._max_line_after_move(x, y)
            if max_line >= 6:
                return FORBIDDEN_OVERLINE
            if max_line == 5:
                return None

            # 흑 금수는 한 수가 만드는 사와 활삼의 개수를 방향 단위로 계산한다.
            four_count = self._count_fours_created(player, x, y)
            three_count = self._count_open_threes_created(player, x, y)
            if four_count >= 2:
                return FORBIDDEN_DOUBLE_FOUR
            if three_count >= 2:
                return FORBIDDEN_DOUBLE_THREE
            if self.forbid_black_four_three and four_count >= 1 and three_count >= 1:
                return FORBIDDEN_FOUR_THREE
            return None
        finally:
            self.board[y, x] = EMPTY

    def _is_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_empty_cell(self, x: int, y: int) -> bool:
        return self._is_on_board(x, y) and self.board[y, x] == EMPTY

    def _line_length(self, x: int, y: int, dx: int, dy: int) -> int:
        """Count contiguous stones for the player at (x,y) along (dx,dy) both directions."""
        # 양방향으로 같은 색 돌이 얼마나 이어지는지 계산한다.
        player = self.board[y, x]
        count = 1
        cx, cy = x + dx, y + dy
        while self._is_on_board(cx, cy) and self.board[cy, cx] == player:
            count += 1
            cx += dx
            cy += dy
        cx, cy = x - dx, y - dy
        while self._is_on_board(cx, cy) and self.board[cy, cx] == player:
            count += 1
            cx -= dx
            cy -= dy
        return count

    def _max_line_after_move(self, x: int, y: int) -> int:
        return max(self._line_length(x, y, dx, dy) for dx, dy in DIRECTIONS)

    def _cell_matches_player_or_fill(
        self,
        player: int,
        x: int,
        y: int,
        fill_x: int,
        fill_y: int,
    ) -> bool:
        if not self._is_on_board(x, y):
            return False
        return (x == fill_x and y == fill_y) or self.board[y, x] == player

    def _window_is_exact_five_after_fill(
        self,
        player: int,
        anchor_x: int,
        anchor_y: int,
        dx: int,
        dy: int,
        start_offset: int,
        fill_x: int,
        fill_y: int,
    ) -> bool:
        # 끊긴 사도 한 칸을 메웠을 때 정확히 오목이 되는지 5칸 창으로 판정한다.
        for offset in range(start_offset, start_offset + 5):
            px = anchor_x + offset * dx
            py = anchor_y + offset * dy
            if not self._cell_matches_player_or_fill(player, px, py, fill_x, fill_y):
                return False

        before_x = anchor_x + (start_offset - 1) * dx
        before_y = anchor_y + (start_offset - 1) * dy
        after_x = anchor_x + (start_offset + 5) * dx
        after_y = anchor_y + (start_offset + 5) * dy
        if self._is_on_board(before_x, before_y) and self.board[before_y, before_x] == player:
            return False
        if self._is_on_board(after_x, after_y) and self.board[after_y, after_x] == player:
            return False
        return True

    def _has_four_in_direction(self, player: int, x: int, y: int, dx: int, dy: int) -> bool:
        # 같은 방향의 열린 사는 완성점이 둘이어도 한 방향의 사 하나로 센다.
        for fill_offset in range(-4, 5):
            fill_x = x + fill_offset * dx
            fill_y = y + fill_offset * dy
            if not self._is_empty_cell(fill_x, fill_y):
                continue
            for start_offset in range(-4, 1):
                if not (start_offset <= fill_offset <= start_offset + 4):
                    continue
                if self._window_is_exact_five_after_fill(
                    player, x, y, dx, dy, start_offset, fill_x, fill_y
                ):
                    return True
        return False

    def _window_is_open_four_after_fill(
        self,
        player: int,
        anchor_x: int,
        anchor_y: int,
        dx: int,
        dy: int,
        start_offset: int,
        fill_offset: int,
        fill_x: int,
        fill_y: int,
    ) -> bool:
        # 활삼은 한 수 뒤 양끝이 열린 연속 사가 되는지를 기준으로 잡는다.
        if not (start_offset <= fill_offset <= start_offset + 3):
            return False
        for offset in range(start_offset, start_offset + 4):
            px = anchor_x + offset * dx
            py = anchor_y + offset * dy
            if not self._cell_matches_player_or_fill(player, px, py, fill_x, fill_y):
                return False

        before_x = anchor_x + (start_offset - 1) * dx
        before_y = anchor_y + (start_offset - 1) * dy
        after_x = anchor_x + (start_offset + 4) * dx
        after_y = anchor_y + (start_offset + 4) * dy
        return self._is_empty_cell(before_x, before_y) and self._is_empty_cell(after_x, after_y)

    def _has_open_three_in_direction(self, player: int, x: int, y: int, dx: int, dy: int) -> bool:
        # 한 방향에서 여러 활삼 형태가 잡혀도 금수 계산에는 방향 하나로만 반영한다.
        for fill_offset in range(-3, 4):
            fill_x = x + fill_offset * dx
            fill_y = y + fill_offset * dy
            if not self._is_empty_cell(fill_x, fill_y):
                continue
            for start_offset in range(-3, 1):
                if self._window_is_open_four_after_fill(
                    player, x, y, dx, dy, start_offset, fill_offset, fill_x, fill_y
                ):
                    return True
        return False

    def _count_fours_created(self, player: int, x: int, y: int) -> int:
        return sum(
            1 for dx, dy in DIRECTIONS if self._has_four_in_direction(player, x, y, dx, dy)
        )

    def _count_open_threes_created(self, player: int, x: int, y: int) -> int:
        return sum(
            1 for dx, dy in DIRECTIONS if self._has_open_three_in_direction(player, x, y, dx, dy)
        )

    def winner_after_virtual_move(self, player: int, x: int, y: int) -> Optional[int]:
        if self.winner is not None:
            return None
        if not self._is_on_board(x, y):
            return None
        if self.board[y, x] != EMPTY:
            return None
        if self.forbidden_reason(player, x, y) is not None:
            return WHITE

        self.board[y, x] = player
        try:
            max_line = self._max_line_after_move(x, y)
        finally:
            self.board[y, x] = EMPTY

        # 흑은 정확히 5목, 백은 5목 이상을 승리로 본다.
        if player == BLACK and max_line >= 6:
            return WHITE
        if max_line == 5 or (player == WHITE and max_line > 5):
            return player
        return None

    def play_move(self, x: int, y: int) -> MoveResult:
        if not self.is_valid_move(x, y):
            raise ValueError("Invalid move")

        self.board[y, x] = self.current_player
        self.empty_count -= 1
        self.last_move = (x, y)

        max_line = self._max_line_after_move(x, y)
        foul = False
        winner: Optional[int] = None

        # 금수 마스크를 끈 경우에도 기존 Long Pro 장목 반칙 판정은 유지한다.
        if self.current_player == BLACK and max_line >= 6:
            foul = True
            winner = WHITE
        elif max_line == 5 or (self.current_player == WHITE and max_line > 5):
            winner = self.current_player
        elif self.empty_count <= 0:
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

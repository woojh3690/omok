import unittest

from omok.game import (
    BLACK,
    FORBIDDEN_DOUBLE_FOUR,
    FORBIDDEN_DOUBLE_THREE,
    FORBIDDEN_FOUR_THREE,
    FORBIDDEN_OVERLINE,
    WHITE,
    GomokuBoard,
)


def make_board(
    *,
    current_player: int,
    black: list[tuple[int, int]],
    white: list[tuple[int, int]] | None = None,
    forbid_black_four_three: bool = True,
) -> GomokuBoard:
    # 테스트 패턴은 직접 돌을 배치한 뒤 남은 칸 수와 차례를 맞춘다.
    board = GomokuBoard(forbid_black_four_three=forbid_black_four_three)
    for x, y in black:
        board.board[y, x] = BLACK
    for x, y in white or []:
        board.board[y, x] = WHITE
    board.empty_count = board.size * board.size - len(black) - len(white or [])
    board.current_player = current_player
    return board


class ChineseRulesTest(unittest.TestCase):
    def test_black_overline_is_forbidden(self) -> None:
        # 흑 장목은 합법수에서 제외되고 가상 승리도 흑 승리로 보지 않는다.
        board = make_board(
            current_player=BLACK,
            black=[(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
        )

        self.assertEqual(board.forbidden_reason(BLACK, 8, 7), FORBIDDEN_OVERLINE)
        self.assertFalse(board.is_valid_move(8, 7))
        self.assertNotIn(board.to_flat_index(8, 7), board.legal_actions_flat())
        self.assertEqual(board.winner_after_virtual_move(BLACK, 8, 7), WHITE)

    def test_white_overline_is_allowed_and_wins(self) -> None:
        # 백은 장목 제한을 받지 않으므로 6목 이상도 승리로 처리한다.
        board = make_board(
            current_player=WHITE,
            black=[],
            white=[(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
        )

        self.assertTrue(board.is_valid_move(8, 7))
        result = board.play_move(8, 7)
        self.assertEqual(result.winner, WHITE)

    def test_black_double_four_is_forbidden(self) -> None:
        # 새 흑돌 하나가 가로와 세로 사를 동시에 만들면 사사 금수다.
        board = make_board(
            current_player=BLACK,
            black=[(5, 7), (6, 7), (8, 7), (7, 5), (7, 6), (7, 8)],
        )

        self.assertEqual(board.forbidden_reason(BLACK, 7, 7), FORBIDDEN_DOUBLE_FOUR)
        self.assertFalse(board.is_valid_move(7, 7))

    def test_black_double_three_is_forbidden(self) -> None:
        # 새 흑돌 하나가 서로 다른 두 방향의 활삼을 만들면 삼삼 금수다.
        board = make_board(
            current_player=BLACK,
            black=[(6, 7), (8, 7), (7, 6), (7, 8)],
        )

        self.assertEqual(board.forbidden_reason(BLACK, 7, 7), FORBIDDEN_DOUBLE_THREE)
        self.assertFalse(board.is_valid_move(7, 7))

    def test_black_four_three_is_forbidden_by_requested_preset(self) -> None:
        # 요청한 중국식 프리셋에서는 한 방향 사와 한 방향 활삼을 동시에 만드는 수도 금지한다.
        board = make_board(
            current_player=BLACK,
            black=[(5, 7), (6, 7), (8, 7), (7, 6), (7, 8)],
        )

        self.assertEqual(board.forbidden_reason(BLACK, 7, 7), FORBIDDEN_FOUR_THREE)
        self.assertFalse(board.is_valid_move(7, 7))

    def test_four_three_ban_can_be_disabled(self) -> None:
        # 공식 렌주식 사삼 허용 변형을 위해 4x3 금지만 별도로 끌 수 있게 둔다.
        board = make_board(
            current_player=BLACK,
            black=[(5, 7), (6, 7), (8, 7), (7, 6), (7, 8)],
            forbid_black_four_three=False,
        )

        self.assertIsNone(board.forbidden_reason(BLACK, 7, 7))
        self.assertTrue(board.is_valid_move(7, 7))

    def test_exact_five_wins_before_fork_bans(self) -> None:
        # 흑이 정확히 오목을 완성하면 동시에 생기는 일반 포크 금수보다 승리를 우선한다.
        board = make_board(
            current_player=BLACK,
            black=[(3, 7), (4, 7), (5, 7), (6, 7), (7, 6), (7, 8)],
        )

        self.assertTrue(board.is_valid_move(7, 7))
        result = board.play_move(7, 7)
        self.assertEqual(result.winner, BLACK)

    def test_swap_available_after_first_black_move(self) -> None:
        # 스왑 기회는 첫 흑 착수 직후 백 차례에 한 번만 열린다.
        board = GomokuBoard()
        board.play_move(7, 7)

        self.assertTrue(board.can_swap())
        board.swap_opening()
        self.assertFalse(board.can_swap())


if __name__ == "__main__":
    unittest.main()

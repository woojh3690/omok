import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .game import BLACK, WHITE, GomokuBoard
from .mcts import MCTS
from .model import ModelWrapper


@dataclass
class TacticalCase:
    name: str
    board: GomokuBoard
    expected_actions: set[int]


@dataclass
class TacticalResult:
    name: str
    passed: bool
    mcts_action: int
    raw_top_action: int
    raw_expected_rank: int
    raw_value: float


def _make_board(
    board_size: int,
    current_player: int,
    black: list[tuple[int, int]],
    white: list[tuple[int, int]],
) -> GomokuBoard:
    board = GomokuBoard(size=board_size)
    for x, y in black:
        board.board[y, x] = BLACK
    for x, y in white:
        board.board[y, x] = WHITE
    board.empty_count = board_size * board_size - len(black) - len(white)
    board.current_player = current_player
    return board


def _idx(board: GomokuBoard, x: int, y: int) -> int:
    return board.to_flat_index(x, y)


def build_tactical_cases(board_size: int = 15) -> list[TacticalCase]:
    mid = board_size // 2
    cases = []

    board = _make_board(
        board_size,
        BLACK,
        black=[(mid - 2, mid), (mid - 1, mid), (mid, mid), (mid + 1, mid)],
        white=[(mid - 2, mid - 1), (mid - 1, mid - 1), (mid, mid - 1)],
    )
    cases.append(
        TacticalCase(
            name="black_immediate_win",
            board=board,
            expected_actions={_idx(board, mid - 3, mid), _idx(board, mid + 2, mid)},
        )
    )

    board = _make_board(
        board_size,
        WHITE,
        black=[(mid - 2, mid - 1), (mid - 1, mid - 1), (mid, mid - 1)],
        white=[(mid - 2, mid), (mid - 1, mid), (mid, mid), (mid + 1, mid)],
    )
    cases.append(
        TacticalCase(
            name="white_immediate_win",
            board=board,
            expected_actions={_idx(board, mid - 3, mid), _idx(board, mid + 2, mid)},
        )
    )

    board = _make_board(
        board_size,
        BLACK,
        black=[(mid, mid - 1), (mid + 1, mid - 1)],
        white=[(mid - 2, mid), (mid - 1, mid), (mid, mid), (mid + 1, mid)],
    )
    cases.append(
        TacticalCase(
            name="black_blocks_white_four",
            board=board,
            expected_actions={_idx(board, mid - 3, mid), _idx(board, mid + 2, mid)},
        )
    )

    board = _make_board(
        board_size,
        WHITE,
        black=[(mid - 2, mid), (mid - 1, mid), (mid, mid), (mid + 1, mid)],
        white=[(mid, mid - 1), (mid + 1, mid - 1)],
    )
    cases.append(
        TacticalCase(
            name="white_blocks_black_four",
            board=board,
            expected_actions={_idx(board, mid - 3, mid), _idx(board, mid + 2, mid)},
        )
    )
    return cases


def _best_legal_action(policy: np.ndarray, board: GomokuBoard) -> int:
    legal_actions = board.legal_actions_flat()
    return int(max(legal_actions, key=lambda action: policy[action]))


def _best_expected_rank(policy: np.ndarray, board: GomokuBoard, expected_actions: set[int]) -> int:
    legal_actions = board.legal_actions_flat()
    ranked = sorted(legal_actions, key=lambda action: policy[action], reverse=True)
    ranks = [rank for rank, action in enumerate(ranked, start=1) if action in expected_actions]
    return min(ranks) if ranks else len(ranked) + 1


def evaluate_tactical_positions(
    model: ModelWrapper, board_size: int = 15, simulations: int = 128
) -> list[TacticalResult]:
    results = []
    for case in build_tactical_cases(board_size):
        raw_policy, raw_value = model.predict(case.board)
        raw_top_action = _best_legal_action(raw_policy, case.board)
        raw_expected_rank = _best_expected_rank(raw_policy, case.board, case.expected_actions)

        mcts = MCTS(model=model, board_size=board_size)
        visit_policy = mcts.run(case.board.clone(), num_simulations=simulations, add_noise=False)
        mcts_action = _best_legal_action(visit_policy, case.board)
        results.append(
            TacticalResult(
                name=case.name,
                passed=mcts_action in case.expected_actions,
                mcts_action=mcts_action,
                raw_top_action=raw_top_action,
                raw_expected_rank=raw_expected_rank,
                raw_value=float(raw_value),
            )
        )
    return results


def _format_action(board_size: int, action: int) -> str:
    x = action % board_size
    y = action // board_size
    return f"({x},{y})"


def load_model(weights: Optional[Path], board_size: int) -> ModelWrapper:
    model = ModelWrapper(board_size=board_size)
    if weights is not None and weights.exists():
        model.load(weights)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained Omok checkpoints.")
    parser.add_argument("--weights", type=Path, default=Path("checkpoints/latest/latest.pt"))
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--simulations", type=int, default=128)
    args = parser.parse_args()

    model = load_model(args.weights, args.board_size)
    results = evaluate_tactical_positions(model, board_size=args.board_size, simulations=args.simulations)
    failed = [result for result in results if not result.passed]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.name}: "
            f"mcts={_format_action(args.board_size, result.mcts_action)} "
            f"raw_top={_format_action(args.board_size, result.raw_top_action)} "
            f"raw_expected_rank={result.raw_expected_rank} raw_value={result.raw_value:.3f}"
        )

    if failed:
        print(f"Tactical evaluation failed: {len(failed)}/{len(results)}")
        return 1
    print(f"Tactical evaluation passed: {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

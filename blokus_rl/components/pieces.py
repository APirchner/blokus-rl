import itertools
from typing import (
    Callable, Dict, List, Tuple
)

import numpy as np
from numba import njit

from blokus_rl.envs.constants import BOARD_SIZE
from blokus_rl.components.piece_config import PIECE_CONFIG

OperationType = Callable[[np.array], np.array] | None
ActionType = Tuple[int, int, OperationType]


class Piece:
    def __init__(
            self,
            name: str,
            shape: np.array,
            operations: Dict[str, OperationType],
            idx_offset: int
    ):
        self.name = name
        self.shape = shape
        self.operations = [op for _, op in operations.items()]
        self.idx_offset = idx_offset
        self.actions = self._generate_actions(self.shape, self.operations)

    @staticmethod
    # @njit(parallel=True)
    def _generate_actions(shape: np.array, operations: List[OperationType]) -> List[ActionType]:
        actions = []
        x_upper = BOARD_SIZE - shape.shape[0] + 1
        y_upper = BOARD_SIZE - shape.shape[1] + 1
        actions.extend([(x, y, None) for x, y in itertools.product(range(x_upper), range(y_upper))])
        for op in operations:
            shape_mod = op(shape)
            x_upper = BOARD_SIZE - shape_mod.shape[0] + 1
            y_upper = BOARD_SIZE - shape_mod.shape[1] + 1
            actions.extend([(x, y, op) for x, y in itertools.product(range(x_upper), range(y_upper))])
        return actions

    def idx_range(self):
        return self.idx_offset, self.idx_offset + len(self.actions) - 1

    def size(self):
        return self.shape.sum(axis=(0, 1))

    def map_action_idx_to_action(self, action_idx) -> ActionType:
        action_idx_offset = action_idx - self.idx_offset
        return self.actions[action_idx_offset]

    def filled_at_origin(self, action_idx: int) -> bool:
        action = self.map_action_idx_to_action(action_idx)
        shape_mod = action[2](self.shape) if action[2] else self.shape
        return shape_mod[0, 0] == 1

    def map_action_idx_to_board(self, action_idx: int) -> np.array:
        action = self.map_action_idx_to_action(action_idx)
        shape_mod = action[2](self.shape) if action[2] else self.shape
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype='u1')
        board[action[0]:action[0] + shape_mod.shape[0], action[1]:action[1] + shape_mod.shape[1]] = shape_mod
        return board


def build_piece_set() -> Tuple[Dict[int, Piece], int]:
    pieces = {}
    idx_offset = 0
    for _, p in PIECE_CONFIG.items():
        piece = Piece(
            name=p.name,
            shape=p.shape,
            operations=p.operations,
            idx_offset=idx_offset
        )
        pieces[piece.idx_offset] = piece
        idx_offset = piece.idx_range()[1] + 1
    return pieces, idx_offset


def total_score() -> int:
    return sum((p.shape.sum(axis=(0, 1)) for _, p in PIECE_CONFIG.items()))

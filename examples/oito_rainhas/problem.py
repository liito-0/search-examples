"""Problema das oito rainhas com construção incremental linha a linha."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib import Action, Problem, State

BOARD_SIZE = 8


@dataclass(frozen=True)
class QueensState(State):
    """Coluna ocupada em cada linha já preenchida, de cima para baixo."""

    columns: Tuple[int, ...] = ()


class PlaceQueen(Action):
    """Coloca uma rainha na próxima linha, na coluna indicada."""

    def __init__(self, column: int) -> None:
        self.column = column

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, QueensState) or len(state.columns) >= BOARD_SIZE:
            return None

        row = len(state.columns)
        if not 0 <= self.column < BOARD_SIZE:
            return None

        for previous_row, previous_column in enumerate(state.columns):
            if (
                previous_column == self.column
                or abs(previous_column - self.column) == row - previous_row
            ):
                return None

        return QueensState(state.columns + (self.column,)), 1.0

    def __repr__(self) -> str:
        return f"COLOCAR_RAINHA({self.column})"


class EightQueens(Problem):
    """Posiciona oito rainhas sem ataques em um tabuleiro 8x8."""

    initial_state = QueensState()
    goal_depth = BOARD_SIZE

    def actions(self, state: State) -> List[Action]:
        if not isinstance(state, QueensState) or len(state.columns) >= BOARD_SIZE:
            return []
        return [PlaceQueen(column) for column in range(BOARD_SIZE)]

    def is_goal(self, state: State) -> bool:
        return isinstance(state, QueensState) and len(state.columns) == BOARD_SIZE

    def heuristic(self, state: State) -> float:
        if not isinstance(state, QueensState):
            raise TypeError("state deve ser QueensState")
        return float(BOARD_SIZE - len(state.columns))
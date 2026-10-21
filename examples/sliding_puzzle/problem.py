"""Problema do quebra-cabeça deslizante."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from search_lib import Action, Problem, State  # type: ignore[import-untyped]


@dataclass(frozen=True)
class SlidingPuzzleState(State):
    """Estado imutável do tabuleiro."""

    tiles: Tuple[int, ...]


class SlidingPuzzleAction(Action):
    """Move o espaço vazio em uma direção."""

    _DELTAS: Dict[str, Tuple[int, int]] = {
        "CIMA": (-1, 0),
        "BAIXO": (1, 0),
        "ESQUERDA": (0, -1),
        "DIREITA": (0, 1),
    }
    _INVERSES: Dict[str, str] = {
        "CIMA": "BAIXO",
        "BAIXO": "CIMA",
        "ESQUERDA": "DIREITA",
        "DIREITA": "ESQUERDA",
    }

    def __init__(self, direction: str, rows: int, cols: int) -> None:
        self.direction: str = direction
        self.rows: int = rows
        self.cols: int = cols

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, SlidingPuzzleState):
            return None
        if len(state.tiles) != self.rows * self.cols or 0 not in state.tiles:
            return None

        blank = state.tiles.index(0)
        row, col = divmod(blank, self.cols)
        row_delta, col_delta = self._DELTAS[self.direction]
        next_row, next_col = row + row_delta, col + col_delta
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return None

        target = next_row * self.cols + next_col
        tiles = list(state.tiles)
        tiles[blank], tiles[target] = tiles[target], tiles[blank]
        return SlidingPuzzleState(tuple(tiles)), 1.0

    def __repr__(self) -> str:
        return f"SlidingPuzzleAction({self.direction})"


class SlidingPuzzle(Problem):
    """Quebra-cabeça parametrizável com heurísticas comuns."""

    def __init__(
        self, rows: int = 3, cols: int = 3, heuristic: str = "manhattan"
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows e cols devem ser positivos")
        if heuristic not in ("manhattan", "misplaced"):
            raise ValueError("heuristic deve ser 'manhattan' ou 'misplaced'")

        self.rows: int = rows
        self.cols: int = cols
        self.heuristic_name: str = heuristic
        self.goal_state: SlidingPuzzleState = SlidingPuzzleState(
            tuple(range(1, rows * cols)) + (0,)
        )
        self.initial_state: SlidingPuzzleState = self.goal_state
        self._actions: List[SlidingPuzzleAction] = [
            SlidingPuzzleAction(direction, rows, cols)
            for direction in ("CIMA", "BAIXO", "ESQUERDA", "DIREITA")
        ]

    def actions(self, state: State) -> List[Action]:
        """Retorna movimentos válidos a partir do estado."""
        return [action for action in self._actions if action.apply(state) is not None]

    def is_goal(self, state: State) -> bool:
        """Verifica se o tabuleiro está resolvido."""
        return isinstance(state, SlidingPuzzleState) and state == self.goal_state

    def heuristic(self, state: State) -> float:
        """Calcula Manhattan ou o número de peças fora do lugar."""
        if not isinstance(state, SlidingPuzzleState):
            raise TypeError("state deve ser um SlidingPuzzleState")

        if self.heuristic_name == "misplaced":
            return float(
                sum(
                    tile != 0 and tile != goal
                    for tile, goal in zip(state.tiles, self.goal_state.tiles)
                )
            )

        distance = 0
        for index, tile in enumerate(state.tiles):
            if tile == 0:
                continue
            goal_index = tile - 1
            row, col = divmod(index, self.cols)
            goal_row, goal_col = divmod(goal_index, self.cols)
            distance += abs(row - goal_row) + abs(col - goal_col)
        return float(distance)

    def random_state(
        self, n_moves: int, avoid_repeats: bool = True, seed: Optional[int] = None
    ) -> SlidingPuzzleState:
        """Gera um estado solúvel por caminhada aleatória desde o objetivo."""
        if n_moves < 0:
            raise ValueError("n_moves não pode ser negativo")

        generator = random.Random(seed)
        current = self.goal_state
        visited = {current}
        previous_direction: Optional[str] = None

        for _ in range(n_moves):
            candidates: List[Tuple[SlidingPuzzleAction, SlidingPuzzleState]] = []
            for action in self._actions:
                if (
                    avoid_repeats
                    and previous_direction is not None
                    and action.direction == action._INVERSES[previous_direction]
                ):
                    continue
                result = action.apply(current)
                if result is None:
                    continue
                next_state = result[0]
                if avoid_repeats and next_state in visited:
                    continue
                candidates.append((action, next_state))

            if not candidates:
                raise ValueError(
                    "não há movimentos sem repetição suficientes para n_moves"
                )

            action, current = generator.choice(candidates)
            previous_direction = action.direction
            visited.add(current)

        return current

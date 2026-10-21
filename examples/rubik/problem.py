"""Problema do cubo de Rubik com movimentos de face."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from search_lib import Action, Problem, State  # type: ignore[import-untyped]

Coordinate = Tuple[int, int, int]
Facelet = Tuple[Coordinate, Coordinate]


@dataclass(frozen=True)
class RubikCubeState(State):
    """Estado imutável representado pelas cores dos facelets."""

    facelets: Tuple[str, ...]


class RubikCubeAction(Action):
    """Aplica uma permutação de facelets pré-calculada."""

    def __init__(self, move: str, permutation: Tuple[int, ...]) -> None:
        self.move: str = move
        self.permutation: Tuple[int, ...] = permutation

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, RubikCubeState):
            return None
        if len(state.facelets) != len(self.permutation):
            return None
        result = tuple(state.facelets[source] for source in self.permutation)
        return RubikCubeState(result), 1.0

    def __repr__(self) -> str:
        return f"RubikCubeAction({self.move})"


class RubikCube(Problem):
    """Cubo 2x2 ou 3x3 com movimentos de quarto, meio ou ambos."""

    _FACE_NORMALS: Dict[str, Coordinate] = {
        "U": (0, 1, 0),
        "D": (0, -1, 0),
        "L": (-1, 0, 0),
        "R": (1, 0, 0),
        "F": (0, 0, 1),
        "B": (0, 0, -1),
    }
    _QUARTER_MOVES: Tuple[str, ...] = (
        "U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"
    )
    _HALF_MOVES: Tuple[str, ...] = ("U2", "D2", "L2", "R2", "F2", "B2")

    def __init__(self, size: int = 3, action_set: str = "quarter") -> None:
        if size not in (2, 3):
            raise ValueError("size deve ser 2 ou 3")
        if action_set not in ("quarter", "half", "full"):
            raise ValueError("action_set deve ser 'quarter', 'half' ou 'full'")

        self.size: int = size
        self.action_set: str = action_set
        self._facelets: List[Facelet] = self._make_facelets(size)
        self._indices: Dict[Facelet, int] = {
            facelet: index for index, facelet in enumerate(self._facelets)
        }
        self.goal_state: RubikCubeState = RubikCubeState(
            tuple(
                face
                for face in ("U", "D", "L", "R", "F", "B")
                for _ in range(size * size)
            )
        )
        self.initial_state: RubikCubeState = self.goal_state

        if action_set == "quarter":
            moves: Tuple[str, ...] = self._QUARTER_MOVES
        elif action_set == "half":
            moves = self._HALF_MOVES
        else:
            moves = self._QUARTER_MOVES + self._HALF_MOVES
        self._actions: List[RubikCubeAction] = [
            RubikCubeAction(move, self._make_permutation(move)) for move in moves
        ]
        self._inverse_moves: Dict[str, str] = {
            move: move[:-1] if move.endswith("'") else move + "'"
            for move in self._QUARTER_MOVES
        }
        self._inverse_moves.update({move: move for move in self._HALF_MOVES})

    @staticmethod
    def _make_facelets(size: int) -> List[Facelet]:
        """Cria a geometria (posição, normal) de cada facelet."""
        coordinates: List[Facelet] = []
        maximum = size - 1
        for face in ("U", "D", "L", "R", "F", "B"):
            for row in range(size):
                for col in range(size):
                    if face == "U":
                        position, normal = (col, maximum, row), (0, 1, 0)
                    elif face == "D":
                        position, normal = (col, 0, maximum - row), (0, -1, 0)
                    elif face == "L":
                        position, normal = (0, maximum - row, col), (-1, 0, 0)
                    elif face == "R":
                        position, normal = (maximum, maximum - row, maximum - col), (1, 0, 0)
                    elif face == "F":
                        position, normal = (col, maximum - row, maximum), (0, 0, 1)
                    else:
                        position, normal = (maximum - col, maximum - row, 0), (0, 0, -1)
                    coordinates.append((position, normal))
        return coordinates

    @staticmethod
    def _rotate_vector(
        vector: Coordinate, axis: int, quarter_turns: int
    ) -> Coordinate:
        x, y, z = vector
        for _ in range(quarter_turns % 4):
            if axis == 0:
                x, y, z = x, -z, y
            elif axis == 1:
                x, y, z = z, y, -x
            else:
                x, y, z = -y, x, z
        return x, y, z

    @classmethod
    def _rotate(
        cls, position: Coordinate, axis: int, quarter_turns: int, size: int
    ) -> Coordinate:
        centered = (
            2 * position[0] - (size - 1),
            2 * position[1] - (size - 1),
            2 * position[2] - (size - 1),
        )
        rotated = cls._rotate_vector(centered, axis, quarter_turns)
        x, y, z = rotated
        return (
            (x + size - 1) // 2,
            (y + size - 1) // 2,
            (z + size - 1) // 2,
        )

    def _make_permutation(self, move: str) -> Tuple[int, ...]:
        """Calcula a permutação para o movimento indicado."""
        face = move[0]
        normal = self._FACE_NORMALS[face]
        axis = next(index for index, value in enumerate(normal) if value != 0)
        face_sign = normal[axis]
        clockwise_turns = 2 if move.endswith("2") else (-1 if move.endswith("'") else 1)
        quarter_turns = -face_sign * clockwise_turns
        maximum = self.size - 1
        permutation = list(range(len(self._facelets)))

        for source, (position, sticker_normal) in enumerate(self._facelets):
            if position[axis] != (maximum if face_sign > 0 else 0):
                continue
            target_position = self._rotate(position, axis, quarter_turns, self.size)
            target_normal = self._rotate_vector(sticker_normal, axis, quarter_turns)
            target = self._indices[(target_position, target_normal)]
            permutation[target] = source
        return tuple(permutation)

    def actions(self, state: State) -> List[Action]:
        """Retorna os movimentos disponíveis para o cubo."""
        return list(self._actions)

    def is_goal(self, state: State) -> bool:
        """Verifica se todas as faces têm uma única cor resolvida."""
        return isinstance(state, RubikCubeState) and state == self.goal_state

    def heuristic(self, state: State) -> float:
        """Estima o custo pelos facelets incorretos; pode não ser admissível."""
        if not isinstance(state, RubikCubeState):
            raise TypeError("state deve ser um RubikCubeState")
        misplaced = sum(
            current != goal
            for current, goal in zip(state.facelets, self.goal_state.facelets)
        )
        return misplaced / 8.0

    def random_state(
        self, n_moves: int, avoid_repeats: bool = True, seed: Optional[int] = None
    ) -> RubikCubeState:
        """Gera um estado por caminhada aleatória desde o cubo resolvido."""
        if n_moves < 0:
            raise ValueError("n_moves não pode ser negativo")

        generator = random.Random(seed)
        current = self.goal_state
        visited = {current}
        previous_move: Optional[str] = None

        for _ in range(n_moves):
            candidates: List[Tuple[RubikCubeAction, RubikCubeState]] = []
            for action in self._actions:
                if (
                    previous_move is not None
                    and self._inverse_moves.get(previous_move) == action.move
                ):
                    continue
                result = action.apply(current)
                if result is None:
                    continue
                next_state = result[0]
                if not isinstance(next_state, RubikCubeState):
                    continue
                if avoid_repeats and next_state in visited:
                    continue
                candidates.append((action, next_state))

            if not candidates:
                raise ValueError(
                    "não há movimentos sem repetição suficientes para n_moves"
                )

            action, current = generator.choice(candidates)
            previous_move = action.move
            visited.add(current)

        return current

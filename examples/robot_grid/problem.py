"""Problema de navegação de um robô em um grid com obstáculos."""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from search_lib import Action, Problem, State  # type: ignore[import-untyped]

Direction = str
Position = Tuple[int, int]
GridMap = Tuple[str, ...]


@dataclass(frozen=True)
class GridRobotState(State):
    """Posição e direção imutáveis do robô."""

    x: int
    y: int
    direction: Direction


class GridRobotAction(Action):
    """Gira o robô ou avança uma célula."""

    _DIRECTIONS: Tuple[Direction, ...] = ("NORTE", "LESTE", "SUL", "OESTE")
    _DELTAS: Dict[Direction, Position] = {
        "NORTE": (0, -1),
        "LESTE": (1, 0),
        "SUL": (0, 1),
        "OESTE": (-1, 0),
    }

    def __init__(
        self,
        action_name: str,
        grid_map: GridMap,
        cost: float,
    ) -> None:
        self.action_name: str = action_name
        self.grid_map: GridMap = grid_map
        self.height: int = len(grid_map)
        self.width: int = len(grid_map[0])
        self.cost: float = cost

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, GridRobotState):
            return None
        if state.direction not in self._DIRECTIONS:
            return None
        if not (0 <= state.x < self.width and 0 <= state.y < self.height):
            return None
        if self.grid_map[state.y][state.x] == "#":
            return None

        direction_index = self._DIRECTIONS.index(state.direction)
        if self.action_name == "GIRAR_HORARIO":
            next_direction = self._DIRECTIONS[(direction_index + 1) % 4]
            return GridRobotState(state.x, state.y, next_direction), self.cost
        if self.action_name == "GIRAR_ANTIHORARIO":
            next_direction = self._DIRECTIONS[(direction_index - 1) % 4]
            return GridRobotState(state.x, state.y, next_direction), self.cost
        if self.action_name == "GIRAR_180":
            next_direction = self._DIRECTIONS[(direction_index + 2) % 4]
            return GridRobotState(state.x, state.y, next_direction), self.cost
        if self.action_name == "FRENTE":
            delta_x, delta_y = self._DELTAS[state.direction]
            next_x, next_y = state.x + delta_x, state.y + delta_y
            if not (0 <= next_x < self.width and 0 <= next_y < self.height):
                return None
            if self.grid_map[next_y][next_x] == "#":
                return None
            return GridRobotState(next_x, next_y, state.direction), self.cost
        return None

    def __repr__(self) -> str:
        return f"GridRobotAction({self.action_name})"


class GridRobot(Problem):
    """Grid retangular com ações, custos, objetivo e heurísticas configuráveis."""

    ACTION_NAMES: Tuple[str, ...] = (
        "GIRAR_HORARIO",
        "GIRAR_ANTIHORARIO",
        "FRENTE",
        "GIRAR_180",
    )
    _INVERSE_ACTIONS: Dict[str, str] = {
        "GIRAR_HORARIO": "GIRAR_ANTIHORARIO",
        "GIRAR_ANTIHORARIO": "GIRAR_HORARIO",
        "GIRAR_180": "GIRAR_180",
    }

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        obstacles: Optional[Sequence[str]] = None,
        file: Optional[Union[str, Path]] = None,
        actions: Optional[Sequence[str]] = None,
        turn_cost: float = 1.0,
        forward_cost: float = 1.0,
        half_turn_cost: Optional[float] = None,
        goal: Optional[Position] = None,
        goal_direction: Optional[Direction] = None,
        initial_state: Optional[GridRobotState] = None,
        heuristic: str = "manhattan",
    ) -> None:
        if obstacles is not None and file is not None:
            raise ValueError("informe obstacles ou file, não ambos")
        if file is not None:
            rows = Path(file).read_text(encoding="utf-8").splitlines()
        elif obstacles is not None:
            rows = list(obstacles)
        else:
            rows = []

        if rows:
            map_width = len(rows[0])
            map_height = len(rows)
            if map_width == 0 or any(len(row) != map_width for row in rows):
                raise ValueError("o mapa deve ser retangular e não vazio")
            if any(cell not in ".# " for row in rows for cell in row):
                raise ValueError("o mapa aceita apenas '.', '#' e espaços")
            if width is not None and width != map_width:
                raise ValueError("width não corresponde à largura do mapa")
            if height is not None and height != map_height:
                raise ValueError("height não corresponde à altura do mapa")
            grid_map: GridMap = tuple(
                row.replace(" ", ".") for row in rows
            )
            width, height = map_width, map_height
        else:
            width = 3 if width is None else width
            height = 3 if height is None else height
            if width < 1 or height < 1:
                raise ValueError("width e height devem ser positivos")
            grid_map = tuple("." * width for _ in range(height))

        if width is None or height is None or width < 1 or height < 1:
            raise ValueError("width e height devem ser positivos")
        if turn_cost < 0 or forward_cost < 0:
            raise ValueError("os custos não podem ser negativos")
        half_cost = 2 * turn_cost if half_turn_cost is None else half_turn_cost
        if half_cost < 0:
            raise ValueError("half_turn_cost não pode ser negativo")
        if heuristic not in ("manhattan", "manhattan_turns"):
            raise ValueError("heuristic deve ser 'manhattan' ou 'manhattan_turns'")
        if goal_direction is not None and goal_direction not in GridRobotAction._DIRECTIONS:
            raise ValueError("goal_direction deve ser uma direção válida")

        enabled_actions = (
            ("GIRAR_HORARIO", "GIRAR_ANTIHORARIO", "FRENTE")
            if actions is None
            else tuple(actions)
        )
        if any(action not in self.ACTION_NAMES for action in enabled_actions):
            raise ValueError(f"actions deve conter apenas {self.ACTION_NAMES}")
        if len(set(enabled_actions)) != len(enabled_actions):
            raise ValueError("actions não pode conter duplicatas")

        goal_position = (width - 1, height - 1) if goal is None else goal
        self._validate_position(goal_position, width, height, grid_map, "goal")
        if initial_state is None:
            initial_state = GridRobotState(0, 0, "NORTE")
        self._validate_position(
            (initial_state.x, initial_state.y), width, height, grid_map, "initial_state"
        )
        if initial_state.direction not in GridRobotAction._DIRECTIONS:
            raise ValueError("initial_state.direction deve ser uma direção válida")

        self.width: int = width
        self.height: int = height
        self.grid_map: GridMap = grid_map
        self.enabled_actions: Tuple[str, ...] = enabled_actions
        self.turn_cost: float = turn_cost
        self.forward_cost: float = forward_cost
        self.half_turn_cost: float = half_cost
        self.goal_position: Position = goal_position
        self.goal_direction: Optional[Direction] = goal_direction
        self.heuristic_name: str = heuristic
        self.initial_state: GridRobotState = initial_state
        goal_heading = "NORTE" if goal_direction is None else goal_direction
        self.goal_state: GridRobotState = GridRobotState(
            goal_position[0], goal_position[1], goal_heading
        )
        action_costs = {
            "GIRAR_HORARIO": turn_cost,
            "GIRAR_ANTIHORARIO": turn_cost,
            "FRENTE": forward_cost,
            "GIRAR_180": half_cost,
        }
        self._actions: List[GridRobotAction] = [
            GridRobotAction(name, grid_map, action_costs[name])
            for name in enabled_actions
        ]

    @staticmethod
    def _validate_position(
        position: Position,
        width: int,
        height: int,
        grid_map: GridMap,
        label: str,
    ) -> None:
        x, y = position
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{label} está fora do grid")
        if grid_map[y][x] == "#":
            raise ValueError(f"{label} está em um obstáculo")

    def actions(self, state: State) -> List[Action]:
        """Retorna apenas ações aplicáveis ao estado atual."""
        return [
            action for action in self._actions if action.apply(state) is not None
        ]

    def is_goal(self, state: State) -> bool:
        """Verifica a posição e, se configurada, a direção objetivo."""
        return (
            isinstance(state, GridRobotState)
            and (state.x, state.y) == self.goal_position
            and (
                self.goal_direction is None
                or state.direction == self.goal_direction
            )
        )

    def heuristic(self, state: State) -> float:
        """Manhattan é admissível; manhattan_turns soma um limite de giros."""
        if not isinstance(state, GridRobotState):
            raise TypeError("state deve ser um GridRobotState")
        distance = (
            abs(state.x - self.goal_position[0])
            + abs(state.y - self.goal_position[1])
        )
        estimate = distance * self.forward_cost
        if self.heuristic_name == "manhattan_turns":
            estimate += self._minimum_turn_cost(state)
        return estimate

    def _minimum_turn_cost(self, state: GridRobotState) -> float:
        required: Set[Direction] = set()
        goal_x, goal_y = self.goal_position
        if goal_x > state.x:
            required.add("LESTE")
        elif goal_x < state.x:
            required.add("OESTE")
        if goal_y > state.y:
            required.add("SUL")
        elif goal_y < state.y:
            required.add("NORTE")

        directions = GridRobotAction._DIRECTIONS
        required_directions = tuple(direction for direction in directions if direction in required)
        required_bits = {
            direction: 1 << index
            for index, direction in enumerate(required_directions)
        }
        initial_mask = required_bits.get(state.direction, 0)
        start = (state.direction, initial_mask)
        distances: Dict[Tuple[Direction, int], float] = {start: 0.0}
        frontier: List[Tuple[float, Direction, int]] = [(0.0, *start)]
        rotation_actions = [
            action for action in self._actions if action.action_name != "FRENTE"
        ]
        full_mask = (1 << len(required_directions)) - 1

        while frontier:
            cost, direction, mask = heapq.heappop(frontier)
            if cost != distances[(direction, mask)]:
                continue
            if (
                mask == full_mask
                and (
                    self.goal_direction is None
                    or direction == self.goal_direction
                )
            ):
                return cost
            for action in rotation_actions:
                turned = action.apply(GridRobotState(state.x, state.y, direction))
                if turned is None or not isinstance(turned[0], GridRobotState):
                    continue
                next_direction = turned[0].direction
                next_mask = mask | required_bits.get(next_direction, 0)
                next_cost = cost + turned[1]
                key = (next_direction, next_mask)
                if next_cost < distances.get(key, float("inf")):
                    distances[key] = next_cost
                    heapq.heappush(frontier, (next_cost, *key))
        return float("inf")

    def random_state(
        self, n_actions: int, avoid_repeats: bool = True, seed: Optional[int] = None
    ) -> GridRobotState:
        """Gera um estado por caminhada aleatória desde o objetivo."""
        if n_actions < 0:
            raise ValueError("n_actions não pode ser negativo")

        generator = random.Random(seed)
        current = self.goal_state
        visited = {current}
        previous_action: Optional[str] = None

        for _ in range(n_actions):
            candidates: List[Tuple[GridRobotAction, GridRobotState]] = []
            for action in self._actions:
                if (
                    avoid_repeats
                    and previous_action is not None
                    and self._INVERSE_ACTIONS.get(previous_action) == action.action_name
                ):
                    continue
                result = action.apply(current)
                if result is None or not isinstance(result[0], GridRobotState):
                    continue
                next_state = result[0]
                if avoid_repeats and next_state in visited:
                    continue
                candidates.append((action, next_state))
            if not candidates:
                raise ValueError(
                    "não há ações sem repetição suficientes para n_actions"
                )
            action, current = generator.choice(candidates)
            previous_action = action.action_name
            visited.add(current)
        return current

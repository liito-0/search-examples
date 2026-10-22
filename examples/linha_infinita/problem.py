"""Busca em um espaço de estados infinito: inteiros conectados em linha."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from search_lib import Action, Problem, State


@dataclass(frozen=True)
class IntegerState(State):
    value: int


class Step(Action):
    def __init__(self, delta: int) -> None:
        if delta not in (-1, 1):
            raise ValueError("delta deve ser -1 ou 1")
        self.delta = delta

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        if not isinstance(state, IntegerState):
            return None
        return IntegerState(state.value + self.delta), 1.0

    def __repr__(self) -> str:
        return "AVANCAR" if self.delta > 0 else "RECUAR"


class InfiniteLine(Problem):
    """Cada inteiro tem dois vizinhos; o conjunto de estados não tem limite."""

    def __init__(self, goal: int = 5, start: int = 0) -> None:
        self.initial_state = IntegerState(start)
        self.goal_state = IntegerState(goal)
        self._actions = (Step(1), Step(-1))

    def actions(self, state: State) -> List[Action]:
        return list(self._actions) if isinstance(state, IntegerState) else []

    def is_goal(self, state: State) -> bool:
        return state == self.goal_state

    def heuristic(self, state: State) -> float:
        if not isinstance(state, IntegerState):
            raise TypeError("state deve ser IntegerState")
        return float(abs(state.value - self.goal_state.value))
"""Testes do cubo de Rubik."""

from collections import Counter

import pytest

from search_lib import BFS

from examples.rubik.problem import RubikCube, RubikCubeAction, RubikCubeState


def apply(problem: RubikCube, state: RubikCubeState, move: str) -> RubikCubeState:
    action = next(action for action in problem.actions(state) if action.move == move)
    result = action.apply(state)
    assert result is not None
    assert isinstance(result[0], RubikCubeState)
    return result[0]


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("move_set", ["quarter", "full"])
def test_quarter_turns_and_inverses(size: int, move_set: str) -> None:
    problem = RubikCube(size, move_set)
    state = problem.initial_state
    for move in ("U", "D", "L", "R", "F", "B"):
        turned = state
        for _ in range(4):
            turned = apply(problem, turned, move)
        assert turned == state
        assert apply(problem, apply(problem, state, move), move + "'") == state


@pytest.mark.parametrize("size", [2, 3])
def test_half_turn_properties(size: int) -> None:
    problem = RubikCube(size, "full")
    state = problem.initial_state
    for face in ("U", "D", "L", "R", "F", "B"):
        half = apply(problem, state, face + "2")
        assert apply(problem, half, face + "2") == state
        assert half == apply(problem, apply(problem, state, face), face)


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("action_set", ["quarter", "half", "full"])
def test_moves_preserve_color_counts(size: int, action_set: str) -> None:
    problem = RubikCube(size, action_set)
    expected = Counter(problem.initial_state.facelets)

    for action in problem.actions(problem.initial_state):
        result = action.apply(problem.initial_state)
        assert result is not None
        assert isinstance(result[0], RubikCubeState)
        assert Counter(result[0].facelets) == expected


def test_half_turn_equals_two_quarter_turns() -> None:
    problem = RubikCube(3, "full")
    for face in ("U", "D", "L", "R", "F", "B"):
        assert apply(problem, problem.initial_state, face + "2") == apply(
            problem,
            apply(problem, problem.initial_state, face),
            face,
        )


@pytest.mark.parametrize(("action_set", "moves"), [("quarter", 4), ("full", 3)])
def test_bfs_solves_short_scrambles(action_set: str, moves: int) -> None:
    problem = RubikCube(2, action_set)
    state = problem.random_state(moves, seed=31)
    result = BFS(problem).search(state)

    assert result.solution_node is not None
    assert result.solution_node.cost <= moves
    replayed = state
    for step in result.solution_node.get_path():
        transition = step.apply(replayed)
        assert transition is not None
        replayed = transition[0]
    assert problem.is_goal(replayed)


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("action_set", ["quarter", "full"])
@pytest.mark.parametrize("moves", [1, 2, 3])
def test_bfs_finds_expected_solution_for_seeded_scrambles(
    size: int, action_set: str, moves: int
) -> None:
    problem = RubikCube(size, action_set)
    state = problem.random_state(moves, seed=31 + moves)
    result = BFS(problem).search(state)

    assert result.solution_node is not None
    assert result.solution_node.cost == moves
    assert result.solution_node.depth == moves

    replayed = state
    for step in result.solution_node.get_path():
        transition = step.apply(replayed)
        assert transition is not None
        replayed = transition[0]
    assert problem.is_goal(replayed)


def test_random_walk_avoids_inverse_moves() -> None:
    problem = RubikCube(3, "full")
    state = problem.random_state(5, seed=13)

    assert len(state.facelets) == 54
    assert problem.heuristic(problem.initial_state) == 0.0


def test_action_repr_is_readable() -> None:
    action = RubikCubeAction("R", tuple(range(24)))

    assert repr(action) == "RubikCubeAction(R)"

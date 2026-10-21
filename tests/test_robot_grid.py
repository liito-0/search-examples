"""Testes do robô em grid."""

from pathlib import Path
from typing import List

import pytest

from search_lib import Action, BFS

from examples.robot_grid.problem import GridRobot, GridRobotAction, GridRobotState


def action(problem: GridRobot, name: str) -> GridRobotAction:
    result = next(item for item in problem.actions(problem.initial_state) if item.action_name == name)
    assert isinstance(result, GridRobotAction)
    return result


def assert_search_solution(
    problem: GridRobot,
    start: GridRobotState,
    expected_cost: float,
    expected_depth: int,
) -> None:
    result = BFS(problem).search(start)

    assert result.solution_node is not None
    assert result.solution_node.cost == expected_cost
    assert result.solution_node.depth == expected_depth
    path: List[Action] = result.solution_node.get_path()
    assert len(path) == expected_depth

    state = start
    total_cost = 0.0
    for step in path:
        transition = step.apply(state)
        assert transition is not None
        state, step_cost = transition
        assert isinstance(state, GridRobotState)
        total_cost += step_cost
    assert problem.is_goal(state)
    assert total_cost == expected_cost


def test_turns_and_forward_transitions() -> None:
    problem = GridRobot(
        width=3,
        height=3,
        initial_state=GridRobotState(1, 1, "NORTE"),
        goal=(2, 2),
    )
    original = problem.initial_state

    state = original
    for _ in range(4):
        result = action(problem, "GIRAR_HORARIO").apply(state)
        assert result is not None
        assert isinstance(result[0], GridRobotState)
        state, cost = result
        assert cost == 1.0
    assert state == original

    clockwise = action(problem, "GIRAR_HORARIO").apply(original)
    assert clockwise is not None
    counterclockwise = action(problem, "GIRAR_ANTIHORARIO").apply(clockwise[0])
    assert counterclockwise == (original, 1.0)

    forward = action(problem, "FRENTE").apply(original)
    assert forward == (GridRobotState(1, 0, "NORTE"), 1.0)


def test_forward_is_invalid_at_obstacles_and_grid_edges() -> None:
    obstacle_problem = GridRobot(
        obstacles=[".#.", "..."],
        initial_state=GridRobotState(0, 0, "LESTE"),
        goal=(2, 1),
    )
    edge_problem = GridRobot(
        width=3,
        height=2,
        initial_state=GridRobotState(0, 0, "NORTE"),
        goal=(2, 1),
    )

    obstacle_front = GridRobotAction(
        "FRENTE", obstacle_problem.grid_map, obstacle_problem.forward_cost
    )
    edge_front = GridRobotAction(
        "FRENTE", edge_problem.grid_map, edge_problem.forward_cost
    )

    assert obstacle_front.apply(obstacle_problem.initial_state) is None
    assert edge_front.apply(edge_problem.initial_state) is None


def test_map_loading_from_strings_and_file(tmp_path: Path) -> None:
    rows = ["..#.", "...."]
    from_strings = GridRobot(obstacles=rows, goal=(3, 1))
    map_file = tmp_path / "mapa.txt"
    map_file.write_text("\n".join(rows), encoding="utf-8")
    from_file = GridRobot(file=map_file, goal=(3, 1))

    assert from_strings.grid_map == tuple(rows)
    assert from_file.grid_map == tuple(rows)
    assert (from_file.width, from_file.height) == (4, 2)


@pytest.mark.parametrize("heuristic", ["manhattan", "manhattan_turns"])
def test_open_grid_search_has_known_optimal_cost(heuristic: str) -> None:
    start = GridRobotState(0, 0, "LESTE")
    problem = GridRobot(
        width=3,
        height=3,
        initial_state=start,
        goal=(2, 0),
        goal_direction="LESTE",
        heuristic=heuristic,
    )

    assert problem.heuristic(start) <= 2.0
    assert_search_solution(problem, start, expected_cost=2.0, expected_depth=2)


def test_search_routes_around_a_wall() -> None:
    start = GridRobotState(0, 0, "LESTE")
    problem = GridRobot(
        obstacles=[".#.", "..."],
        initial_state=start,
        goal=(2, 0),
        heuristic="manhattan_turns",
    )

    assert problem.heuristic(start) <= 7.0
    assert_search_solution(problem, start, expected_cost=7.0, expected_depth=7)


def test_high_turn_cost_has_known_optimum() -> None:
    start = GridRobotState(0, 0, "NORTE")
    problem = GridRobot(
        width=3,
        height=3,
        initial_state=start,
        goal=(1, 0),
        goal_direction="LESTE",
        turn_cost=5.0,
        forward_cost=1.0,
        heuristic="manhattan_turns",
    )

    assert problem.heuristic(start) <= 6.0
    assert_search_solution(problem, start, expected_cost=6.0, expected_depth=2)


def test_bfs_returns_no_solution_for_a_dead_end() -> None:
    problem = GridRobot(
        width=2,
        height=2,
        actions=("FRENTE",),
        initial_state=GridRobotState(0, 0, "NORTE"),
        goal=(0, 1),
    )

    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is None


def test_half_turn_action_and_random_walk_are_deterministic() -> None:
    problem = GridRobot(
        width=3,
        height=3,
        actions=("GIRAR_HORARIO", "GIRAR_ANTIHORARIO", "FRENTE", "GIRAR_180"),
        goal=(2, 2),
    )
    original = problem.initial_state
    half_turn = action(problem, "GIRAR_180").apply(original)

    assert half_turn == (GridRobotState(0, 0, "SUL"), 2.0)
    assert problem.random_state(3, seed=19) == problem.random_state(3, seed=19)

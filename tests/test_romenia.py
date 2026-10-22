"""Testes do mapa da Romênia."""

import heapq
from typing import Dict, List

import pytest

from search_lib import Action, BFS, DLS

from examples.mapa_romenia.problem import (
    ROMANIA_ROADS,
    STRAIGHT_LINE_TO_BUCHAREST,
    Romania,
    RomaniaAction,
    RomaniaState,
)


def shortest_costs(problem: Romania, source: str) -> Dict[str, float]:
    """Oráculo independente: Dijkstra sobre a lista de estradas."""
    best = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        cost, city = heapq.heappop(queue)
        if cost > best[city]:
            continue
        for neighbor, distance in problem.neighbors[city].items():
            if cost + distance < best.get(neighbor, float("inf")):
                best[neighbor] = cost + distance
                heapq.heappush(queue, (cost + distance, neighbor))
    return best


def replay(problem: Romania, start: RomaniaState, path: List[Action]) -> float:
    """Reexecuta a lista de ações; confere que chega ao objetivo e devolve o custo."""
    state = start
    total = 0.0
    for step in path:
        transition = step.apply(state)
        assert transition is not None
        state, cost = transition
        assert isinstance(state, RomaniaState)
        total += cost
    assert problem.is_goal(state)
    return total


def test_map_matches_the_book() -> None:
    problem = Romania()

    assert len(problem.cities) == 20
    assert len(ROMANIA_ROADS) == 23
    assert set(problem.cities) == set(STRAIGHT_LINE_TO_BUCHAREST)
    for city, roads in problem.neighbors.items():
        for neighbor, distance in roads.items():
            assert problem.neighbors[neighbor][city] == distance
    assert problem.neighbors["Arad"] == {"Zerind": 75, "Sibiu": 140, "Timisoara": 118}
    assert problem.initial_state == RomaniaState("Arad")
    assert problem.goal_state == RomaniaState("Bucharest")


def test_actions_list_only_neighbors() -> None:
    problem = Romania()

    assert [repr(a) for a in problem.actions(RomaniaState("Arad"))] == [
        "IR_PARA(Sibiu)",
        "IR_PARA(Timisoara)",
        "IR_PARA(Zerind)",
    ]
    assert [repr(a) for a in problem.actions(RomaniaState("Neamt"))] == ["IR_PARA(Iasi)"]


def test_action_apply_returns_new_state_and_distance() -> None:
    problem = Romania()
    go_to_sibiu = RomaniaAction("Sibiu", problem.neighbors)

    assert go_to_sibiu.apply(RomaniaState("Arad")) == (RomaniaState("Sibiu"), 140.0)
    assert go_to_sibiu.apply(RomaniaState("Fagaras")) == (RomaniaState("Sibiu"), 99.0)
    assert go_to_sibiu.apply(RomaniaState("Bucharest")) is None  # sem estrada direta


def test_goal_test() -> None:
    problem = Romania()

    assert problem.is_goal(RomaniaState("Bucharest"))
    assert not problem.is_goal(RomaniaState("Arad"))
    assert Romania(goal="Neamt").is_goal(RomaniaState("Neamt"))


def test_straight_line_heuristic_values() -> None:
    problem = Romania()

    assert problem.heuristic(RomaniaState("Arad")) == 366
    assert problem.heuristic(RomaniaState("Bucharest")) == 0
    assert problem.heuristic(RomaniaState("Pitesti")) == 100


def test_straight_line_heuristic_is_admissible_and_consistent() -> None:
    problem = Romania()
    true_cost = shortest_costs(problem, "Bucharest")  # estradas são simétricas

    for city in problem.cities:
        assert problem.heuristic(RomaniaState(city)) <= true_cost[city]
        for neighbor, distance in problem.neighbors[city].items():
            h_city = problem.heuristic(RomaniaState(city))
            assert h_city <= distance + problem.heuristic(RomaniaState(neighbor))


def test_bfs_finds_fewest_roads_not_cheapest_route() -> None:
    problem = Romania()
    result = BFS(problem).search(problem.initial_state)

    assert result.solution_node is not None
    path = result.solution_node.get_path()
    assert [repr(a) for a in path] == [
        "IR_PARA(Sibiu)",
        "IR_PARA(Fagaras)",
        "IR_PARA(Bucharest)",
    ]
    assert result.solution_node.depth == 3
    assert result.solution_node.cost == 450
    assert replay(problem, problem.initial_state, path) == 450
    # O ótimo do livro (Sibiu, Rimnicu Vilcea, Pitesti, Bucharest) custa 418 e usa 4 estradas.
    assert shortest_costs(problem, "Arad")["Bucharest"] == 418


def test_bfs_from_goal_returns_root() -> None:
    problem = Romania()
    result = BFS(problem).search(RomaniaState("Bucharest"))

    assert result.solution_node is not None
    assert result.solution_node.depth == 0
    assert result.solution_node.cost == 0
    assert result.metrics.nodes_generated == 1


@pytest.mark.parametrize("limit, solvable", [(2, False), (3, True), (4, True)])
def test_depth_limited_search_respects_the_limit(limit: int, solvable: bool) -> None:
    problem = Romania()
    result = DLS(problem, limit).search(problem.initial_state)

    if not solvable:
        assert result.solution_node is None
        return
    assert result.solution_node is not None
    assert result.solution_node.depth <= limit
    cost = replay(problem, problem.initial_state, result.solution_node.get_path())
    assert cost == result.solution_node.cost


def test_bfs_works_for_every_start_city() -> None:
    problem = Romania()
    for city in problem.cities:
        start = RomaniaState(city)
        result = BFS(problem).search(start)
        assert result.solution_node is not None
        replay(problem, start, result.solution_node.get_path())


def test_custom_goal_and_map() -> None:
    problem = Romania(goal="Neamt", initial_city="Arad")
    result = BFS(problem).search(problem.initial_state)

    assert problem.heuristic_name == "zero"
    assert result.solution_node is not None
    assert result.solution_node.depth == 7  # Arad, Sibiu, Fagaras, Bucharest, Urziceni, Vaslui, Iasi, Neamt


def test_random_state_is_deterministic_and_reaches_goal() -> None:
    problem = Romania()

    assert problem.random_state(0) == problem.goal_state
    assert problem.random_state(4, seed=7) == problem.random_state(4, seed=7)
    for seed in range(20):
        state = problem.random_state(5, seed=seed)
        result = BFS(problem).search(state)
        assert result.solution_node is not None
        assert result.solution_node.depth <= 5


def test_random_state_avoids_repeats_until_stuck() -> None:
    # Em uma cadeia A-B-C-D-E, sem repetições a distância ao objetivo A é exatamente n.
    chain = Romania(
        roads=[("A", "B", 1), ("B", "C", 1), ("C", "D", 1), ("D", "E", 1)],
        goal="A",
        initial_city="E",
    )
    assert chain.random_state(4, seed=1) == RomaniaState("E")
    # Em E não há saída sem repetição: o passo seguinte volta para D.
    assert chain.random_state(5, seed=1) == RomaniaState("D")
    # Em um problema real nunca falha, mesmo passando por cidades sem saída.
    problem = Romania()
    for seed in range(50):
        assert problem.random_state(12, seed=seed).city in problem.cities


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        Romania(goal="Paris")
    with pytest.raises(ValueError):
        Romania(initial_city="Paris")
    with pytest.raises(ValueError):
        Romania(heuristic="manhattan")
    with pytest.raises(ValueError):
        Romania(goal="Arad", heuristic="straight_line")
    with pytest.raises(ValueError):
        Romania(roads=[("A", "B", 1)], goal="B", heuristic="straight_line")
    with pytest.raises(ValueError):
        Romania(roads=[("A", "B", -1)], goal="B", initial_city="A")
    with pytest.raises(ValueError):
        Romania(roads=[("A", "A", 1)], goal="A", initial_city="A")
    with pytest.raises(ValueError):
        Romania(roads=[("A", "B", 1), ("B", "A", 1)], goal="B", initial_city="A")
    with pytest.raises(ValueError):
        Romania().random_state(-1)
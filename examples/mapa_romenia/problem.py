"""Problema de caminhos no mapa da Romênia (exemplo do livro Russell & Norvig)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from search_lib import Action, Problem, State  # type: ignore[import-untyped]

# Uma estrada é (cidade A, cidade B, distância em km).
Road = Tuple[str, str, float]


# Dados do mapa
# Estradas do mapa. Cada tupla vale nos dois sentidos:
ROMANIA_ROADS: Tuple[Road, ...] = (
    ("Arad", "Zerind", 75),
    ("Arad", "Sibiu", 140),
    ("Arad", "Timisoara", 118),
    ("Zerind", "Oradea", 71),
    ("Oradea", "Sibiu", 151),
    ("Timisoara", "Lugoj", 111),
    ("Lugoj", "Mehadia", 70),
    ("Mehadia", "Drobeta", 75),
    ("Drobeta", "Craiova", 120),
    ("Craiova", "Rimnicu Vilcea", 146),
    ("Craiova", "Pitesti", 138),
    ("Sibiu", "Fagaras", 99),
    ("Sibiu", "Rimnicu Vilcea", 80),
    ("Rimnicu Vilcea", "Pitesti", 97),
    ("Fagaras", "Bucharest", 211),
    ("Pitesti", "Bucharest", 101),
    ("Bucharest", "Giurgiu", 90),
    ("Bucharest", "Urziceni", 85),
    ("Urziceni", "Hirsova", 98),
    ("Urziceni", "Vaslui", 142),
    ("Hirsova", "Eforie", 86),
    ("Vaslui", "Iasi", 92),
    ("Iasi", "Neamt", 87),
)

# Distância em linha reta (em km) de cada cidade até Bucharest
# É o valor da heurística h(n) usada por buscas informadas 
STRAIGHT_LINE_TO_BUCHAREST: Dict[str, float] = {
    "Arad": 366,
    "Bucharest": 0,
    "Craiova": 160,
    "Drobeta": 242,
    "Eforie": 161,
    "Fagaras": 176,
    "Giurgiu": 77,
    "Hirsova": 151,
    "Iasi": 226,
    "Lugoj": 244,
    "Mehadia": 241,
    "Neamt": 234,
    "Oradea": 380,
    "Pitesti": 100,
    "Rimnicu Vilcea": 193,
    "Sibiu": 253,
    "Timisoara": 329,
    "Urziceni": 80,
    "Vaslui": 199,
    "Zerind": 374,
}

# Heurísticas que o problema sabe calcular.
HEURISTICS: Tuple[str, ...] = ("straight_line", "zero")


# Estado
@dataclass(frozen=True)
class RomaniaState(State):
    """Estado = a cidade onde o viajante está.

    `frozen=True` deixa o estado imutável e hashável, o que as buscas exigem
    (por exemplo, para guardar estados em conjuntos).
    """

    city: str


# Ação
class RomaniaAction(Action):
    """Ação parametrizada IR_PARA(destino): dirige até uma cidade vizinha.

    A ação guarda só o destino. A cidade de origem vem do estado em que ela é
    aplicada, então a mesma ação IR_PARA(Sibiu) serve para sair de Arad, de
    Fagaras, etc.
    """

    def __init__(self, destination: str, neighbors: Dict[str, Dict[str, float]]) -> None:
        self.destination: str = destination
        # Tabela de vizinhança compartilhada com o problema:
        # neighbors[cidade][vizinha] = distância.
        self._neighbors: Dict[str, Dict[str, float]] = neighbors

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        """Retorna (novo estado, custo) ou None se não houver estrada direta."""
        if not isinstance(state, RomaniaState):
            return None
        # Procura a distância da cidade atual até o destino; None se não forem vizinhas.
        distance = self._neighbors.get(state.city, {}).get(self.destination)
        if distance is None:
            return None
        return RomaniaState(self.destination), distance

    def __repr__(self) -> str:
        # Facilita imprimir o caminho encontrado: IR_PARA(Sibiu), IR_PARA(Fagaras), ...
        return f"IR_PARA({self.destination})"



# Problema
class Romania(Problem):
    """Mapa de estradas com objetivo e heurística configuráveis.

    - `roads`: lista de `(cidade_a, cidade_b, distância)` (padrão: mapa do livro).
    - `goal`: cidade objetivo (padrão: `"Bucharest"`).
    - `heuristic`: `"straight_line"` (padrão; só existe para o mapa do livro com
      objetivo `"Bucharest"`) ou `"zero"`.
    - `initial_city`: cidade de partida (padrão: `"Arad"`).
    """

    def __init__(
        self,
        roads: Optional[Sequence[Road]] = None,
        goal: str = "Bucharest",
        heuristic: Optional[str] = None,
        initial_city: str = "Arad",
    ) -> None:
        # Se o usuário não passou `roads`, usamos o mapa do livro.
        default_map = roads is None
        road_list: Tuple[Road, ...] = ROMANIA_ROADS if roads is None else tuple(roads)

        # Monta a tabela de vizinhança nos dois sentidos, validando cada estrada.
        neighbors: Dict[str, Dict[str, float]] = {}
        for city_a, city_b, distance in road_list:
            if city_a == city_b:
                raise ValueError(f"estrada de {city_a} para ela mesma")
            if distance <= 0:
                raise ValueError(f"distância inválida entre {city_a} e {city_b}")
            if city_b in neighbors.get(city_a, {}):
                raise ValueError(f"estrada duplicada entre {city_a} e {city_b}")
            neighbors.setdefault(city_a, {})[city_b] = float(distance)
            neighbors.setdefault(city_b, {})[city_a] = float(distance)

        # Objetivo e partida precisam ser cidades que existem no mapa.
        if goal not in neighbors:
            raise ValueError(f"goal desconhecido: {goal!r}")
        if initial_city not in neighbors:
            raise ValueError(f"initial_city desconhecida: {initial_city!r}")

        # A tabela de linha reta só vale para o mapa do livro e objetivo Bucharest.
        # Sem heurística informada, o padrão vira "zero" (busca sem informação).
        can_use_straight_line = default_map and goal == "Bucharest"
        if heuristic is None:
            heuristic = "straight_line" if can_use_straight_line else "zero"
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic deve ser um de {HEURISTICS}")
        if heuristic == "straight_line" and not can_use_straight_line:
            raise ValueError(
                "'straight_line' só está disponível para o mapa do livro com goal='Bucharest'"
            )

        self.neighbors: Dict[str, Dict[str, float]] = neighbors
        self.cities: Tuple[str, ...] = tuple(sorted(neighbors))
        self.goal_city: str = goal
        self.goal_state: RomaniaState = RomaniaState(goal)
        self.initial_state: RomaniaState = RomaniaState(initial_city)
        self.heuristic_name: str = heuristic
        # Uma ação IR_PARA por cidade, criada uma única vez e reaproveitada.
        self._actions: Dict[str, RomaniaAction] = {
            city: RomaniaAction(city, neighbors) for city in self.cities
        }

    def actions(self, state: State) -> List[Action]:
        """Uma ação IR_PARA para cada cidade vizinha (em ordem alfabética).

        A ordem fixa torna as buscas determinísticas e reproduzíveis.
        """
        if not isinstance(state, RomaniaState):
            return []
        return [self._actions[city] for city in sorted(self.neighbors.get(state.city, {}))]

    def is_goal(self, state: State) -> bool:
        """Teste de objetivo: o viajante chegou na cidade objetivo?"""
        return isinstance(state, RomaniaState) and state.city == self.goal_city

    def heuristic(self, state: State) -> float:
        """Estimativa do custo até o objetivo.

        `straight_line`: distância em linha reta até Bucharest, que nunca
        superestima o custo real (admissível) e é consistente. `zero`: sem informação.
        """
        if not isinstance(state, RomaniaState):
            raise TypeError("state deve ser um RomaniaState")
        if self.heuristic_name == "zero":
            return 0.0
        return STRAIGHT_LINE_TO_BUCHAREST[state.city]

    def random_state(
        self, n_actions: int, avoid_repeats: bool = True, seed: Optional[int] = None
    ) -> RomaniaState:
        """Gera um estado por caminhada aleatória de `n_actions` passos desde o objetivo.

        Como as estradas valem nos dois sentidos, o estado gerado sempre consegue
        chegar ao objetivo em no máximo `n_actions` passos. Com `avoid_repeats`,
        tenta não visitar a mesma cidade duas vezes; se chegar a uma cidade sem
        saída (ex.: Giurgiu), permite voltar por onde veio.
        """
        if n_actions < 0:
            raise ValueError("n_actions não pode ser negativo")

        # `seed` torna o resultado reproduzível.
        generator = random.Random(seed)
        current = self.goal_state
        visited = {current}
        for _ in range(n_actions):
            # Vizinhas possíveis, descartando as já visitadas se avoid_repeats.
            candidates = [
                RomaniaState(city)
                for city in sorted(self.neighbors[current.city])
                if not (avoid_repeats and RomaniaState(city) in visited)
            ]
            # Sem saída nova: aceita qualquer vizinha (repetindo cidade).
            if not candidates:
                candidates = [RomaniaState(city) for city in sorted(self.neighbors[current.city])]
            current = generator.choice(candidates)
            visited.add(current)
        return current
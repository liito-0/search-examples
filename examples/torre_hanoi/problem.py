"""Problema da Torre de Hanói com número de discos e de pinos configurável."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from search_lib import Action, Problem, State  

# Heurísticas que o problema sabe calcular.
HEURISTICS: Tuple[str, ...] = ("disks_off_goal", "zero")


# Estado
@dataclass(frozen=True)
class HanoiState(State):
    """Estado = o conteúdo de cada pino.

    `pegs[i]` é uma tupla com os discos do pino `i`, do fundo para o topo.
    Os discos são numerados pelo tamanho: 1 é o menor. Exemplo com 3 discos
    todos no primeiro pino: `((3, 2, 1), (), ())`.

    Tuplas deixam o estado imutável e hashável, como as buscas exigem.
    """

    pegs: Tuple[Tuple[int, ...], ...]

    def __str__(self) -> str:
        return " | ".join(" ".join(map(str, peg)) or "-" for peg in self.pegs)


# Ação
class HanoiAction(Action):
    """Ação parametrizada MOVER(origem, destino): move o disco do topo de um pino para outro."""

    def __init__(self, source: int, target: int) -> None:
        self.source: int = source
        self.target: int = target

    def apply(self, state: State) -> Optional[Tuple[State, float]]:
        """Retorna (novo estado, custo 1) ou None se o movimento for ilegal."""
        if not isinstance(state, HanoiState):
            return None
        pegs = state.pegs
        if self.source == self.target:
            return None
        if not (0 <= self.source < len(pegs) and 0 <= self.target < len(pegs)):
            return None
        # Não há disco para mover se o pino de origem está vazio.
        if not pegs[self.source]:
            return None
        disk = pegs[self.source][-1]
        # Regra do jogo: nunca colocar um disco maior sobre um menor.
        if pegs[self.target] and pegs[self.target][-1] < disk:
            return None

        new_pegs = list(pegs)
        new_pegs[self.source] = pegs[self.source][:-1]
        new_pegs[self.target] = pegs[self.target] + (disk,)
        return HanoiState(tuple(new_pegs)), 1.0

    def __repr__(self) -> str:
        return f"MOVER({self.source}, {self.target})"


# Problema
class TowerOfHanoi(Problem):
    """Torre de Hanói parametrizável.

    - `n_disks`: quantidade de discos (padrão: 3).
    - `n_pegs`: quantidade de pinos, no mínimo 3 (padrão: 3).
    - `initial_peg`: pino onde todos os discos começam (padrão: 0, o primeiro).
    - `goal_peg`: pino onde todos os discos devem terminar (padrão: o último).
    - `heuristic`: `"disks_off_goal"` (padrão) ou `"zero"`.

    Com 3 pinos a solução ótima tem `2**n_disks - 1` movimentos.
    """

    def __init__(
        self,
        n_disks: int = 3,
        n_pegs: int = 3,
        initial_peg: int = 0,
        goal_peg: Optional[int] = None,
        heuristic: str = "disks_off_goal",
    ) -> None:
        if n_disks < 1:
            raise ValueError("n_disks deve ser positivo")
        if n_pegs < 3:
            raise ValueError("n_pegs deve ser pelo menos 3")
        goal_peg = n_pegs - 1 if goal_peg is None else goal_peg
        if not 0 <= initial_peg < n_pegs:
            raise ValueError("initial_peg está fora dos pinos")
        if not 0 <= goal_peg < n_pegs:
            raise ValueError("goal_peg está fora dos pinos")
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic deve ser um de {HEURISTICS}")

        self.n_disks: int = n_disks
        self.n_pegs: int = n_pegs
        self.goal_peg: int = goal_peg
        self.heuristic_name: str = heuristic
        self.initial_state: HanoiState = self._stacked_state(initial_peg)
        self.goal_state: HanoiState = self._stacked_state(goal_peg)
        # Uma ação MOVER para cada par (origem, destino) de pinos diferentes,
        # criadas uma única vez e reaproveitadas.
        self._actions: Dict[Tuple[int, int], HanoiAction] = {
            (source, target): HanoiAction(source, target)
            for source in range(n_pegs)
            for target in range(n_pegs)
            if source != target
        }

    def _stacked_state(self, peg: int) -> HanoiState:
        """Estado com todos os discos empilhados em `peg` (maior no fundo)."""
        tower = tuple(range(self.n_disks, 0, -1))
        return HanoiState(
            tuple(tower if index == peg else () for index in range(self.n_pegs))
        )

    def actions(self, state: State) -> List[Action]:
        """Apenas os movimentos legais no estado, em ordem de (origem, destino).
        A ordem fixa torna as buscas determinísticas e reproduzíveis.
        """
        return [
            action
            for action in self._actions.values()
            if action.apply(state) is not None
        ]

    def is_goal(self, state: State) -> bool:
        """Teste de objetivo: todos os discos estão empilhados no pino objetivo?"""
        return state == self.goal_state

    def heuristic(self, state: State) -> float:
        """Estimativa do número de movimentos até o objetivo.

        `disks_off_goal`: quantidade de discos fora do pino objetivo. Cada um
        precisa de pelo menos um movimento, então nunca superestima (admissível)
        e um movimento muda esse valor em no máximo 1 (consistente). `zero`: sem
        informação.
        """
        if not isinstance(state, HanoiState):
            raise TypeError("state deve ser um HanoiState")
        if self.heuristic_name == "zero":
            return 0.0
        return float(sum(len(peg) for index, peg in enumerate(state.pegs) if index != self.goal_peg))

    def random_state(
        self, n_actions: int, avoid_repeats: bool = True, seed: Optional[int] = None
    ) -> HanoiState:
        """Gera um estado por caminhada aleatória de `n_actions` movimentos desde o objetivo.

        Todo movimento tem um movimento inverso, então o estado gerado sempre
        consegue chegar ao objetivo em no máximo `n_actions` movimentos. Com
        `avoid_repeats`, tenta não passar duas vezes pelo mesmo estado; se não
        houver saída nova, permite repetir.
        """
        if n_actions < 0:
            raise ValueError("n_actions não pode ser negativo")

        # `seed` torna o resultado reproduzível.
        generator = random.Random(seed)
        current = self.goal_state
        visited = {current}
        for _ in range(n_actions):
            # Estados alcançáveis em um movimento, descartando os já visitados se avoid_repeats.
            next_states = [
                result[0]
                for action in self.actions(current)
                for result in [action.apply(current)]
                if result is not None and isinstance(result[0], HanoiState)
            ]
            candidates = [s for s in next_states if not (avoid_repeats and s in visited)]
            # Sem saída nova: aceita qualquer movimento (repetindo estado).
            current = generator.choice(candidates or next_states)
            visited.add(current)
        return current
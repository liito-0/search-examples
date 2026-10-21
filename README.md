# search-examples
Problemas de exemplo para search_lib (IA)

## Problemas implementados

### Sliding puzzle

`examples.sliding_puzzle.problem.SlidingPuzzle` aceita `rows` e `cols` (ambos
com padrão `3`) e `heuristic` (`"manhattan"` ou `"misplaced"`, padrão
`"manhattan"`). O estado é uma tupla plana; `0` representa o espaço vazio e o
objetivo ordena as peças de `1` a `n - 1`, seguido de `0`.

### Cubo de Rubik

`examples.rubik.problem.RubikCube` aceita `size` (`2` ou `3`, padrão `3`) e
`action_set` (`"quarter"`, `"half"` ou `"full"`, padrão `"quarter"`). Os
facelets são armazenados em uma tupla de cores, e os movimentos são permutações
pré-calculadas.

### Testes

Execute os testes com `pytest` e verifique os tipos dos exemplos com
`mypy examples`.

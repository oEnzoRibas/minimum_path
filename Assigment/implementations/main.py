import os
import csv
import math
import time
import heapq
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt


# =========================================================
# General Settings
# =========================================================

random.seed(42)

SPARSE_SIZES = [100, 500, 1000, 5000, 10000]
DENSE_SIZES = [100, 500, 1000]

K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]

PESO_MIN = 1
PESO_MAX = 100

RESULTS_CSV = "results/tables/results.csv"


# =========================================================
# CRIA PASTAS
# =========================================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/grafics", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

Graph = Dict[
    int, 
    List[
        Tuple[int, int]
        ]
    ]


# =========================================================
# GERAÇÃO DE GraphS
# =========================================================

"""
Adds directed edge without repetition.
used: set of used edges (u, v)
"""
def add_edge(
    graph: Graph,
    used: Set[Tuple[int, int]],
    u: int,
    v: int,
    w: int
):
    """
    Adiciona aresta direcionada sem repetição.
    """

    if u == v:
        return False

    if (u, v) in used:
        return False

    graph[u].append((v, w))

    used.add((u, v))

    return True



def generate_sparse_graph(n: int, multiplier: int) -> Graph:
    """
    Gera Graph sparse:

    e = multiplier * n

    Estratégia:
    - cria árvore inicial
    - adiciona arestas extras
    """

    edges_target = multiplier * n

    graph = {i: [] for i in range(n)}

    used = set()

    # árvore inicial
    for v in range(1, n):

        u = random.randint(0, v - 1)

        w = random.randint(PESO_MIN, PESO_MAX)

        add_edge(graph, used, u, v, w)

    while sum(len(graph[u]) for u in graph) < edges_target:

        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)

        w = random.randint(PESO_MIN, PESO_MAX)

        add_edge(graph, used, u, v, w)

    return graph



def generate_dense_graph(n: int, divisor: int) -> Graph:
    """
    Gera Graph dense:

    e = n(n-1)/divisor
    """

    edges_target = (n * (n - 1)) // divisor

    graph = {i: [] for i in range(n)}

    used = set()

    while sum(len(graph[u]) for u in graph) < edges_target:

        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)

        w = random.randint(PESO_MIN, PESO_MAX)

        add_edge(graph, used, u, v, w)

    return graph



def save_graph(filename: str, graph: Graph, source: int = 0):
    """
    Salva Graph em formato txt.
    """

    n = len(graph)

    e = sum(len(graph[u]) for u in graph)

    with open(filename, "w") as f:

        f.write(f"{n} {e}\n")

        for u in graph:
            for v, w in graph[u]:
                f.write(f"{u} {v} {w}\n")

        f.write(f"{source}\n")



def generate_all_graphs():
    """
    Gera todos os Graphs dos experimentos.
    """

    print("\nGerando Graphs sparses...")

    for n in SPARSE_SIZES:

        for mult in [2, 3, 4]:

            graph = generate_sparse_graph(n, mult)

            filename = f"Graphs/sparse_n{n}_e{mult}n.txt"

            save_graph(filename, graph)

    print("\nGerando Graphs denses...")

    for n in DENSE_SIZES:

        for divisor in [4, 2]:

            graph = generate_dense_graph(n, divisor)

            filename = f"Graphs/dense_n{n}_div{divisor}.txt"

            save_graph(filename, graph)


# =========================================================
# DIJKSTRA
# =========================================================


def dijkstra(
    graph: Graph,
    source: int
):
    """
    Dijkstra utilizando heap binário (heapq).

    Complexidade:
    O((V + E) log V)
    """

    start = time.perf_counter()

    dist = {v: math.inf for v in graph}

    pred = {v: None for v in graph}

    dist[source] = 0

    heap = [(0, source)]

    while heap:

        d_u, u = heapq.heappop(heap)

        if d_u > dist[u]:
            continue

        for v, w in graph[u]:

            nd = dist[u] + w

            if nd < dist[v]:

                dist[v] = nd

                pred[v] = u

                heapq.heappush(heap, (nd, v))

    elapsed = time.perf_counter() - start

    return dist, pred, elapsed


# =========================================================
# BELLMAN-FORD
# =========================================================


def bellman_ford(
    graph: Graph,
    source: int
):
    """
    Bellman-Ford.

    Complexidade:
    O(VE)
    """

    start = time.perf_counter()

    dist = {v: math.inf for v in graph}

    pred = {v: None for v in graph}

    dist[source] = 0

    vertices = list(graph.keys())

    n = len(vertices)

    for _ in range(n - 1):

        updated = False

        for u in graph:

            for v, w in graph[u]:

                if dist[u] != math.inf and dist[u] + w < dist[v]:

                    dist[v] = dist[u] + w

                    pred[v] = u

                    updated = True

        if not updated:
            break

    negative_cycle = False

    for u in graph:

        for v, w in graph[u]:

            if dist[u] != math.inf and dist[u] + w < dist[v]:

                negative_cycle = True

    elapsed = time.perf_counter() - start

    return dist, pred, elapsed, negative_cycle


# =========================================================
# DIJKSTRA PARCIAL
# =========================================================


def partial_dijkstra(
    graph: Graph,
    source: int,
    limit: int
):
    """
    Executa Dijkstra parcial até processar sqrt(n) vértices.

    Retorna:
    - distâncias
    - predecessores
    - conjunto processado U_hat
    - fronteira S
    - valor B
    """

    dist = {v: math.inf for v in graph}

    pred = {v: None for v in graph}

    dist[source] = 0

    heap = [(0, source)]

    processed = set()

    last_distance = 0

    while heap and len(processed) < limit:

        d_u, u = heapq.heappop(heap)

        if u in processed:
            continue

        processed.add(u)

        last_distance = d_u

        for v, w in graph[u]:

            nd = dist[u] + w

            if nd < dist[v]:

                dist[v] = nd

                pred[v] = u

                heapq.heappush(heap, (nd, v))

    frontier = set()

    for u in processed:

        for v, w in graph[u]:

            if v not in processed:
                frontier.add(v)

    B = last_distance

    return dist, pred, processed, frontier, B


# =========================================================
# FIND PIVOTS
# =========================================================

# array global de distâncias estimadas

d_hat = {}

pred_global = {}



def find_pivots(
    B: float,
    S: List[int],
    graph: Graph,
    k: int
):
    """
    Implementação do Algoritmo 1 (FindPivots).

    Mantém:
    - array global d_hat
    - floresta F
    - conjunto W
    - pivôs P
    """

    W = set(S)

    W_prev = set(S)

    F = defaultdict(set)

    start = time.perf_counter()

    # exatamente k iterações
    for _ in range(k):

        W_i = set()

        for u in W_prev:

            for v, peso in graph[u]:

                if d_hat[u] + peso <= d_hat[v]:

                    d_hat[v] = d_hat[u] + peso

                    pred_global[v] = u

                    # floresta F
                    F[u].add(v)

                    if d_hat[v] < B:
                        W_i.add(v)

        W.update(W_i)

        # parada antecipada
        if len(W) > k * len(S):

            elapsed = time.perf_counter() - start

            return set(S), W, F, elapsed

        W_prev = W_i

    P = set()

    def subtree_size(root):

        visited = set()

        stack = [root]

        while stack:

            u = stack.pop()

            if u in visited:
                continue

            visited.add(u)

            for nxt in F[u]:
                stack.append(nxt)

        return len(visited)

    # raízes com >= k vértices
    for s in S:

        if subtree_size(s) >= k:
            P.add(s)

    elapsed = time.perf_counter() - start

    return P, W, F, elapsed


# =========================================================
# COBERTURA
# =========================================================


def compute_coverage(U_hat, W, P, pred):
    """
    Verifica cobertura do Lema 3.2.
    """

    covered = 0

    for x in U_hat:

        if x in W:

            covered += 1

            continue

        current = x

        while current is not None:

            if current in P:

                covered += 1

                break

            current = pred[current]

    return 100 * covered / max(1, len(U_hat))


# =========================================================
# DIJKSTRA COM PRÉ-PROCESSAMENTO
# =========================================================


def dijkstra_with_pivots(
    graph: Graph,
    source: int,
    k: int,
    partial_size: int
):
    """
    Dijkstra utilizando FindPivots como pré-processamento.
    """

    dist_partial, pred_partial, U_hat, S, B = partial_dijkstra(
        graph,
        source,
        partial_size
    )

    for v in graph:
        d_hat[v] = dist_partial[v]

    P, W, F, fp_time = find_pivots(
        B,
        list(S),
        graph,
        k
    )

    combined_dist = {v: math.inf for v in graph}

    combined_pred = {v: None for v in graph}

    for v in dist_partial:
        combined_dist[v] = dist_partial[v]

    for p in P:

        dist_p, pred_p, _ = dijkstra(graph, p)

        for v in graph:

            if dist_partial[p] == math.inf:
                continue

            candidate = dist_partial[p] + dist_p[v]

            if candidate < combined_dist[v]:

                combined_dist[v] = candidate

                combined_pred[v] = p

    return combined_dist, combined_pred, P, W, U_hat, S, fp_time


# =========================================================
# PRECISÃO
# =========================================================


def compare_distances(d1, d2):
    """
    Compara precisão entre algoritmos.
    """

    total = 0

    correct = 0

    for v in d1:

        total += 1

        if d1[v] == d2[v]:
            correct += 1

    return 100 * correct / total


# =========================================================
# CSV
# =========================================================


def init_csv():
    """
    Inicializa CSV de results.
    """

    with open(RESULTS_CSV, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "experimento",
            "algoritmo",
            "tipo_Graph",
            "n",
            "e",
            "k",
            "tempo",
            "precisao",
            "speedup",
            "S",
            "W",
            "P",
            "U_hat",
            "coverage"
        ])



def append_csv(row):

    with open(RESULTS_CSV, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(row)


# =========================================================
# EXPERIMENTO 1 — ESCALABILIDADE
# =========================================================


def experiment_scalability():
    """
    Compara:
    - Dijkstra
    - Bellman-Ford
    - Dijkstra + FindPivots
    """

    print("\n================================================")
    print("EXPERIMENTO 1 — ESCALABILIDADE")
    print("================================================")

    dijkstra_sparse_times = []
    dijkstra_dense_times = []

    bellman_sparse_times = []
    bellman_dense_times = []

    preprocess_sparse_times = []
    preprocess_dense_times = []

    sparse_ns = []
    dense_ns = []

    # -----------------------------------------------------
    # sparseS
    # -----------------------------------------------------

    for n in SPARSE_SIZES:

        graph = generate_sparse_graph(n, 2)

        e = sum(len(graph[u]) for u in graph)

        source = 0

        print(f"\n[sparse] n={n} e={e}")

        dist_dij, pred_dij, t_dij = dijkstra(graph, source)

        print(f"Dijkstra: {t_dij:.6f}s")

        if n <= 1000:

            dist_bf, pred_bf, t_bf, neg = bellman_ford(
                graph,
                source
            )

            bellman_sparse_times.append(t_bf)

            print(f"Bellman-Ford: {t_bf:.6f}s")

        else:
            t_bf = None

        partial_size = int(math.sqrt(n))

        dist_pre, pred_pre, P, W, U_hat, S, fp_time = dijkstra_with_pivots(
            graph,
            source,
            8,
            partial_size
        )

        precision = compare_distances(dist_dij, dist_pre)

        speedup = t_dij / fp_time if fp_time > 0 else 0

        print(f"FindPivots+Dijkstra: {fp_time:.6f}s")
        print(f"Precisão: {precision:.2f}%")
        print(f"Speedup: {speedup:.4f}x")

        sparse_ns.append(n)

        dijkstra_sparse_times.append(t_dij)

        preprocess_sparse_times.append(fp_time)

        append_csv([
            "Escalabilidade",
            "Dijkstra",
            "sparse",
            n,
            e,
            "-",
            t_dij,
            100,
            1,
            "-",
            "-",
            "-",
            "-",
            "-"
        ])

        append_csv([
            "Escalabilidade",
            "FindPivots+Dijkstra",
            "sparse",
            n,
            e,
            8,
            fp_time,
            precision,
            speedup,
            len(S),
            len(W),
            len(P),
            len(U_hat),
            "-"
        ])

    # -----------------------------------------------------
    # denseS
    # -----------------------------------------------------

    for n in DENSE_SIZES:

        graph = generate_dense_graph(n, 4)

        e = sum(len(graph[u]) for u in graph)

        source = 0

        print(f"\n[dense] n={n} e={e}")

        dist_dij, pred_dij, t_dij = dijkstra(graph, source)

        print(f"Dijkstra: {t_dij:.6f}s")

        dist_bf, pred_bf, t_bf, neg = bellman_ford(
            graph,
            source
        )

        print(f"Bellman-Ford: {t_bf:.6f}s")

        partial_size = int(math.sqrt(n))

        dist_pre, pred_pre, P, W, U_hat, S, fp_time = dijkstra_with_pivots(
            graph,
            source,
            8,
            partial_size
        )

        precision = compare_distances(dist_dij, dist_pre)

        speedup = t_dij / fp_time if fp_time > 0 else 0

        print(f"FindPivots+Dijkstra: {fp_time:.6f}s")
        print(f"Precisão: {precision:.2f}%")
        print(f"Speedup: {speedup:.4f}x")

        dense_ns.append(n)

        dijkstra_dense_times.append(t_dij)

        bellman_dense_times.append(t_bf)

        preprocess_dense_times.append(fp_time)

    # -----------------------------------------------------
    # GRÁFICOS
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(sparse_ns, dijkstra_sparse_times, marker='o', label='Dijkstra sparse')

    plt.plot(sparse_ns, preprocess_sparse_times, marker='o', label='FindPivots sparse')

    plt.xlabel('n')

    plt.ylabel('Tempo (s)')

    plt.title('Tempo x n (sparses)')

    plt.legend()

    plt.savefig('results/grafics/sparses_tempo.png')

    plt.close()


# =========================================================
# EXPERIMENTO 2 — LEMA 3.2
# =========================================================


def experiment_lemma():
    """
    Verificação experimental do Lema 3.2.
    """

    print("\n================================================")
    print("EXPERIMENTO 2 — LEMA 3.2")
    print("================================================")

    n = 1000

    graph = generate_sparse_graph(n, 2)

    source = 0

    Vn = int(math.sqrt(n))

    dist, pred, U_hat, S, B = partial_dijkstra(
        graph,
        source,
        Vn
    )

    for v in graph:
        d_hat[v] = dist[v]

    ratio_values = []

    p_values = []

    s_values = []

    k_values = []

    coverage_values = []

    execution_x = []

    execution_y = []

    for k in K_VALUES:

        P, W, F, elapsed = find_pivots(
            B,
            list(S),
            graph,
            k
        )

        ratio = len(P) / max(1, (len(W) / k))

        coverage = compute_coverage(U_hat, W, P, pred)

        complexity_left = k * len(W)

        complexity_right = min(
            k * k * len(S),
            k * len(U_hat)
        )

        print("\n----------------------------------------")
        print(f"k = {k}")
        print("----------------------------------------")

        print(f"|S| = {len(S)}")
        print(f"|W| = {len(W)}")
        print(f"|P| = {len(P)}")
        print(f"|Û| = {len(U_hat)}")

        print(f"Razão do lema = {ratio:.4f}")

        print(f"Cobertura = {coverage:.2f}%")

        print(
            f"Complexidade válida? "
            f"{complexity_left <= complexity_right}"
        )

        print(f"Tempo = {elapsed:.6f}s")

        ratio_values.append(ratio)

        p_values.append(len(P))

        s_values.append(len(S))

        k_values.append(k)

        coverage_values.append(coverage)

        execution_x.append(k * len(W))

        execution_y.append(elapsed)

        append_csv([
            "Lema 3.2",
            "FindPivots",
            "sparse",
            n,
            sum(len(graph[u]) for u in graph),
            k,
            elapsed,
            "-",
            "-",
            len(S),
            len(W),
            len(P),
            len(U_hat),
            coverage
        ])

    # histograma
    plt.figure(figsize=(8, 5))

    plt.hist(ratio_values, bins=10)

    plt.xlabel('|P|/(|W|/k)')

    plt.ylabel('Frequência')

    plt.title('Distribuição da Razão do Lema')

    plt.savefig('results/grafics/histograma_lema.png')

    plt.close()

    # k x |P|
    plt.figure(figsize=(8, 5))

    plt.plot(k_values, p_values, marker='o')

    plt.xlabel('k')

    plt.ylabel('|P|')

    plt.title('k x |P|')

    plt.savefig('results/grafics/k_vs_p.png')

    plt.close()

    # |S| x |P|
    plt.figure(figsize=(8, 5))

    plt.scatter(s_values, p_values)

    plt.xlabel('|S|')

    plt.ylabel('|P|')

    plt.title('|S| x |P|')

    plt.savefig('results/grafics/s_vs_p.png')

    plt.close()

    # tempo x k|W|
    plt.figure(figsize=(8, 5))

    plt.plot(execution_x, execution_y, marker='o')

    plt.xlabel('k|W|')

    plt.ylabel('Tempo (s)')

    plt.title('Tempo vs k|W|')

    plt.savefig('results/grafics/tempo_vs_kw.png')

    plt.close()


# =========================================================
# EXPERIMENTO 3 — SENSIBILIDADE AO k
# =========================================================


def experiment_k_sensitivity():
    """
    Avalia impacto do parâmetro k.
    """

    print("\n================================================")
    print("EXPERIMENTO 3 — SENSIBILIDADE AO k")
    print("================================================")

    n = 1000

    graph = generate_sparse_graph(n, 2)

    source = 0

    dist_classic, pred_classic, t_classic = dijkstra(
        graph,
        source
    )

    partial_size = int(math.sqrt(n))

    gains = []

    precisions = []

    pivots = []

    for k in K_VALUES:

        dist_pre, pred_pre, P, W, U_hat, S, fp_time = dijkstra_with_pivots(
            graph,
            source,
            k,
            partial_size
        )

        precision = compare_distances(
            dist_classic,
            dist_pre
        )

        gain = t_classic / fp_time if fp_time > 0 else 0

        gains.append(gain)

        precisions.append(precision)

        pivots.append(len(P))

        print("\n----------------------------------------")
        print(f"k = {k}")
        print("----------------------------------------")

        print(f"|P| = {len(P)}")
        print(f"Precisão = {precision:.2f}%")
        print(f"Ganho = {gain:.4f}x")

    # k x ganho
    plt.figure(figsize=(8, 5))

    plt.plot(K_VALUES, gains, marker='o')

    plt.xlabel('k')

    plt.ylabel('Speedup')

    plt.title('k x Speedup')

    plt.savefig('results/grafics/k_vs_speedup.png')

    plt.close()

    # k x precisão
    plt.figure(figsize=(8, 5))

    plt.plot(K_VALUES, precisions, marker='o')

    plt.xlabel('k')

    plt.ylabel('Precisão (%)')

    plt.title('k x Precisão')

    plt.savefig('results/grafics/k_vs_precisao.png')

    plt.close()


# =========================================================
# MAIN
# =========================================================


def main():
    """
    Executa todos os experimentos automaticamente.
    """

    total_start = time.perf_counter()

    init_csv()

    generate_all_graphs()

    experiment_scalability()

    experiment_lemma()

    experiment_k_sensitivity()

    total_elapsed = time.perf_counter() - total_start

    print("\n================================================")
    print("All experiments completed!")
    print("================================================")

    print(f"TOtal time: {total_elapsed:.2f}s")

    print("\nGenenerated files:")
    print("- Graphs/")
    print("- results/grafics/")
    print("- results/tables/")


if __name__ == "__main__":
    main()

import matplotlib, time, os, random, heapq, math, csv

matplotlib.use("TkAgg") 

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple

def main():

    # tests()

    runner = BenchmarkRunner()
    runner.run_and_export()


def tests():
    g,s = Graph.graph_generator(
        e=1,
        n=4
    )

    # g = Graph(3)
    g.add_edge(2,0,0)

    g.draw()
    g.print_adj_list()
class Graph:
    
    def __init__(
            self, 
            nodes: int
        ):
        """
        Initializes the graph adjacency matrix with a nodes X nodes size matrix
        """
        self.nodes = nodes
        self.adj_list:list[
                        dict[int,int]
                    ] = [{} for _ in range(nodes)]

    def add_edge(
            self,
            u: int, 
            v: int, 
            w: int
        ):
        self.adj_list[u][v] = w

    def remove_edge(
            self, 
            u: int, 
            v: int
        ):
        self.adj_list[u].pop(v,None)

    def get_neighbors(
            self, 
            u: int
        ) -> list[tuple[int, int]]:
        return list(self.adj_list[u].items())

    def has_edge(
            self, 
            u: int, 
            v: int
        ) -> bool:
        return v in self.adj_list[u]

    def print_adj_list(self):
        for row in self.adj_list:
            print(row)

    
    def draw(self):
        """
        Responsible for drawing the graph network using the networkx library.
        """
        G = nx.DiGraph()

        G.add_nodes_from(range(self.nodes))

        for u in range(self.nodes):
            for v, weight in self.adj_list[u].items():
                G.add_edge(u,v, weight= weight)

        pos = nx.shell_layout(G)

        node_values = [G.degree(node) for node in G.nodes()]

        nx.draw(
            G, 
            pos, 
            with_labels=(self.nodes <= 20), 
            node_color= node_values, 
            cmap="viridis",
            node_size=5000 * 1 / self.nodes
        )
        
        if self.nodes <= 20 : 
            edge_labels = nx.get_edge_attributes(
                G, 
                'weight'
            )
            nx.draw_networkx_edge_labels(
                G, 
                pos, 
                edge_labels=edge_labels
            )

        plt.show(block=True)

        save_path = f"./results/graphs/graph_{self.nodes}_nodes.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph visualization saved to {save_path}")


    
    def dijkstra(
        self,
        source: int
    ):
        """
        Implementation of Dijkstra's ALgorithm for Single Source Shortest Path.
        This runs in O( log(V) (V+E) )
        """
        start = time.   perf_counter()
        
        dist: dict[int, float] = {
            node: float('inf') 
            for node in range(self.nodes)
        }
        
        visited: dict[int, bool] = {
            node: False 
            for node in range(self.nodes)
        }
        
        prev: dict[int, int | None ] = {
            node: None 
            for node in range(self.nodes)
        }
        
        dist[source] = 0

        pq: list[tuple[float, int]] = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue

            visited[current_node] = True

            for neighbor, weight in self.get_neighbors(current_node):

                if weight <= 0:
                    continue

                new_distance = current_dist + weight

                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    prev[neighbor] = current_node
                    heapq.heappush(
                        pq, 
                        (new_distance, neighbor)
                    )
        end = time.perf_counter()
        execution_time = end - start
        
        return dist, prev, execution_time
    
    def find_pivots(
            self, 
            B: float, 
            S: list[int], 
            d_hat: dict[int, float], 
            k: int
    ) -> tuple[set[int], set[int]]:
        """
        Args:
            B: Superior distance Limit
            S: Source nodes set (Boundary)
            d_hat: Global Array (dictionary) with estimated distances.
            k: Deepness parameter. ( lo^{1/3}(n))
            
        Returns:
            P: Pivot's Set
            W: Completed Visited
        """
        W: set[int] = set(S)
        P: set[int] = set()

        root_of: dict[int, int] = {s: s for s in S}
        tree_size: dict[int, int] = {s: 1 for s in S}

        current_layer: list[int] = list(S)

        for step in range(k):
            if not current_layer:
                break  

            next_layer: list[int] = []

            for u in current_layer:
                for v, weight in self.get_neighbors(u):
                    
                    if d_hat.get(u, float('inf')) + weight < d_hat.get(v, float('inf')):
                        
                        new_dist = d_hat[u] + weight
                        
                        if new_dist <= B:
                            d_hat[v] = new_dist
                            
                            if v not in W:
                                W.add(v)
                                next_layer.append(v)
                            
                            old_root = root_of.get(v)
                            new_root = root_of[u]
                            
                            if old_root != new_root:
                                if old_root is not None:
                                    tree_size[old_root] -= 1
                                root_of[v] = new_root
                                tree_size[new_root] += 1

            current_layer = next_layer

            if len(W) > k * len(S):
                break

        for s in S:
            if tree_size.get(s, 0) >= k:
                P.add(s)

        return P, W

    def partial_dijkstra(
            self,
            source: int,
            extraction_limit: int
        ) -> tuple[dict[int, float], float, list[int]]:
        """
        Executes Dijkstra's algorithm until a specific number of vertices 
        are processed from the priority queue.
        
        Args:
            source: The starting node for the algorithm.
            extract_limit: The maximum number of vertices to pop from the queue (V_n).
            
        Returns:
            dist: The current estimated distances dictionary (d_hat).
            B: The distance of the last extracted node.
            S: The frontier/boundary set of nodes currently in the priority queue.
        """

        dist: dict[int, float] = {
            node: float('inf') 
            for node in range(self.nodes)
        }
        visited: dict[int, bool] = {
            node: False 
            for node in range(self.nodes)
        }

        dist[source] = 0
        pq: list[tuple[float, int]] = [(0, source)]

        processed_count = 0
        last_dist = 0.0

        while pq and processed_count < extraction_limit:
            current_dist, current_node = heapq.heappop(pq)

            if visited[current_node]:
                continue

            visited[current_node] = True
            processed_count += 1
            last_dist = current_dist

            for neighbor, weight in self.get_neighbors(current_node):
                if weight <= 0:
                    continue

                new_distance = current_dist + weight

                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(
                        pq, 
                        (new_distance, neighbor)
                    )

        S = list(set([node for _, node in pq]))
        
        return dist, last_dist, S


    def optimized_dijkstra_with_pivots(
            self,
            source: int,
            extraction_limit: int,
            k: int
    ):
        """
        Implements the application of find_pivots as a pre-processing step 
        for Dijkstra, treating pivots as super-sources and combining the results.
        
        Args:
            source: The initial starting node.
            extract_limit: Vertices to process in the initial partial Dijkstra.
            k: Deepness parameter for the find_pivots algorithm.
            
        Returns:
            combined_dist: The final combined minimum distances from the source.
            metrics: Dictionary containing S, W, P, and U_tilde sizes.
            execution_time: Total execution time in seconds.
        """

        start = time.   perf_counter()
        
        d_hat, B, S = self.partial_dijkstra(source, extraction_limit)

        P, W = self.find_pivots(B, S, d_hat, k)

        pivot_distances: list[dict[int,float]] = []

        for p in P:
            p_dist, _, _ = self.dijkstra(p)
            pivot_distances.append(p_dist)
            
        combined_dist: dict[int, float] = {}
        
        for node in range(self.nodes):
            min_dist = d_hat.get(node, float('inf'))
            
            for index, p in enumerate(P):
                p_dist = pivot_distances[index]
                path_via_p = d_hat.get(p, float('inf')) + p_dist.get(node, float('inf'))
                
                if path_via_p < min_dist:
                    min_dist = path_via_p
                    
            combined_dist[node] = min_dist

        end = time.perf_counter()
        execution_time = end - start
            
        u_tilde = sum(1 for dist in d_hat.values() if dist <= B)
        
        metrics = {
            "size_S": len(S),
            "size_W": len(W),
            "size_P": len(P),
            "size_U_tilde": u_tilde
        }

        return combined_dist, metrics, execution_time

    def bellman_ford(self, source: int):
        """
        Implementation of bellman_ford algorithms for Single Source Shortest Path.
        This runs in O(V * E)
        """
        dist: dict[int, float] = {
            node: float('inf') for node in range(self.nodes)
        }
        
        prev: dict[int, int | None ] = {
            node: None 
            for node in range(self.nodes)
        }
        dist[source] = 0

        # relaxation V-1 times
        for _ in range(self.nodes - 1):
            changed = False

            for u in range(self.nodes):
                for v, weight in self.get_neighbors(u):
                    if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        prev[v] = u
                        changed = True

            if not changed:
                break

        for u in range(self.nodes):
            for v, weight in self.get_neighbors(u):
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    raise ValueError("The graph has a negative cycle")

        return dist, prev
    

    @staticmethod
    def graph_generator(
            e: int,
            n: int,
            minimum_weight: int = 1,
            maximum_weight: int = 20,
            directed: bool = True,
            file_name: str | None = None,
            file_path: str | None = None
    ) -> tuple['Graph', int]:
        
        max_possible_edges = n * (n - 1) if directed else (n * (n - 1)) // 2
        if e > max_possible_edges:
            raise ValueError(f"Impossible to generate {e} unique edges for {n} nodes. Maximum Capacity: {max_possible_edges}")

        graph = Graph(n)
        edges: list[tuple[int, int, int]] = []
        seen_edges: set[tuple[int, int]] = set()

        while len(edges) < e:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)

            if u == v:
                continue

            check_u, check_v = (u, v) if directed or u < v else (v, u)

            if (check_u, check_v) in seen_edges:
                continue
            
            seen_edges.add((check_u, check_v))
            weight = random.randint(minimum_weight, maximum_weight)
            
            edges.append((check_u, check_v, weight))

            graph.add_edge(check_u, check_v, weight)
            if not directed:
                graph.add_edge(check_v, check_u, weight)

        s = random.randint(0, n - 1)

        if file_name and file_path:
            os.makedirs(file_path, exist_ok=True)
            full_path = os.path.join(file_path, file_name)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(f"{n} {e}\n")
                for u, v, weight in edges:
                    f.write(f"{u} {v} {weight}\n")
                f.write(f"{s}\n")

            print(f"Graph generated and saved to {full_path}, with {n} nodes, {e} edges")

        return graph, s
   
    @classmethod
    def from_txt(
            cls,
            filename: str
    ):
        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            n, e = map(
                int,
                f.readline().split()
            )

            graph = cls(n)

            for _ in range(e):
                u, v, w = map(
                    int,
                    f.readline().split()
                )

                graph.add_edge(u, v, w)

            s = int(f.readline())

        return graph, s

import csv
import math
from pathlib import Path

class BenchmarkRunner:
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.graphs_dir = self.output_dir / "generated_graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    def _get_or_create_graph(self, e: int, n: int, filename: str) -> tuple['Graph', int]:
        """
        Acts as a proxy: if file exists, reads from disk O(V+E).
        Otherwise, calls the generator and saves to disk.
        """
        file_path = self.graphs_dir / filename
        
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"Cache Hit: Loading {filename} from disk...")
            return Graph.from_txt(str(file_path))
        
        print(f"Cache Miss: Generating new graph {filename}...")
        return Graph.graph_generator(
            e=e, n=n, directed=True, 
            file_name=filename, file_path=str(self.graphs_dir)
        )

    def _format_result(
            self, 
            exp_name: str, 
            g_type: str, 
            n: int, 
            e: int, 
            k: int, 
            metrics: dict, 
            base_time: float, 
            opt_time: float
    ) -> dict:
        """Helper function to keep dictionary creation DRY."""
        return {
            "Experiment": exp_name,
            "Type": g_type,
            "N": n,
            "E": e,
            "K": k,
            "|S|": metrics["size_S"],
            "|W|": metrics["size_W"],
            "|P|": metrics["size_P"],
            "|U_tilde|": metrics["size_U_tilde"],
            "Dijkstra Time (s)": f"{base_time:.6f}",
            "Pivots Time (s)": f"{opt_time:.6f}"
        }

    def _run_experiment_1_scalability(self) -> list[dict]:
        """Executes Experiment 1 and returns the results."""
        results = []
        configs = [
            {"n": 100, "e_mult": 2, "e_div": 4},
            {"n": 500, "e_mult": 2, "e_div": 4},
            {"n": 1000, "e_mult": 2, "e_div": 4},
            {"n": 5000, "e_mult": 2, "e_div": 4},
            {"n": 10000, "e_mult": 2, "e_div": 4}
        ]
        
        print("\n--- Starting Experiment 1: Scalability ---")
        
        for config in configs:
            n = config["n"]
            extraction_limit = int(math.sqrt(n)) 
            k_default = 4 

            # Sparse Graph Run
            mult = config["e_mult"]
            e_sparse = n * mult
            sparse_filename = f"sparse_n_{n}_e_{mult}n.txt"
            
            graph_s, source_s = self._get_or_create_graph(e=e_sparse, n=n, filename=sparse_filename)
            _, _, base_time_s = graph_s.dijkstra(source_s)
            _, metrics_s, opt_time_s = graph_s.optimized_dijkstra_with_pivots(
                source=source_s, extraction_limit=extraction_limit, k=k_default
            )
            
            results.append(self._format_result(
                "1_Scalability", "sparse", n, e_sparse, k_default, metrics_s, base_time_s, opt_time_s
            ))

            # Dense Graph Run
            div = config["e_div"]
            e_dense = (n * (n - 1)) // div
            dense_filename = f"dense_n_{n}_e_div_{div}.txt"
            
            graph_d, source_d = self._get_or_create_graph(e=e_dense, n=n, filename=dense_filename)
            _, _, base_time_d = graph_d.dijkstra(source_d)
            _, metrics_d, opt_time_d = graph_d.optimized_dijkstra_with_pivots(
                source=source_d, extraction_limit=extraction_limit, k=k_default
            )
            
            results.append(self._format_result(
                "1_Scalability", "dense", n, e_dense, k_default, metrics_d, base_time_d, opt_time_d
            ))

        return results

    def _run_experiment_2_sensitivity(self) -> list[dict]:
        """Executes Experiment 2 and returns the results."""
        results = []
        k_values = [2, 4, 6, 8, 10, 12, 14, 16]
        n = 1000
        mult = 2
        e = n * mult
        
        print("\n--- Starting Experiment 2: k Sensitivity ---")
        
        filename = f"sparse_n_{n}_e_{mult}n.txt"
        graph, source = self._get_or_create_graph(e=e, n=n, filename=filename)
        extraction_limit = int(math.sqrt(n))
        
        _, _, base_time = graph.dijkstra(source)

        for k in k_values:
            _, metrics, opt_time = graph.optimized_dijkstra_with_pivots(
                source=source, extraction_limit=extraction_limit, k=k
            )
            
            results.append(self._format_result(
                "2_Sensitivity", "sparse", n, e, k, metrics, base_time, opt_time
            ))

        return results

    def run_and_export(self):
        """Orchestrates all experiments and exports data to CSV."""
        exp1_results = self._run_experiment_1_scalability()
        self._save_to_csv(exp1_results, "benchmark_exp1_scalability.csv")

        exp2_results = self._run_experiment_2_sensitivity()
        self._save_to_csv(exp2_results, "benchmark_exp2_sensitivity.csv")


    def _save_to_csv(self, data: list[dict], filename: str):
        """Saves a list of dictionaries to a CSV file."""
        if not data:
            return
            
        csv_path = self.output_dir / filename
        keys = data[0].keys()
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
            
        print(f"\nSuccess! Results exported to: {csv_path}")

if __name__ == "__main__":
    main()
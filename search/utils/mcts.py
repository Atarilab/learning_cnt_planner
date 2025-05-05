import random
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from typing import Any, Tuple
from tqdm import tqdm, trange

from .abstract import Graph

Node = Any

class MCTSBase(ABC):
    def __init__(self, graph: Graph, C: float = 1.0):
        self.graph = graph
        self.value_visit: dict[tuple[Node, Node], tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        self.visit_count: dict[Node, int] = defaultdict(int)
        self.current_search_path: list[Node] = []
        self.C = C
        self.it = 0
        
    def to_mcts_node(self, node : Node):
        return node
    
    def get_value_visit(self, parent : Node, child : Node) -> Tuple[float, int, int]:
        parent_mcts, child_mcts = self.to_mcts_node(parent), self.to_mcts_node(child)
        w, n = self.value_visit[(parent_mcts, child_mcts)]
        N = self.visit_count[parent_mcts]
        return w, n, N
    
    def increase_value_visit(self, parent : Node, child : Node, value) -> None:
        parent_mcts, child_mcts = self.to_mcts_node(parent),  self.to_mcts_node(child)
        w, n = self.value_visit[(parent_mcts, child_mcts)]
        self.value_visit[(parent_mcts, child_mcts)] = (w + value, n + 1)
        self.visit_count[parent_mcts] += 1
        
    def increase_visit(self, node : Node) -> None:
        node_mcts = self.to_mcts_node(node)
        self.visit_count[node_mcts] += 1
    
    def has_been_explored(self, node : Node) -> bool:
        node_mcts = self.to_mcts_node(node)
        return self.visit_count[node_mcts] > 0

    def heuristic_bias(self, parent: Node, node: Node) -> float:
        return 0.0

    def UCB(self, parent: Node, child: Node) -> float:
        w, n, N = self.get_value_visit(parent, child)
        if n == 0:
            return float("inf")
        return ((w + self.heuristic_bias(parent, child)) / n) + self.C * np.sqrt(np.log(N) / n)

    def is_leaf(self, node: Node) -> bool:
        return len(self.graph.neighbors(node)) == 0

    def best_child(self, node: Node) -> Node:
        if self.is_leaf(node):
            return node
        children = self.graph.neighbors(node)
        np.random.shuffle(children)
        return max(children, key=lambda child: self.UCB(node, child))

    def select(self, root: Node) -> Node:
        self.current_search_path = [root]
        node = root

        while self.has_been_explored(node) and not self.is_leaf(node):
            node = self.best_child(node)
            self.current_search_path.append(node)

        return node

    @abstractmethod
    def evaluate(self, simulation_path: list[Node]) -> float:
        pass

    def rollout_policy(self, node: Node) -> Node:
        children = self.graph.neighbors(node)
        return random.choice(children) if children else node

    def simulate(self, node: Node) -> float:
        simulation_path = []
        while not self.is_leaf(node):
            node = self.rollout_policy(node)
            simulation_path.append(node)
        return self.evaluate(self.current_search_path + simulation_path)

    def backpropagate(self, reward: float):
        # Update all transitions in reverse
        for parent, child in zip(self.current_search_path[-1::-1], self.current_search_path[-2::-1]):
            self.increase_value_visit(parent, child, reward)
        
        if len(self.current_search_path) > 1:
            self.increase_visit(child)
        else:
            self.increase_visit(self.current_search_path[0])
            
    def run(self, root: Node, iterations: int):
        MAX_ATTEMPS = 5
        
        pbar = trange(0, iterations, desc="MCTS")

        for self.it in pbar:
            reward = None
            attempts = 0
            selected = None

            while reward is None and attempts < MAX_ATTEMPS:
                selected = self.select(root)
                reward = self.simulate(selected)
                attempts += 1

            if reward is None:
                reward = 0.0

            self.backpropagate(reward)

            # Update statistics
            depth = len(self.current_search_path)
            pbar.set_postfix({
                "depth": depth,
                "expanded": len(self.visit_count),
            })

    def best_path(self, root: Node) -> list[Node]:
        path = [root]
        node = root
        while not self.is_leaf(node):
            children = self.graph.neighbors(node)
            if not children:
                break
            node = max(children, key=lambda child: self.value_visit[(node, child)][0] / (self.value_visit[(node, child)][1] + 1e-6))
            if node == path[-1]:
                break
            path.append(node)
        return path

    def best_value_node(self) -> Node:
        if not self.value_visit:
            return None
        best_edge = max(self.value_visit.items(), key=lambda item: item[1][0] / (item[1][1] + 1e-6))[0]
        return best_edge[1]

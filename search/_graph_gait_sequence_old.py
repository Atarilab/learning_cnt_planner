import numpy as np
from itertools import product

try:
    from .utils.abstract import Graph
    from .utils.mcts import MCTSBase
    
except:
    from utils.abstract import Graph
    from utils.mcts import MCTSBase

type Node = str

class ChunkBinaryTree(Graph):
    def __init__(self, n_nodes : int = 64, allow_all_equal : bool = True):
        super().__init__()
        base_float = np.log(n_nodes) / np.log(2)
        self.base = round(base_float)
        self.resolution = self.base
        self.allow_all_equal : bool = allow_all_equal
        
        if self.base != base_float:
            self.n_nodes = int(2 ** self.base)
            print(f"Number of nodes set to a multiple of 2: ({self.n_nodes}).")
        else:
            self.n_nodes = n_nodes
        
        self.stage_n_bytes = len(bin(self.base)) - 2
        self.start_node = "0" * (self.stage_n_bytes) + "1" * self.n_nodes
        
    def increase_res(self) -> None:
        self.resolution += 1
        
    def decrease_res(self) -> None:
        self.resolution -= 1   
        
    def set_resolution(self, resolution : int) -> None:
        self.resolution = resolution
        
    def get_stage_bin_state(self, node) -> tuple[int, str]:
        stage = int(node[:self.stage_n_bytes], 2)
        bin_state = node[self.stage_n_bytes:]
        return stage, bin_state
    
    def all_equal(self, bin_state : str):
        return bin_state.count(bin_state[0]) == len(bin_state)
        
    def get_neighbors(self, node):
        stage, bin_state = self.get_stage_bin_state(node)
        if stage == self.base or stage >= self.resolution:
            return []
        
        # Option to terminate with the current state -> go to last stage
        if not self.allow_all_equal and self.all_equal(bin_state):
            neighbors = []
        else:
            neighbors = [f"{bin(stage+1)[2:].zfill(self.stage_n_bytes)}{bin_state}"]
        
        # Generate all neighbors
        subdiv = 2 ** (stage+1)
        chunk_size = self.n_nodes // (subdiv)
        for i in range(subdiv):
            # Flip one chunk according to the current subdivision
            chunk_start = chunk_size * i
            chunk_end = chunk_size * (i + 1)
            chunk = bin_state[chunk_start:chunk_end]
            flipped_chunk = format(int(chunk, 2) ^ (2**chunk_size - 1), f'0{chunk_size}b')
            new_bin_state = bin_state[:chunk_start] + flipped_chunk + bin_state[chunk_end:]
            
            if not self.allow_all_equal and self.all_equal(new_bin_state):
                continue
            else:
                new_node = f"{bin(stage+1)[2:].zfill(self.stage_n_bytes)}{new_bin_state}"
                neighbors.append(new_node)
                
                new_node = f"{bin(stage)[2:].zfill(self.stage_n_bytes)}{new_bin_state}"
                if new_node != node:
                    neighbors.append(new_node)
        
        return neighbors
    
class GaitParallelGraph(Graph):
    def __init__(self, n_eeff : int, n_nodes_per_eeff : int):
        super().__init__()
        # Create one separate tree for each end effector
        # A node is a tupple of all tree binary state
        self.n_nodes_per_eeff = n_nodes_per_eeff
        self.trees = [ChunkBinaryTree(n_nodes_per_eeff, False) for _ in range(n_eeff)]
        self.base = self.trees[0].base
        self.start_node = tuple(tree.start_node for tree in self.trees)
        self.mcts_edges = {}
        
    def increase_res(self) -> None:
        map(lambda tree : tree.increase_res(), self.trees)
        
    def decrease_res(self) -> None:
        map(lambda tree : tree.decrease_res(), self.trees)
        
    def set_resolution(self, resolution : int) -> None:
        map(lambda tree : tree.set_resolution(resolution), self.trees)       
        
    def get_neighbors(self, node : Node) -> list[Node]:
        neighbors = list(product(*(map(lambda children : children if children else n, tree.get_neighbors(n)) for tree, n in zip(self.trees, node))))
        return neighbors
    
    def to_mcts_node(self, node : Node):
        return tuple(tree_state[self.trees[0].stage_n_bytes:] for tree_state in node)
    
class GaitSequencialGraph(Graph):
    SPLIT_MODE = {
        "chunk" : 0,
        "interval" : 1,
    }
    def __init__(self, n_eeff : int, n_nodes_per_eeff : int, split_eeff_mode : str = "chunk"):
        self.n_eeff = n_eeff
        self.n_nodes_per_eeff = n_nodes_per_eeff
        n_nodes = n_eeff * n_nodes_per_eeff
        
        self.split_eeff_id = dict.get(GaitSequencialGraph.SPLIT_MODE, split_eeff_mode, 0)

        # Create one large tree for all end effectors combined
        self.tree = ChunkBinaryTree(n_nodes)
        self.start_node = self.convert_to_graph_node(self.tree.start_node)
        self.base = self.tree.base
        
    def increase_res(self) -> None:
        self.tree.increase_res()
        
    def decrease_res(self) -> None:
        self.tree.decrease_res()
        
    def set_resolution(self, resolution : int) -> None:
        self.tree.set_resolution(resolution)     
        
    def convert_to_graph_node(self, node : Node) -> Node:
        """
        Split node from the tree state to individual state for each end effectors.
        """
        o = self.tree.stage_n_bytes
        stage = node[:o]
        if self.split_eeff_id == 0:
            node_eeff = (node[o+i*self.n_nodes_per_eeff:o+(i+1)*self.n_nodes_per_eeff] for i in range(self.n_eeff))
        elif self.split_eeff_id == 1:
            node_eeff = (node[o+i::self.n_eeff] for i in range(self.n_eeff))
             
        return (stage, *node_eeff)
    
    def convert_to_tree_node(self, node : Node) -> Node:
        if self.split_eeff_id == 0:
            return "".join(node)
        elif self.split_eeff_id == 1:
            s = node[0] + "".join(node[j+1][i] for i in range(self.n_nodes_per_eeff) for j in range(self.n_eeff))
            return s
        
    def get_neighbors(self, node : Node) -> list[Node]:
        neighbors_tree = self.tree.get_neighbors(self.convert_to_tree_node(node))
        neighbors = list(map(self.convert_to_graph_node, neighbors_tree))
        return neighbors
    
    def to_mcts_node(self, node : Node):
        # Remove the resolution state
        return node[1:]
    
    
if __name__ == "__main__":
    import time
    
    #########
    ######### ChunkBinaryTree
    #########
    N_NODES = 16
    tree = ChunkBinaryTree(N_NODES, allow_all_equal=False)
    
    print("Test neighbors")
    n = [tree.start_node]
    tree.set_resolution(2)
    for i in range(tree.base + 1):
        print("Stage", i)
        node = n[-1]
        print("node", node)
        n = tree.neighbors(node)
        print("neighbors", n)

    # timings
    
    N_NODES = 8
    tree = ChunkBinaryTree(N_NODES, allow_all_equal=False)
    start = time.time()
    print("Time add all leaves")
    all_leaves = []
    explored = []
    from collections import deque
    def add_all_leaves(node):

        queue = deque([tree.start_node])

        while queue:
            node = queue.popleft()

            if node in all_leaves or node in explored:
                continue

            explored.append(node)
            neighbors = tree.neighbors(node)

            if not neighbors:
                all_leaves.append(node)
            else:
                queue.extend(neighbors) 

    add_all_leaves(tree.start_node)
    end = time.time()
    
    print(f"Time taken: {end - start} seconds")
    print(f"Number of leaves: {len(all_leaves)}")
    import sys
    total_size = sys.getsizeof(tree.edges)
    total_size_mb = total_size / (1024)
    print(f"Total memory size of all nodes: {total_size_mb:.2f} KB")
    
    
    # Test gait parrallel graph
    N_EFFF  = 4
    N_NODES = 8
    graph = GaitParallelGraph(N_EFFF, N_NODES)
    graph.set_resolution(1)
    start_node = graph.start_node
    print("--- Test neighbors parallel")
    print("Depth 1")
    print("start_node:", start_node)
    n = graph.get_neighbors(start_node)
    print("n neighbors", len(n))
    print(n)
    
    print("Depth 2")
    I = 0
    print("node:", n[I])
    n = graph.get_neighbors(n[I])
    print("n neighbors", len(n))

    
    # # Test gait sequencial graph
    # N_EFFF  = 4
    # N_NODES = 4
    # gait_tree = GaitSequencialGraph(N_EFFF, N_NODES, split_eeff_mode="chunk")
    # start_node = gait_tree.start_node
    # print("---- Test neighbors sequential")
    # print("Depth 1")
    # print("start_node:", start_node)
    # n = gait_tree.get_neighbors(start_node)
    # print("n neighbors", len(n))
    # print(n)
    
    # print("Depth 2")
    # I = 1
    # print("node:", n[I])
    # n = gait_tree.get_neighbors(n[I])
    # print(n)
    # print("n neighbors", len(n))

    # print("Depth 3")
    # I = 1
    # print("node:", n[I])
    # n = gait_tree.get_neighbors(n[I])
    # print(n)
    # print("n neighbors", len(n))

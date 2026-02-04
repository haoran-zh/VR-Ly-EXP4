



class HierarchicalNode:
    def __init__(self, node_id, level, memory_capacity, available_models, cost_budget_gamma):
        self.node_id = node_id
        self.level = level  # layer level
        self.memory_capacity = memory_capacity
        self.available_models = available_models
        self.cost_budget_gamma = cost_budget_gamma
        self.children = []
        self.parent = None
        self.onloaded_models = []
        self.Q = None
        self.w = None
        self.S = None
        # temp below
        self.b_true = None
        self.b_hat = None
        self.C_true = None
        self.C_hat = None
        self.exp_loss = None
        self.actual_loss = None
        self.p_to_cloud = None
        self.p_offload = None
        self.confidence = None

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self


class HierarchicalSystem:
    def __init__(self, num_layers, nodes_per_layer, layer_configs):
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.layer_configs = layer_configs
        self.nodes = {}
        self._build_tree()

    def _build_tree(self):
        layer_nodes = []
        for layer_idx in range(self.num_layers - 1, -1, -1):
            config = self.layer_configs[layer_idx]
            current_layer_nodes = []
            for i in range(self.nodes_per_layer[layer_idx]):
                node_id = f"L{layer_idx}_N{i}"
                is_cloud = (layer_idx == self.num_layers - 1)
                node = HierarchicalNode(
                    node_id=node_id, level=layer_idx,
                    memory_capacity=config.get('memory', 0) if not is_cloud else float('inf'),
                    available_models=config.get('models', []),
                    cost_budget_gamma=config.get('gamma', 0)
                )
                if layer_idx > 0:
                    node.Q = 0.0  # initialize the queue, except the initial layer
                self.nodes[node_id] = node
                current_layer_nodes.append(node)
                if layer_idx < self.num_layers - 1 and layer_nodes:  # if layer_nodes is empty, then it's the initial layer
                    parent_idx = i % len(layer_nodes)
                    layer_nodes[parent_idx].add_child(node)
            layer_nodes = current_layer_nodes

    def get_leaf_nodes(self):
        return [n for n in self.nodes.values() if n.level == 0]  # initial layer

    def get_cloud_node(self):
        return [n for n in self.nodes.values() if n.level == self.num_layers - 1][0]

    def get_non_cloud_nodes(self):
        return [n for n in self.nodes.values() if n.level < self.num_layers - 1]
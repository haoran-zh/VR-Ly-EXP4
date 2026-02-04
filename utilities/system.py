



class HierarchicalNode:
    def __init__(self, node_id, level, memory_capacity, available_models, cost_budget_gamma):
        self.node_id = node_id
        self.level = level  # layer level
        self.memory_capacity = memory_capacity
        self.available_models = available_models
        self.cost_budget_gamma = cost_budget_gamma
        self.children = []
        self.parents = []
        self.K_parents = None
        self.num_actions = None
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
        self.to = None

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def add_child_multi(self, child_node):
        # child_node is at lower level
        if child_node not in self.children:
            self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)
        # Keep a deterministic .parent for backward compatibility (first parent)
        if child_node.parent is None:
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




class HierarchicalSystemMulti:
    def __init__(self, num_layers, nodes_per_layer, layer_configs, multi_parent=False):
        """
        multi_parent=False: build a tree (each node has exactly one parent, as in the original code).
        multi_parent=True : build a layered DAG where each node at level l can offload to ANY node at level l+1.
        """
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.layer_configs = layer_configs
        self.multi_parent = multi_parent
        self.nodes = {}
        self._build()

    def _build(self):
        # Build nodes per layer
        layers = []
        for layer_idx in range(self.num_layers):
            config = self.layer_configs[layer_idx]
            current_layer_nodes = []
            for i in range(self.nodes_per_layer[layer_idx]):
                node_id = f"L{layer_idx}_N{i}"
                is_cloud = (layer_idx == self.num_layers - 1)
                node = HierarchicalNode(
                    node_id=node_id,
                    level=layer_idx,
                    memory_capacity=config.get('memory', 0) if not is_cloud else float('inf'),
                    available_models=config.get('models', []),
                    cost_budget_gamma=config.get('gamma', 0),
                )
                # Initialize queue for non-initial nodes (except cloud)
                # if 0 < layer_idx < self.num_layers - 1:
                node.Q = 0.0
                self.nodes[node_id] = node
                current_layer_nodes.append(node)
            layers.append(current_layer_nodes)

        # Connect edges upward (parent at level l+1, child at level l)
        for l in range(self.num_layers - 1):
            lower = layers[l]
            upper = layers[l + 1]
            if self.multi_parent:
                # Fully connect: every lower node can offload to any upper node
                for child in lower:
                    for parent in upper:
                        parent.add_child_multi(child)
            else:
                # Original modulo assignment (tree-like)
                for i, child in enumerate(lower):
                    parent = upper[i % len(upper)]
                    parent.add_child(child)

    def get_leaf_nodes(self):
        return [n for n in self.nodes.values() if n.level == 0]  # initial layer

    def get_cloud_node(self):
        return [n for n in self.nodes.values() if n.level == self.num_layers - 1][0]

    def get_non_cloud_nodes(self):
        return [n for n in self.nodes.values() if n.level < self.num_layers - 1]

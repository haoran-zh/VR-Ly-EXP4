import numpy as np
import matplotlib.pyplot as plt
from utilities.model_performance import estimate_model_perforance, convert_offloadingCost, estimate_offloadingCost, \
    lowest_avg_error
import pickle as pkl
import random
import copy

# read dataset
FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
with open(FILENAME, 'rb') as f:
    data = pkl.load(f)

Avg_acc, Avg_err = estimate_model_perforance(data)
ERROR_RATES = Avg_err

OffloadCost_SCALE = 0.001
avg_offloadCost = estimate_offloadingCost(data, OffloadCost_SCALE)

NUM_TASK_TYPES = len(data['task_types'])
TASK_NAMES = data['task_types']
TOTAL_JOBS = data['total_samples']

model_keywords = [
    'openai-community__gpt2-large',  # 0 - 0.7GB
    'gpt2-xl',  # 1 - 1.5GB
    'deepseek-llm-7b',  # 2 - 7.0GB
    'deepseek-ai__deepseek-moe-16b',  # 3 - 16.0GB
    'deepseek-ai__deepseek-llm-67b',  # 4 - 67.0GB
    'Qwen__Qwen2-0.5B',  # 5 - 0.5GB
    'Qwen__Qwen1.5-0.5B',  # 6 - 0.5GB
    'Qwen__Qwen2-72B',  # 7 - 72.0GB
]

MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])
MODEL_ONLOADING_COSTS = MODEL_SIZES
ONLOADING_EPOCH_LENGTH = 500
V_ONLOAD = 700


# ============================================================
# SYSTEM CONFIGURATIONS
# ============================================================

def get_system_configs():
    """
    Returns configurations for different system topologies.

    Model allocation strategy:
    - Layer 0 (Client): Tiny models (0.5-1.5GB) - models [0, 5, 6] or [0, 1, 5, 6]
    - Layer 1 (Edge): Small-medium models - models [0, 2, 3, 5, 6]
    - Layer 2 (Regional): Medium-large models - models [1, 3, 4, 7]
    - Layer 3 (Cloud): All models (infinite capacity)

    Gamma allocation:
    - Each node's gamma is the budget for INCOMING traffic from children
    - Higher layers typically have larger gamma to handle aggregated traffic
    """

    configs = {}

    # ========== 3-Layer (1-1-1): Original baseline ==========
    configs['3layer_1-1-1'] = {
        'num_layers': 3,
        'nodes_per_layer': [1, 1, 1],
        'layer_configs': [
            # Layer 0: Client
            {'memory': 17, 'models': [0, 2, 3, 5, 6], 'gamma': 0},
            # Layer 1: Edge (receives from 1 client)
            {'memory': 85, 'models': [1, 4, 7], 'gamma': 0.5},
            # Layer 2: Cloud (receives from 1 edge)
            {'gamma': 0.5},
        ],
        'description': '3-layer: 1 client → 1 edge → cloud'
    }

    # ========== 3-Layer (4-2-1): Multiple clients per edge ==========
    # Structure: 4 clients, 2 edges (each edge has 2 clients), 1 cloud
    configs['3layer_4-2-1'] = {
        'num_layers': 3,
        'nodes_per_layer': [4, 2, 1],
        'layer_configs': [
            # Layer 0: Clients (4 nodes, each with small memory)
            {'memory': 17, 'models': [0, 2, 3, 5, 6], 'gamma': 0},
            # Layer 1: Edges (2 nodes, each receives from 2 clients)
            # gamma = 0.5 * 2 = 1.0 to handle 2x traffic, or keep 0.5 per-client budget
            {'memory': 85, 'models': [1, 4, 7], 'gamma': 0.5},
            # Layer 2: Cloud (receives from 2 edges)
            # gamma = 0.5 * 2 = 1.0 to handle aggregated edge traffic
            {'gamma': 0.5},
        ],
        'description': '3-layer: 4 clients → 2 edges → cloud (2 clients per edge)'
    }

    # ========== 4-Layer (1-1-1-1): Single chain ==========
    # Structure: client → edge → regional → cloud
    configs['4layer_1-1-1-1'] = {
        'num_layers': 4,
        'nodes_per_layer': [1, 1, 1, 1],
        'layer_configs': [
            # Layer 0: Client (smallest devices, ~8GB)
            {'memory': 8, 'models': [0, 5, 6], 'gamma': 0},
            # Layer 1: Edge server (~20GB)
            {'memory': 20, 'models': [0, 2, 5, 6], 'gamma': 0.4},
            # Layer 2: Regional server (~85GB)
            {'memory': 85, 'models': [1, 3, 4, 7], 'gamma': 0.4},
            # Layer 3: Cloud
            {'gamma': 0.4},
        ],
        'description': '4-layer: 1 client → 1 edge → 1 regional → cloud'
    }

    # ========== 4-Layer (8-4-2-1): Full hierarchical tree ==========
    # Structure: 8 clients → 4 edges → 2 regional → 1 cloud
    # Each node has 2 children
    configs['4layer_8-4-2-1'] = {
        'num_layers': 4,
        'nodes_per_layer': [8, 4, 2, 1],
        'layer_configs': [
            # Layer 0: Clients (8 nodes, smallest devices)
            {'memory': 8, 'models': [0, 5, 6], 'gamma': 0},
            # Layer 1: Edge servers (4 nodes, each receives from 2 clients)
            {'memory': 20, 'models': [0, 2, 5, 6], 'gamma': 0.4},
            # Layer 2: Regional servers (2 nodes, each receives from 2 edges)
            {'memory': 85, 'models': [1, 3, 4, 7], 'gamma': 0.4},
            # Layer 3: Cloud (receives from 2 regional servers)
            {'gamma': 0.4},
        ],
        'description': '4-layer: 8 clients → 4 edges → 2 regional → cloud'
    }

    return configs


# ============================================================
# NODE AND SYSTEM CLASSES (same as before)
# ============================================================

class HierarchicalNode:
    """Node in hierarchical system"""

    def __init__(self, node_id, level, memory_capacity, available_models, cost_budget_gamma):
        self.node_id = node_id
        self.level = level
        self.memory_capacity = memory_capacity
        self.available_models = available_models
        self.cost_budget_gamma = cost_budget_gamma

        self.children = []
        self.parent = None
        self.onloaded_models = []

        self.Q = None
        self.w = None
        self.S = None

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
        """Build from top (cloud) to bottom (clients)"""
        layer_nodes = []

        for layer_idx in range(self.num_layers - 1, -1, -1):
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
                    cost_budget_gamma=config.get('gamma', 0)
                )

                if layer_idx > 0:
                    node.Q = 0.0

                self.nodes[node_id] = node
                current_layer_nodes.append(node)

                if layer_idx < self.num_layers - 1 and layer_nodes:
                    parent_idx = i % len(layer_nodes)
                    layer_nodes[parent_idx].add_child(node)

            layer_nodes = current_layer_nodes

    def get_leaf_nodes(self):
        return [n for n in self.nodes.values() if n.level == 0]

    def get_cloud_node(self):
        return [n for n in self.nodes.values() if n.level == self.num_layers - 1][0]

    def get_non_cloud_nodes(self):
        return [n for n in self.nodes.values() if n.level < self.num_layers - 1]

    def print_structure(self):
        """Print the tree structure"""

        def print_node(node, indent=0):
            prefix = "  " * indent
            if node.level == self.num_layers - 1:
                print(f"{prefix}{node.node_id} (Cloud)")
            else:
                mem_used = sum(MODEL_SIZES[m] for m in node.onloaded_models) if node.onloaded_models else 0
                print(
                    f"{prefix}{node.node_id} (L{node.level}, mem={node.memory_capacity}GB, gamma={node.cost_budget_gamma})")
            for child in node.children:
                print_node(child, indent + 1)

        cloud = self.get_cloud_node()
        print_node(cloud)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def initialize_onloaded_models(memory_capacity, model_ids):
    """Initialize models that fit in memory"""
    S = set()
    remaining_capacity = memory_capacity
    available_models = list(model_ids)
    np.random.shuffle(available_models)
    for model_id in available_models:
        if MODEL_SIZES[model_id] <= remaining_capacity:
            S.add(model_id)
            remaining_capacity -= MODEL_SIZES[model_id]
    return list(S)


def enhanced_greedy_onloading(V, prev_onloaded_models, memory_capacity, model_sizes,
                              onloading_costs, error_rates_dict, task_dist_estimate,
                              available_models, task_names):
    """Greedy knapsack for onloading"""
    S = set()
    remaining_capacity = memory_capacity
    prev_onloaded_set = set(prev_onloaded_models)

    while True:
        best_model, best_ratio = -1, -np.inf
        candidate_models = []

        for idx, model_id in enumerate(available_models):
            if model_id not in S and model_sizes[idx] <= remaining_capacity:
                candidate_models.append((idx, model_id))

        if not candidate_models:
            break

        expected_error_S = []
        for t_idx, task_type in enumerate(task_names):
            if not S:
                expected_error_S.append(1.0)
            else:
                errors_for_task = [error_rates_dict[task_type][m]
                                   for m in S if m in error_rates_dict[task_type]]
                expected_error_S.append(min(errors_for_task) if errors_for_task else 1.0)

        for idx, model_id in candidate_models:
            sum_task_acc_gain = 0.0
            S_plus_m = S.union({model_id})

            for t_idx, task_type in enumerate(task_names):
                errors_for_task_plus_m = [error_rates_dict[task_type][m]
                                          for m in S_plus_m if m in error_rates_dict[task_type]]
                error_S_plus_m = min(errors_for_task_plus_m) if errors_for_task_plus_m else 1.0
                accuracy_gain = V * task_dist_estimate[t_idx] * (expected_error_S[t_idx] - error_S_plus_m)
                sum_task_acc_gain += accuracy_gain

            cost_increase = onloading_costs[idx] if model_id not in prev_onloaded_set else 0
            marginal_gain = sum_task_acc_gain - cost_increase

            if marginal_gain > 0:
                ratio = marginal_gain / model_sizes[idx]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_model = model_id
                    best_idx = idx

        if best_model == -1:
            break

        S.add(best_model)
        remaining_capacity -= model_sizes[best_idx]

    total_cost = sum(onloading_costs[idx] for idx, model_id in enumerate(available_models)
                     if model_id in S and model_id not in prev_onloaded_set)

    return list(S), total_cost


# ============================================================
# MAIN ALGORITHM
# ============================================================

def multi_layer_exp4(system, num_jobs, num_experts, learning_rate, v_param,
                     use_variance_reduction=False, enable_onloading=True,
                     initial_error_rates=None):
    """
    Multi-layer EXP4 with corrected variance reduction.
    """
    expert_thresholds = np.linspace(0, 1, num_experts)

    if initial_error_rates is None:
        estimated_error_rates = copy.deepcopy(ERROR_RATES)
    else:
        estimated_error_rates = copy.deepcopy(initial_error_rates)

    ERROR_LEARNING_RATE = 0.05
    MAX_IPS_WEIGHT = 20.0

    # Initialize all non-cloud nodes
    for node in system.get_non_cloud_nodes():
        node.S = np.zeros((NUM_TASK_TYPES, num_experts))
        node.w = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts

        if enable_onloading and len(node.available_models) > 0:
            node.onloaded_models = initialize_onloaded_models(
                node.memory_capacity, node.available_models)
        else:
            node.onloaded_models = list(node.available_models)

    task_counts = {t: 0 for t in TASK_NAMES}
    node_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    node_offload_probs = {n.node_id: {t: 0.5 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_offload_sum = {n.node_id: {t: 0.0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}

    if use_variance_reduction:
        avg_cost_estimate = {t: avg_offloadCost[t] for t in TASK_NAMES}
        cost_obs_count = {t: 2000 for t in TASK_NAMES}

    history = {
        'errors': [], 'costs_total': [],
        'node_costs': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'node_queues': {n.node_id: [] for n in system.nodes.values() if n.Q is not None},
        'node_offload_decisions': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'feedback_received': [],
        'onload_costs': [],
        'execution_layer': [],
    }

    # Main loop
    for j in range(num_jobs):

        if j % 5000 == 0 and j > 0:
            recent_errors = np.mean(history['errors'][max(0, j - 1000):j])
            recent_feedback = np.mean(history['feedback_received'][max(0, j - 1000):j])
            recent_cost = np.mean(history['costs_total'][max(0, j - 1000):j])
            print(f"  Job {j}: error={recent_errors:.3f}, feedback={recent_feedback:.3f}, cost={recent_cost:.3f}")

        # Periodic onloading
        if enable_onloading and j > 0 and j % ONLOADING_EPOCH_LENGTH == 0:
            total_onload_cost = 0

            for node in system.get_non_cloud_nodes():
                if len(node.available_models) == 0:
                    continue

                total_tasks = sum(node_task_counts[node.node_id].values())
                task_dist_array = np.array([
                    node_task_counts[node.node_id][t] / (total_tasks + 1e-9)
                    for t in TASK_NAMES
                ])

                for t in TASK_NAMES:
                    if epoch_task_counts[node.node_id][t] > 0:
                        node_offload_probs[node.node_id][t] = \
                            epoch_offload_sum[node.node_id][t] / epoch_task_counts[node.node_id][t]

                task_dist = np.array([
                    task_dist_array[i] * (1 - node_offload_probs[node.node_id][TASK_NAMES[i]])
                    for i in range(NUM_TASK_TYPES)
                ])

                model_sizes_node = MODEL_SIZES[node.available_models]
                onload_costs = MODEL_ONLOADING_COSTS[node.available_models]

                new_onloaded, cost = enhanced_greedy_onloading(
                    V=V_ONLOAD,
                    prev_onloaded_models=node.onloaded_models,
                    memory_capacity=node.memory_capacity,
                    model_sizes=model_sizes_node,
                    onloading_costs=onload_costs,
                    error_rates_dict=estimated_error_rates,
                    task_dist_estimate=task_dist,
                    available_models=node.available_models,
                    task_names=TASK_NAMES
                )

                node.onloaded_models = new_onloaded
                total_onload_cost += cost

            history['onload_costs'].append(total_onload_cost)

            for node in system.get_non_cloud_nodes():
                epoch_offload_sum[node.node_id] = {t: 0.0 for t in TASK_NAMES}
                epoch_task_counts[node.node_id] = {t: 0 for t in TASK_NAMES}
        else:
            history['onload_costs'].append(0)

        # Select job
        idx = random.randint(0, TOTAL_JOBS - 1)
        task_type = data['full_data'][idx]['category']
        task_type_idx = TASK_NAMES.index(task_type)
        task_counts[task_type] += 1

        # Start at random leaf
        leaf_nodes = system.get_leaf_nodes()
        current_node = random.choice(leaf_nodes)

        # Track path and decisions
        path = [current_node]
        offload_decisions = {}
        confidences = {}
        offload_probs = {}
        best_models = {}
        node_err_estimates = {}

        executed_at_node = None

        while current_node is not None:
            if current_node.level == system.num_layers - 1:
                executed_at_node = current_node
                break

            node_task_counts[current_node.node_id][task_type] += 1
            epoch_task_counts[current_node.node_id][task_type] += 1

            if len(current_node.onloaded_models) == 0:
                node_err_est = 1.0
                best_model = None
            else:
                node_err_est, best_model = lowest_avg_error(
                    estimated_error_rates[task_type],
                    current_node.onloaded_models
                )
                if node_err_est is None:
                    node_err_est = 1.0

            best_models[current_node.node_id] = best_model
            node_err_estimates[current_node.node_id] = node_err_est

            mean_confidence = 1.0 - node_err_est
            confidence_Z = np.random.normal(loc=mean_confidence, scale=0.1)
            confidence_Z = np.clip(confidence_Z, 0, 1)
            confidences[current_node.node_id] = confidence_Z

            offload_experts = expert_thresholds > confidence_Z
            prob_offload = np.sum(current_node.w[task_type_idx, offload_experts])
            prob_offload = np.clip(prob_offload, 1e-5, 1 - 1e-5)
            offload_probs[current_node.node_id] = prob_offload
            epoch_offload_sum[current_node.node_id][task_type] += prob_offload

            o = 1 if np.random.rand() < prob_offload else 0
            offload_decisions[current_node.node_id] = o

            if o == 0:
                executed_at_node = current_node
                break
            else:
                current_node = current_node.parent
                if current_node is not None:
                    path.append(current_node)
                else:
                    executed_at_node = system.get_cloud_node()
                    break

        if executed_at_node is None:
            executed_at_node = system.get_cloud_node()

        # Determine outcome
        if executed_at_node.level == system.num_layers - 1:
            job_error = 0
        else:
            if len(executed_at_node.onloaded_models) == 0 or best_models.get(executed_at_node.node_id) is None:
                job_error = 1
            else:
                best_model = best_models[executed_at_node.node_id]
                job_error = int(1 - data['full_data'][idx]['results'][best_model])

        feedback_received = (executed_at_node.level == system.num_layers - 1)

        # Compute node-specific probability to cloud
        node_prob_to_cloud = {}
        if feedback_received:
            for path_idx, node in enumerate(path[:-1]):
                prob_to_cloud = 1.0
                for i in range(path_idx, len(path) - 1):
                    prob_to_cloud *= offload_probs[path[i].node_id]
                node_prob_to_cloud[node.node_id] = max(prob_to_cloud, 1e-9)

        joint_prob = node_prob_to_cloud.get(path[0].node_id, 1.0) if path and feedback_received else 1.0

        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)

        # IPS error rate update
        if feedback_received:
            ips_weight = min(1.0 / (joint_prob + 1e-9), MAX_IPS_WEIGHT)

            for node in path[:-1]:
                if len(node.onloaded_models) == 0:
                    continue
                best_model_node = best_models.get(node.node_id)
                if best_model_node is None:
                    continue

                b_true = int(1 - data['full_data'][idx]['results'][best_model_node])
                old_est = estimated_error_rates[task_type][best_model_node]
                new_est = (1 - ERROR_LEARNING_RATE) * old_est + ERROR_LEARNING_RATE * (b_true * ips_weight)
                estimated_error_rates[task_type][best_model_node] = np.clip(new_est, 0.0, 1.0)

        if use_variance_reduction and feedback_received:
            count = cost_obs_count[task_type]
            avg_cost_estimate[task_type] = (avg_cost_estimate[task_type] * count + C_c) / (count + 1)
            cost_obs_count[task_type] += 1

        # Determine nodes to update
        if use_variance_reduction:
            nodes_to_update = [n for n in path if n.level < system.num_layers - 1]
        elif feedback_received:
            nodes_to_update = [n for n in path if n.level < system.num_layers - 1]
        else:
            nodes_to_update = []

        # Update EXP4 weights
        for path_idx, node in enumerate(nodes_to_update):
            node_path_idx = path.index(node)

            L_hat = np.zeros(num_experts)
            p_node_to_cloud = node_prob_to_cloud.get(node.node_id, 1.0)
            b_hat = node_err_estimates.get(node.node_id, 1.0)
            parent_node = path[node_path_idx + 1] if node_path_idx + 1 < len(path) else None

            for a in range(num_experts):
                o_hat_a = 1 if expert_thresholds[a] > confidences[node.node_id] else 0

                if o_hat_a == 0:
                    if use_variance_reduction:
                        if feedback_received:
                            best_model_node = best_models.get(node.node_id)
                            b_true = int(
                                1 - data['full_data'][idx]['results'][best_model_node]) if best_model_node else 1
                            residual = (b_true - b_hat) / p_node_to_cloud
                            L_hat[a] = v_param * (residual + b_hat)
                        else:
                            L_hat[a] = v_param * b_hat
                    elif feedback_received:
                        best_model_node = best_models.get(node.node_id)
                        b_true = int(1 - data['full_data'][idx]['results'][best_model_node]) if best_model_node else 1
                        L_hat[a] = v_param * b_true / p_node_to_cloud
                else:
                    if parent_node is None:
                        L_hat[a] = 0.0
                        continue

                    Q_parent = parent_node.Q if parent_node.Q is not None else 0.0

                    if parent_node.level == system.num_layers - 1:
                        if use_variance_reduction:
                            C_hat = avg_cost_estimate[task_type]
                            if feedback_received:
                                residual_cost = (C_c - C_hat) / p_node_to_cloud
                                L_hat[a] = Q_parent * (residual_cost + C_hat)
                            else:
                                L_hat[a] = Q_parent * C_hat
                        elif feedback_received:
                            L_hat[a] = Q_parent * C_c / p_node_to_cloud
                    else:
                        parent_prob_offload = offload_probs.get(parent_node.node_id, 0.5)
                        b_hat_parent = node_err_estimates.get(parent_node.node_id, 1.0)
                        grandparent_node = parent_node.parent
                        Q_grandparent = grandparent_node.Q if (
                                    grandparent_node and grandparent_node.Q is not None) else 0.0

                        if use_variance_reduction:
                            C_hat = avg_cost_estimate[task_type]
                            L_parent_hat = v_param * (1 - parent_prob_offload) * b_hat_parent + \
                                           parent_prob_offload * Q_grandparent * C_hat
                            L_offload_hat = Q_parent * C_hat + L_parent_hat

                            if feedback_received:
                                best_model_parent = best_models.get(parent_node.node_id)
                                b_true_parent = int(1 - data['full_data'][idx]['results'][
                                    best_model_parent]) if best_model_parent else 1
                                L_parent_actual = v_param * (1 - parent_prob_offload) * b_true_parent + \
                                                  parent_prob_offload * Q_grandparent * C_c
                                L_offload_actual = Q_parent * C_c + L_parent_actual
                                residual = (L_offload_actual - L_offload_hat) / p_node_to_cloud
                                L_hat[a] = residual + L_offload_hat
                            else:
                                L_hat[a] = L_offload_hat
                        elif feedback_received:
                            best_model_parent = best_models.get(parent_node.node_id)
                            b_true_parent = int(
                                1 - data['full_data'][idx]['results'][best_model_parent]) if best_model_parent else 1
                            L_parent = v_param * (1 - parent_prob_offload) * b_true_parent + \
                                       parent_prob_offload * Q_grandparent * C_c
                            L_offload = Q_parent * C_c + L_parent
                            L_hat[a] = L_offload / p_node_to_cloud

            node.S[task_type_idx] += L_hat

            log_w = -learning_rate * node.S[task_type_idx]
            log_w_shifted = log_w - np.max(log_w)
            node.w[task_type_idx] = np.exp(log_w_shifted) / np.sum(np.exp(log_w_shifted))

        # Update queues
        path_node_ids = set(n.node_id for n in path[:-1])
        total_job_cost = 0

        for node in system.get_non_cloud_nodes():
            if node.node_id in path_node_ids and offload_decisions.get(node.node_id, 0) == 1:
                job_cost = C_c
                if node.parent and node.parent.Q is not None:
                    node.parent.Q = max(0, node.parent.Q + job_cost - node.parent.cost_budget_gamma)
                history['node_costs'][node.node_id].append(job_cost)
                total_job_cost += job_cost
            else:
                history['node_costs'][node.node_id].append(0)

            history['node_offload_decisions'][node.node_id].append(offload_decisions.get(node.node_id, 0))

        for node in system.nodes.values():
            if node.Q is not None:
                history['node_queues'][node.node_id].append(node.Q)

        history['errors'].append(job_error)
        history['costs_total'].append(total_job_cost)
        history['feedback_received'].append(int(feedback_received))
        history['execution_layer'].append(executed_at_node.level)

    return history


def aggregate_results(all_histories):
    """Average results over trials"""
    aggregated = {}
    keys = all_histories[0].keys()
    skip_keys = {'node_costs', 'node_queues', 'node_offload_decisions'}

    for key in keys:
        if key in skip_keys:
            aggregated[key] = {}
            for node_id in all_histories[0][key].keys():
                stacked = np.array([h[key][node_id] for h in all_histories])
                aggregated[key][node_id] = {
                    'mean': np.mean(stacked, axis=0),
                    'std': np.std(stacked, axis=0)
                }
            continue

        try:
            stacked = np.array([h[key] for h in all_histories])
            aggregated[f'{key}_mean'] = np.mean(stacked, axis=0)
            aggregated[f'{key}_std'] = np.std(stacked, axis=0)
        except (ValueError, TypeError):
            aggregated[key] = all_histories[0][key]

    return aggregated


def plot_multi_layer_results(agg_no_vr, agg_vr, system, num_jobs, save_prefix='results'):
    """Plot results"""
    jobs_axis = np.arange(num_jobs)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Multi-Layer Performance ({system.num_layers} Layers, {system.nodes_per_layer})', fontsize=16)

    # Error Rate
    avg_error_no_vr = np.cumsum(agg_no_vr['errors_mean']) / (jobs_axis + 1)
    avg_error_vr = np.cumsum(agg_vr['errors_mean']) / (jobs_axis + 1)
    axes[0, 0].plot(jobs_axis, avg_error_no_vr, label='No VR')
    axes[0, 0].plot(jobs_axis, avg_error_vr, label='With VR', linestyle='--')
    axes[0, 0].set_ylabel('Average Error Rate')
    axes[0, 0].set_xlabel('Jobs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Feedback Rate
    avg_feedback_no_vr = np.cumsum(agg_no_vr['feedback_received_mean']) / (jobs_axis + 1)
    avg_feedback_vr = np.cumsum(agg_vr['feedback_received_mean']) / (jobs_axis + 1)
    axes[0, 1].plot(jobs_axis, avg_feedback_no_vr, label='No VR')
    axes[0, 1].plot(jobs_axis, avg_feedback_vr, label='With VR', linestyle='--')
    axes[0, 1].set_ylabel('Feedback Rate')
    axes[0, 1].set_xlabel('Jobs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # # Total Cost with budget
    # avg_cost_no_vr = np.cumsum(agg_no_vr['costs_total_mean']) / (jobs_axis + 1)
    # avg_cost_vr = np.cumsum(agg_vr['costs_total_mean']) / (jobs_axis + 1)
    # axes[0, 2].plot(jobs_axis, avg_cost_no_vr, label='No VR')
    # axes[0, 2].plot(jobs_axis, avg_cost_vr, label='With VR', linestyle='--')
    #
    # total_budget = sum(node.cost_budget_gamma for node in system.nodes.values() if node.Q is not None)
    # axes[0, 2].axhline(y=total_budget, color='r', linestyle=':', label=f'Total Budget={total_budget:.2f}')
    # axes[0, 2].set_ylabel('Average Total Cost')
    # axes[0, 2].set_xlabel('Jobs')
    # axes[0, 2].legend()
    # axes[0, 2].grid(True)

    # Per-node costs (limit legend to first 6 nodes)
    node_ids = list(agg_no_vr['node_costs'].keys())
    for i, node_id in enumerate(node_ids[:6]):
        avg_cost_node = np.cumsum(agg_no_vr['node_costs'][node_id]['mean']) / (jobs_axis + 1)
        axes[1, 0].plot(jobs_axis, avg_cost_node, label=f'{node_id}', alpha=0.7)
    if len(node_ids) > 6:
        axes[1, 0].set_title(f'Showing 6/{len(node_ids)} nodes')
    axes[1, 0].set_ylabel('Avg Cost per Node')
    axes[1, 0].set_xlabel('Jobs')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True)

    # # Queue sizes (limit legend)
    # queue_ids = list(agg_no_vr['node_queues'].keys())
    # for i, node_id in enumerate(queue_ids[:6]):
    #     axes[1, 1].plot(jobs_axis, agg_no_vr['node_queues'][node_id]['mean'],
    #                     label=f'{node_id}', alpha=0.7)
    # axes[1, 1].set_ylabel('Virtual Queue Size')
    # axes[1, 1].set_xlabel('Jobs')
    # axes[1, 1].legend(fontsize=8)
    # axes[1, 1].grid(True)

    # Execution distribution
    exec_layer_no_vr = np.bincount(agg_no_vr['execution_layer_mean'].astype(int),
                                   minlength=system.num_layers)
    exec_layer_vr = np.bincount(agg_vr['execution_layer_mean'].astype(int),
                                minlength=system.num_layers)

    x = np.arange(system.num_layers)
    width = 0.35

    axes[1, 1].bar(x - width / 2, exec_layer_no_vr, width, label='No VR')
    axes[1, 1].bar(x + width / 2, exec_layer_vr, width, label='With VR')
    axes[1, 1].set_ylabel('Number of Jobs Executed')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'L{i}' for i in range(system.num_layers)])
    axes[1, 1].set_title('Job Execution Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()

    filename = f'{save_prefix}_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filename}")

    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    NUM_TRIALS = 3
    NUM_JOBS = 40000
    NUM_EXPERTS = 50
    LEARNING_RATE_ETA = 0.01
    V_PARAM = 500

    configs = get_system_configs()

    # Select which configurations to run
    configs_to_run = [
        '3layer_1-1-1',
        '3layer_4-2-1',
        '4layer_1-1-1-1',
        '4layer_8-4-2-1',
    ]

    for config_name in configs_to_run:
        config = configs[config_name]

        print("\n" + "=" * 70)
        print(f"Running: {config_name}")
        print(f"Description: {config['description']}")
        print("=" * 70)

        system = HierarchicalSystem(
            num_layers=config['num_layers'],
            nodes_per_layer=config['nodes_per_layer'],
            layer_configs=config['layer_configs']
        )

        print("\nSystem Structure:")
        system.print_structure()

        all_results_no_vr, all_results_vr = [], []

        for i in range(NUM_TRIALS):
            print(f"\nTrial {i + 1}/{NUM_TRIALS}")

            # Reset system state for each trial
            system = HierarchicalSystem(
                num_layers=config['num_layers'],
                nodes_per_layer=config['nodes_per_layer'],
                layer_configs=config['layer_configs']
            )

            results_no_vr = multi_layer_exp4(
                system=system,
                num_jobs=NUM_JOBS,
                num_experts=NUM_EXPERTS,
                learning_rate=LEARNING_RATE_ETA,
                v_param=V_PARAM,
                use_variance_reduction=False,
                enable_onloading=True,
                initial_error_rates=Avg_err
            )
            all_results_no_vr.append(results_no_vr)

            # Reset system for VR run
            system = HierarchicalSystem(
                num_layers=config['num_layers'],
                nodes_per_layer=config['nodes_per_layer'],
                layer_configs=config['layer_configs']
            )

            results_vr = multi_layer_exp4(
                system=system,
                num_jobs=NUM_JOBS,
                num_experts=NUM_EXPERTS,
                learning_rate=LEARNING_RATE_ETA,
                v_param=V_PARAM,
                use_variance_reduction=True,
                enable_onloading=True,
                initial_error_rates=Avg_err
            )
            all_results_vr.append(results_vr)

        agg_no_vr = aggregate_results(all_results_no_vr)
        agg_vr = aggregate_results(all_results_vr)

        plot_multi_layer_results(agg_no_vr, agg_vr, system, NUM_JOBS, save_prefix=config_name)

        # Print final statistics
        print(f"\n--- Final Results for {config_name} ---")
        print(f"Without VR - Final avg error: {np.mean(agg_no_vr['errors_mean'][-2000:]):.4f}")
        print(f"With VR    - Final avg error: {np.mean(agg_vr['errors_mean'][-2000:]):.4f}")
        print(f"Without VR - Final avg feedback: {np.mean(agg_no_vr['feedback_received_mean'][-2000:]):.4f}")
        print(f"With VR    - Final avg feedback: {np.mean(agg_vr['feedback_received_mean'][-2000:]):.4f}")
        print(f"Without VR - Final avg cost: {np.mean(agg_no_vr['costs_total_mean'][-2000:]):.4f}")
        print(f"With VR    - Final avg cost: {np.mean(agg_vr['costs_total_mean'][-2000:]):.4f}")

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)
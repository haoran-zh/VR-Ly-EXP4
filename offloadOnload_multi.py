import numpy as np
import matplotlib.pyplot as plt
from utilities.model_performance import model_perforance_GT, estimate_model_perforance, convert_offloadingCost, estimate_offloadingCost, \
    lowest_avg_error
from utilities.system import HierarchicalNode, HierarchicalSystem
import pickle as pkl
import random
import copy

# read dataset
FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
with open(FILENAME, 'rb') as f:
    data = pkl.load(f)

Avg_acc, Avg_err = estimate_model_perforance(data)
ERROR_RATES = Avg_err

GT_acc, GT_err = model_perforance_GT(data)
ERROR_RATES_GT = GT_err

OffloadCost_SCALE = 0.001
avg_offloadCost = estimate_offloadingCost(data, OffloadCost_SCALE)

NUM_TASK_TYPES = len(data['task_types'])
TASK_NAMES = data['task_types']
TOTAL_JOBS = data['total_samples']

MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])
MODEL_ONLOADING_COSTS = MODEL_SIZES
ONLOADING_EPOCH_LENGTH = 500
V_ONLOAD = 700
BEST_MODEL_IDX=7

# error rate if all jobs were executed by a specific model
def compute_baseline_error_rate(model_idx=BEST_MODEL_IDX):
    """
    Compute the average error rate if all jobs were executed by a specific model.

    Args:
        model_idx: Index of the model (default: 7 = Qwen2-72B)

    Returns:
        overall_error: Average error rate across all jobs
        per_task_error: Dict of error rates per task type
    """
    total_errors = 0
    total_jobs = 0
    per_task_errors = {t: 0 for t in TASK_NAMES}
    per_task_counts = {t: 0 for t in TASK_NAMES}

    for idx in range(TOTAL_JOBS):
        task_type = data['full_data'][idx]['category']
        # result = 1 means correct, 0 means incorrect
        result = data['full_data'][idx]['results'][model_idx]
        error = 1 - result  # error = 1 if incorrect, 0 if correct

        total_errors += error
        total_jobs += 1
        per_task_errors[task_type] += error
        per_task_counts[task_type] += 1

    overall_error = total_errors / total_jobs
    per_task_error = {t: per_task_errors[t] / per_task_counts[t] if per_task_counts[t] > 0 else 0
                      for t in TASK_NAMES}

    return overall_error, per_task_error


def get_system_configs():
    configs = {}
    configs['3layer_1-1-1'] = {
        'num_layers': 3, 'nodes_per_layer': [1, 1, 1],
        'layer_configs': [
            {'memory': 17, 'models': [0, 2, 3, 5, 6], 'gamma': 0},
            {'memory': 85, 'models': [1, 4, 7], 'gamma': 0.4},
            {'gamma': 0.4},
        ],
    }
    configs['3layer_4-2-1'] = {
        'num_layers': 3, 'nodes_per_layer': [4, 2, 1],
        'layer_configs': [
            {'memory': 17, 'models': [0, 2, 3, 5, 6], 'gamma': 0},
            {'memory': 85, 'models': [1, 4, 7], 'gamma': 0.4},
            {'gamma': 0.4},
        ],
    }
    configs['4layer_1-1-1-1'] = {
        'num_layers': 4, 'nodes_per_layer': [1, 1, 1, 1],
        'layer_configs': [
            {'memory': 8, 'models': [0, 5, 6], 'gamma': 0},
            {'memory': 60, 'models': [0, 2, 5, 6], 'gamma': 0.4},
            {'memory': 85, 'models': [1, 3, 4, 7], 'gamma': 0.4},
            {'gamma': 0.4},
        ],
    }
    configs['4layer_8-4-2-1'] = {
        'num_layers': 4, 'nodes_per_layer': [8, 4, 2, 1],
        'layer_configs': [
            {'memory': 8, 'models': [0, 5, 6], 'gamma': 0},
            {'memory': 60, 'models': [0, 2, 5, 6], 'gamma': 0.4},
            {'memory': 85, 'models': [1, 3, 4, 7], 'gamma': 0.4},
            {'gamma': 0.4},
        ],
    }
    return configs



def initialize_onloaded_models(memory_capacity, model_ids):
    S = set()
    remaining = memory_capacity
    models = list(model_ids)
    np.random.shuffle(models)
    for m in models:
        if MODEL_SIZES[m] <= remaining:
            S.add(m)
            remaining -= MODEL_SIZES[m]
    return list(S)


def enhanced_greedy_onloading(V, prev_models, memory, model_sizes, onload_costs,
                              error_dict, task_dist, available_models, task_names):
    S = set()
    remaining = memory
    prev_set = set(prev_models)
    while True:
        best_model, best_ratio = -1, -np.inf
        candidates = [(i, m) for i, m in enumerate(available_models)
                      if m not in S and model_sizes[i] <= remaining]
        if not candidates:
            break
        exp_err = []
        for t in task_names:
            if not S:
                exp_err.append(1.0)
            else:
                errs = [error_dict[t][m] for m in S if m in error_dict[t]]
                exp_err.append(min(errs) if errs else 1.0)
        for i, m in candidates:
            gain = 0.0
            S_plus = S.union({m})
            for ti, t in enumerate(task_names):
                errs_plus = [error_dict[t][x] for x in S_plus if x in error_dict[t]]
                err_plus = min(errs_plus) if errs_plus else 1.0
                gain += V * task_dist[ti] * (exp_err[ti] - err_plus)
            cost = onload_costs[i] if m not in prev_set else 0
            net = gain - cost
            if net > 0 and net / model_sizes[i] > best_ratio:
                best_ratio = net / model_sizes[i]
                best_model, best_idx = m, i
        if best_model == -1:
            break
        S.add(best_model)
        remaining -= model_sizes[best_idx]
    total = sum(onload_costs[i] for i, m in enumerate(available_models) if m in S and m not in prev_set)
    return list(S), total


def multi_layer_exp4(system, num_jobs, num_experts, learning_rate, v_param,
                     use_variance_reduction=False, enable_onloading=True, initial_error_rates=None):
    """
    Multi-layer EXP4 with CORRECTED:
    1. Full path computation (leaf to cloud) BEFORE decisions
    2. Node-specific probability to cloud for importance sampling
    3. Proper baseline computation for VR
    """
    expert_thresholds = np.linspace(0, 1, num_experts)


    ERROR_LEARNING_RATE = 0.05
    MAX_IPS_WEIGHT = 20.0

    # Initialize nodes
    for node in system.get_non_cloud_nodes():
        node.S = np.zeros((NUM_TASK_TYPES, num_experts))  # accumulated loss
        node.w = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts  # weights for each expert
        if enable_onloading and len(node.available_models) > 0:
            node.onloaded_models = initialize_onloaded_models(node.memory_capacity, node.available_models)
        else:
            node.onloaded_models = list(node.available_models)

    task_counts = {t: 0 for t in TASK_NAMES}
    node_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    node_offload_probs = {n.node_id: {t: 0.5 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_offload_sum = {n.node_id: {t: 0.0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}

    estimated_error_rates_per_node = {n.node_id: copy.deepcopy(initial_error_rates) for n in system.get_non_cloud_nodes()}
    # question: what's the difference between node_task_counts and epoch_task_counts

    # Variance reduction control variants setup
    if use_variance_reduction:
        avg_cost_estimate = {t: avg_offloadCost[t] for t in TASK_NAMES}
        cost_obs_count = {t: 2000 for t in TASK_NAMES}  # how many samples we use to estimate the avg cost

    history = {
        'errors': [], 'costs_total': [],
        'node_costs': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'node_queues': {n.node_id: [] for n in system.nodes.values() if n.Q is not None},
        'node_offload_decisions': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'feedback_received': [], 'onload_costs': [], 'execution_layer': [],
        # Track loss value per update for each node (only when node is updated)
        # Each entry is the sum of L_hat for that job
        'node_loss_values': {n.node_id: [] for n in system.get_non_cloud_nodes()},
    }

    for j in range(num_jobs):
        if j % 5000 == 0 and j > 0:
            print(f"  Job {j}: err={np.mean(history['errors'][-1000:]):.3f}, "
                  f"fb={np.mean(history['feedback_received'][-1000:]):.3f}")

        # Onloading part
        if enable_onloading and j > 0 and j % ONLOADING_EPOCH_LENGTH == 0:
            total_onload_cost = 0
            for node in system.get_non_cloud_nodes():
                if len(node.available_models) == 0:
                    continue
                total_tasks = sum(node_task_counts[node.node_id].values())
                task_dist = np.array([node_task_counts[node.node_id][t] / (total_tasks + 1e-9) for t in TASK_NAMES])
                for t in TASK_NAMES:
                    if epoch_task_counts[node.node_id][t] > 0:
                        node_offload_probs[node.node_id][t] = epoch_offload_sum[node.node_id][t] / \
                                                              epoch_task_counts[node.node_id][t]
                local_dist = np.array([task_dist[i] * (1 - node_offload_probs[node.node_id][TASK_NAMES[i]]) for i in
                                       range(NUM_TASK_TYPES)])
                new_models, cost = enhanced_greedy_onloading(
                    V_ONLOAD, node.onloaded_models, node.memory_capacity,
                    MODEL_SIZES[node.available_models], MODEL_ONLOADING_COSTS[node.available_models],
                    estimated_error_rates_per_node[node.node_id], local_dist, node.available_models, TASK_NAMES
                )
                node.onloaded_models = new_models
                total_onload_cost += cost
            history['onload_costs'].append(total_onload_cost)
            for node in system.get_non_cloud_nodes():
                epoch_offload_sum[node.node_id] = {t: 0.0 for t in TASK_NAMES}
                epoch_task_counts[node.node_id] = {t: 0 for t in TASK_NAMES}
        else:
            history['onload_costs'].append(0)
        # TODO: I just skipped reviewing onloading part, I need to come back later.

        # Select job
        idx = random.randint(0, TOTAL_JOBS - 1)  # decide job idx from all data
        task_type = data['full_data'][idx]['category']
        task_type_idx = TASK_NAMES.index(task_type)
        task_counts[task_type] += 1

        leaf_nodes = system.get_leaf_nodes()
        start_node = random.choice(leaf_nodes)

        # Build full path to cloud
        full_path = []
        node = start_node
        while node is not None:
            full_path.append(node)
            node = node.parent
        # full_path = [leaf, ..., cloud]

        # Pre-compute info for ALL nodes on path (except cloud)

        all_node_best_models = {}
        all_node_confidences = {}
        all_node_offload_probs = {}

        for node in full_path[:-1]:  # Exclude cloud
            # Error estimate
            if len(node.onloaded_models) == 0:  # if we don't have available models.
                # TODO: we will add Video datasets soon, in this part, need to modify.
                err_exp, best_model = 1.0, None
            else:
                err_exp, best_model = lowest_avg_error(ERROR_RATES_GT[task_type], node.onloaded_models)
                if err_exp is None:
                    err_exp = 1.0
            all_node_best_models[node.node_id] = best_model

            # Confidence (same for all nodes)
            confidence = np.clip(np.random.normal(1.0 - err_exp, 0.1), 0, 1)
            all_node_confidences[node.node_id] = confidence

            # Offload probability from current weights
            offload_experts = expert_thresholds > confidence
            prob = np.clip(np.sum(node.w[task_type_idx, offload_experts]), 1e-5, 1 - 1e-5)
            all_node_offload_probs[node.node_id] = prob
            node.p_offload = prob

        # ================================================================
        # Now do the actual traversal (decisions)
        # ================================================================
        current_node = start_node
        visited_path = [current_node]
        offload_decisions = {}
        executed_at_node = None

        while current_node is not None:
            if current_node.level == system.num_layers - 1:  # reach the cloud (if the system only has 1 layer)
                executed_at_node = current_node
                break

            node_task_counts[current_node.node_id][task_type] += 1
            epoch_task_counts[current_node.node_id][task_type] += 1
            # pseudoTODO: here the offload_sum is the prob sum (seems okay)
            epoch_offload_sum[current_node.node_id][task_type] += all_node_offload_probs[current_node.node_id]

            # Make decision
            prob_offload = all_node_offload_probs[current_node.node_id]
            o = 1 if np.random.rand() < prob_offload else 0
            offload_decisions[current_node.node_id] = o

            if o == 0:
                executed_at_node = current_node
                break
            else:
                current_node = current_node.parent
                if current_node is not None:
                    visited_path.append(current_node)
                else:
                    executed_at_node = system.get_cloud_node()
                    break


        # Determine outcome
        if executed_at_node.level == system.num_layers - 1:
            job_error = 0
        else:
            best_model = all_node_best_models.get(executed_at_node.node_id)
            job_error = 1 if best_model is None else int(1 - data['full_data'][idx]['results'][best_model])

        feedback_received = (executed_at_node.level == system.num_layers - 1)
        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)

        # ================================================================
        # Compute node-specific probability to cloud using FULL path
        # ================================================================
        node_prob_to_cloud = {}  # we only need this prob when the task offloads to the cloud
        for path_idx, node in enumerate(full_path[:-1]):  # Exclude cloud
            prob_to_cloud = 1.0
            for i in range(path_idx, len(full_path) - 1):
                prob_to_cloud *= all_node_offload_probs[full_path[i].node_id]
            node_prob_to_cloud[node.node_id] = max(prob_to_cloud, 1e-9)

        # IPS error rate update (only with feedback)
        # this is the update for control variants
        if feedback_received:
            joint_prob = node_prob_to_cloud[full_path[0].node_id]  # the whole path prob
            ips_weight = min(1.0 / (joint_prob + 1e-9), MAX_IPS_WEIGHT)

            for node in full_path[:-1]:
                if len(node.onloaded_models) == 0:
                    continue
                best_model = all_node_best_models.get(node.node_id)
                if best_model is None:
                    continue
                b_true = int(1 - data['full_data'][idx]['results'][best_model])
                old_est = estimated_error_rates_per_node[node.node_id][task_type][best_model]
                new_est = (1 - ERROR_LEARNING_RATE) * old_est + ERROR_LEARNING_RATE * (b_true * ips_weight)
                estimated_error_rates_per_node[node.node_id][task_type][best_model] = np.clip(new_est, 0.0, 1.0)

        # Update cost estimate for VR
        if use_variance_reduction:
            count = cost_obs_count[task_type]
            avg_cost_estimate[task_type] = (avg_cost_estimate[task_type] * count + C_c) / (count + 1)
            cost_obs_count[task_type] += 1

        # ================================================================
        # Update EXP4 weights
        # VR: Update ALL nodes on full_path (except cloud)
        # Standard: Only update visited nodes when feedback received
        # ================================================================
        if use_variance_reduction:
            nodes_to_update = full_path[-2::-1]  # All except cloud
        elif feedback_received:  # only without VR can reach this option
            nodes_to_update = full_path[-2::-1]
        else:  # if no feedback (without VR)
            nodes_to_update = []  # without VR, no node to update if no feedback
        # nodes_to_update is in reverse order, from parent to child

        for node in nodes_to_update:
            # path_idx = full_path.index(node)  # 5->4->3->2->1
            node.exp_loss = None
            node.actual_loss = np.zeros(num_experts)
            # This node's probability to cloud
            node.p_to_cloud = node_prob_to_cloud[node.node_id]
            # Baseline error estimate for this node (for VR)
            best_model = all_node_best_models.get(node.node_id)
            node.b_hat = estimated_error_rates_per_node[node.node_id][task_type][best_model]
            node.b_true = int(1 - data['full_data'][idx]['results'][best_model]) if feedback_received else node.b_hat
            node.confidence = all_node_confidences[node.node_id]
            node.C_true = C_c
            node.C_hat = avg_cost_estimate[task_type] if use_variance_reduction else C_c

            for a in range(num_experts):
                loss_recursive_exp(node, v_param, system, use_variance_reduction, if_feedback=feedback_received)
                loss_recursive_actual(node, v_param, system, a, expert_thresholds, use_variance_reduction, if_feedback=feedback_received)

            # Update cumulative loss
            node.S[task_type_idx] += node.actual_loss

            # Record the total loss for this job (sum of L_hat across all experts)
            # feels useless
            history['node_loss_values'][node.node_id].append(np.sum(node.actual_loss))

            # Normalize weights
            log_w = -learning_rate * node.S[task_type_idx]
            log_w_shifted = log_w - np.max(log_w)  # shifted w may be better
            node.w[task_type_idx] = np.exp(log_w_shifted) / np.sum(np.exp(log_w_shifted))
        # set back to None for temp values
        for node in nodes_to_update:
            node.b_true = None
            node.b_hat = None
            node.C_true = None
            node.C_hat = None
            node.exp_loss = None
            node.actual_loss = None
            node.p_to_cloud = None
            node.p_offload = None
            node.confidence = None

        # For layer 0 (starting) node: record 0 if it wasn't updated (no-VR without feedback)
        start_node_id = start_node.node_id
        updated_node_ids = set(n.node_id for n in nodes_to_update)
        if start_node_id not in updated_node_ids:
            # This happens for no-VR without feedback
            history['node_loss_values'][start_node_id].append(0.0)

        # Update queues
        visited_ids = set(n.node_id for n in visited_path[:-1]) if visited_path[
                                                                       -1].level == system.num_layers - 1 else set(
            n.node_id for n in visited_path)
        total_job_cost = 0

        for node in system.get_non_cloud_nodes():
            if node.node_id in visited_ids and offload_decisions.get(node.node_id, 0) == 1:
                if node.parent and node.parent.Q is not None:
                    node.parent.Q = max(0, node.parent.Q + C_c - node.parent.cost_budget_gamma)
                history['node_costs'][node.node_id].append(C_c)
                total_job_cost += C_c
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


def loss_recursive_exp(node, v_param, system, use_variance_reduction, if_feedback):
    # exp loss (do not use experts), for its child node
    parent = node.parent
    if use_variance_reduction is True:
        residual_b = (node.b_true - node.b_hat) / node.p_to_cloud * if_feedback
        residual_C = (node.C_true - node.C_hat) / node.p_to_cloud * if_feedback
        if parent.level == system.num_layers - 1: # end of the recurse, parent is the cloud
            node.exp_loss = (v_param * (1 - node.p_offload) * (residual_b + node.b_hat) +
                             node.p_offload * parent.Q * (residual_C + node.C_hat))
        else:
            node.exp_loss = (v_param * (1 - node.p_offload) * (residual_b + node.b_hat) +
                             node.p_offload * parent.Q * (residual_C + node.C_hat) + parent.exp_loss)
    else:  # without VR
        if parent.level == system.num_layers - 1: # end of the recurse, parent is the cloud
            node.exp_loss = (v_param * (1 - node.p_offload) * node.b_true
                             + node.p_offload * parent.Q * node.C_true)
        else:
            node.exp_loss = (v_param * (1 - node.p_offload) * node.b_true
                             + node.p_offload * parent.Q * node.C_true + parent.exp_loss)


def loss_recursive_actual(node, v_param, system, a, expert_thresholds, use_variance_reduction, if_feedback):
    # actual loss for each expert
    o_hat_a = 1 if expert_thresholds[a] > node.confidence else 0
    parent = node.parent
    if use_variance_reduction is True:
        residual_b = (node.b_true - node.b_hat) / node.p_to_cloud * if_feedback
        residual_C = (node.C_true - node.C_hat) / node.p_to_cloud * if_feedback
        if parent.level == system.num_layers - 1: # end of the recurse, parent is the cloud
            node.actual_loss[a] = (v_param * (1 - o_hat_a) * (residual_b + node.b_hat) +
                             o_hat_a * parent.Q * (residual_C + node.C_hat))
        else:
            node.actual_loss[a] = (v_param * (1 - o_hat_a) * (residual_b + node.b_hat) +
                             o_hat_a * parent.Q * (residual_C + node.C_hat) + parent.exp_loss)
    else:  # without VR
        if parent.level == system.num_layers - 1: # end of the recurse, parent is the cloud
            node.actual_loss[a] = (v_param * (1 - o_hat_a) * node.b_true
                             + o_hat_a * parent.Q * node.C_true)
        else:
            node.actual_loss[a] = (v_param * (1 - o_hat_a) * node.b_true
                             + o_hat_a * parent.Q * node.C_true + parent.exp_loss)




def aggregate_results(all_histories):
    aggregated = {}
    # These fields have variable length per node, don't try to stack them
    skip = {'node_costs', 'node_queues', 'node_offload_decisions', 'node_loss_values'}
    for key in all_histories[0].keys():
        if key in skip:
            aggregated[key] = {}
            for nid in all_histories[0][key].keys():
                # For node_loss_values, just collect all values from all trials
                if key == 'node_loss_values':
                    aggregated[key][nid] = [h[key][nid] for h in all_histories]
                else:
                    stacked = np.array([h[key][nid] for h in all_histories])
                    aggregated[key][nid] = {'mean': np.mean(stacked, axis=0), 'std': np.std(stacked, axis=0)}
        else:
            try:
                stacked = np.array([h[key] for h in all_histories])
                aggregated[f'{key}_mean'] = np.mean(stacked, axis=0)
                aggregated[f'{key}_std'] = np.std(stacked, axis=0)
            except:
                aggregated[key] = all_histories[0][key]
    return aggregated


def plot_results(agg_no_vr, agg_vr, all_no_vr, all_vr, system, num_jobs, prefix='results'):
    """Plot main performance results"""
    jobs = np.arange(num_jobs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{system.num_layers}-Layer Performance ({system.nodes_per_layer})', fontsize=24)

    # Error rate
    axes[0, 0].plot(jobs, np.cumsum(agg_no_vr['errors_mean']) / (jobs + 1), label='No VR')
    axes[0, 0].plot(jobs, np.cumsum(agg_vr['errors_mean']) / (jobs + 1), '--', label='With VR')
    axes[0, 0].set_ylabel('Avg Error Rate', fontsize=24)
    axes[0, 0].set_xlabel('Jobs', fontsize=24)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
    axes[0, 0].legend(fontsize=24);
    axes[0, 0].grid(True)

    # Feedback rate
    axes[0, 1].plot(jobs, np.cumsum(agg_no_vr['feedback_received_mean']) / (jobs + 1), label='No VR')
    axes[0, 1].plot(jobs, np.cumsum(agg_vr['feedback_received_mean']) / (jobs + 1), '--', label='With VR')
    axes[0, 1].set_ylabel('Feedback Rate', fontsize=24)
    axes[0, 1].set_xlabel('Jobs', fontsize=24)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[0, 1].legend(fontsize=24);
    axes[0, 1].grid(True)

    # # Total cost
    # axes[0, 2].plot(jobs, np.cumsum(agg_no_vr['costs_total_mean']) / (jobs + 1), label='No VR')
    # axes[0, 2].plot(jobs, np.cumsum(agg_vr['costs_total_mean']) / (jobs + 1), '--', label='With VR')
    # budget = sum(n.cost_budget_gamma for n in system.nodes.values() if n.Q is not None)
    # axes[0, 2].axhline(budget, color='r', linestyle=':', label=f'Budget={budget:.2f}')
    # axes[0, 2].set_ylabel('Avg Total Cost')
    # axes[0, 2].set_xlabel('Jobs')
    # axes[0, 2].legend();
    # axes[0, 2].grid(True)

    # Per-node costs
    for nid in list(agg_no_vr['node_costs'].keys())[:6]:
        axes[1, 0].plot(jobs, np.cumsum(agg_no_vr['node_costs'][nid]['mean']) / (jobs + 1), label=nid, alpha=0.7)
    axes[1, 0].set_ylabel('Avg Cost/Node', fontsize=24)
    axes[1, 0].set_xlabel('Jobs', fontsize=24)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 0].legend(fontsize=24);
    axes[1, 0].grid(True)

    # Queue sizes
    # for nid in list(agg_no_vr['node_queues'].keys())[:6]:
    #     axes[1, 1].plot(jobs, agg_no_vr['node_queues'][nid]['mean'], label=nid, alpha=0.7)
    # axes[1, 1].set_ylabel('Queue Size')
    # axes[1, 1].set_xlabel('Jobs')
    # axes[1, 1].legend(fontsize=8);
    # axes[1, 1].grid(True)

    # Execution distribution - FIXED: compute histogram per trial, then average
    exec_no = np.zeros(system.num_layers)
    exec_vr = np.zeros(system.num_layers)
    for h in all_no_vr:
        counts = np.bincount(np.array(h['execution_layer']).astype(int), minlength=system.num_layers)
        exec_no += counts
    for h in all_vr:
        counts = np.bincount(np.array(h['execution_layer']).astype(int), minlength=system.num_layers)
        exec_vr += counts
    exec_no /= len(all_no_vr)
    exec_vr /= len(all_vr)

    x = np.arange(system.num_layers)
    axes[1, 1].bar(x - 0.175, exec_no, 0.35, label='No VR')
    axes[1, 1].bar(x + 0.175, exec_vr, 0.35, label='With VR')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'L{i}' for i in range(system.num_layers)])
    axes[1, 1].set_ylabel('Jobs Executed', fontsize=24)
    axes[1, 1].set_title('Execution Distribution', fontsize=24)
    axes[1, 1].legend(fontsize=24);
    axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'{prefix}_performance_small.png', dpi=300, bbox_inches='tight')
    print(f"Saved {prefix}_performance_small.png")
    plt.close()


def plot_layer0_loss(all_no_vr, all_vr, system, prefix='results'):
    """
    Plot loss values for layer 0 (initial layer) nodes.

    X-axis: count of jobs received by that node
    Y-axis: sum of L_hat (loss estimate) for that job

    For no-VR: loss is 0 when no feedback (not updated)
    For VR: loss uses baseline even without feedback
    """
    # Get layer 0 node IDs
    layer0_nodes = [n.node_id for n in system.get_non_cloud_nodes() if n.level == 0]
    num_nodes = len(layer0_nodes)

    # Create figure - one subplot per layer 0 node
    cols = min(num_nodes, 4)
    rows = (num_nodes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f'Loss Values for Layer 0 Nodes ({system.num_layers}-Layer, {system.nodes_per_layer})', fontsize=14)

    for idx, nid in enumerate(layer0_nodes):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Get loss values from first trial (for clarity, or could average)
        loss_no_vr = all_no_vr[0]['node_loss_values'][nid]
        loss_vr = all_vr[0]['node_loss_values'][nid]

        # X-axis: job count for this node
        x_no_vr = np.arange(1, len(loss_no_vr) + 1)
        x_vr = np.arange(1, len(loss_vr) + 1)

        # Plot
        ax.scatter(x_no_vr, loss_no_vr, s=1, alpha=0.3, label='No VR', color='blue')
        ax.scatter(x_vr, loss_vr, s=1, alpha=0.3, label='With VR', color='orange')

        ax.set_xlabel('Job Count (for this node)')
        ax.set_ylabel('Loss (sum of L_hat)')
        ax.set_title(f'{nid}')
        ax.legend(markerscale=5)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_nodes, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{prefix}_layer0_loss.png', dpi=300, bbox_inches='tight')
    print(f"Saved {prefix}_layer0_loss.png")
    plt.close()

    # Also create a zoomed-in version showing rolling statistics
    fig2, axes2 = plt.subplots(1, 1, figsize=(5, 5))
    fig2.suptitle(f'Loss Statistics for Layer 0 ({system.num_layers}-Layer)', fontsize=14)

    # Pick first layer 0 node for detailed analysis
    nid = layer0_nodes[0]
    loss_no_vr = np.array(all_no_vr[0]['node_loss_values'][nid])
    loss_vr = np.array(all_vr[0]['node_loss_values'][nid])

    # Rolling mean
    window = 200
    if len(loss_no_vr) > window:
        roll_no = np.convolve(loss_no_vr, np.ones(window) / window, mode='valid')
        roll_vr = np.convolve(loss_vr, np.ones(window) / window, mode='valid')
        axes2.plot(roll_no, label='No VR', alpha=0.8)
        axes2.plot(roll_vr, label='With VR', alpha=0.8)
        axes2.set_xlabel('Job Count')
        axes2.set_ylabel(f'Rolling Mean Loss (window={window})')
        axes2.set_title(f'{nid}: Rolling Mean')
        axes2.legend();
        axes2.grid(True)

    # # Rolling variance (standard deviation)
    # if len(loss_no_vr) > window:
    #     # Compute rolling std
    #     roll_std_no = np.array([np.std(loss_no_vr[max(0, i - window):i]) for i in range(window, len(loss_no_vr))])
    #     roll_std_vr = np.array([np.std(loss_vr[max(0, i - window):i]) for i in range(window, len(loss_vr))])
    #     axes2[1].plot(roll_std_no, label='No VR', alpha=0.8)
    #     axes2[1].plot(roll_std_vr, label='With VR', alpha=0.8)
    #     axes2[1].set_xlabel('Job Count')
    #     axes2[1].set_ylabel(f'Rolling Std Dev (window={window})')
    #     axes2[1].set_title(f'{nid}: Rolling Std (Variance Reduction Effect)')
    #     axes2[1].legend();
    #     axes2[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{prefix}_layer0_loss_stats.png', dpi=300, bbox_inches='tight')
    print(f"Saved {prefix}_layer0_loss_stats.png")
    plt.close()


if __name__ == '__main__':
    NUM_TRIALS = 3
    NUM_JOBS = 20000
    NUM_EXPERTS = 50
    LEARNING_RATE = 0.01
    V_PARAM = 500

    baseline_error, per_task_baseline = compute_baseline_error_rate(BEST_MODEL_IDX)

    print("=" * 60)
    print("BASELINE PERFORMANCE (Qwen2-72B on all jobs)")
    print("=" * 60)
    print(f"Overall Error Rate: {baseline_error:.4f}")

    configs = get_system_configs()
    to_run = ['4layer_8-4-2-1']
    # , '3layer_4-2-1', '4layer_1-1-1-1', '4layer_8-4-2-1'

    for name in to_run:
        cfg = configs[name]
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")

        all_no_vr, all_vr = [], []
        for trial in range(NUM_TRIALS):
            print(f"Trial {trial + 1}/{NUM_TRIALS}")

            system = HierarchicalSystem(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'])
            res_no = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM,
                                      use_variance_reduction=False, initial_error_rates=Avg_err)
            all_no_vr.append(res_no)

            system = HierarchicalSystem(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'])
            res_vr = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM,
                                      use_variance_reduction=True, initial_error_rates=Avg_err)
            all_vr.append(res_vr)

        agg_no = aggregate_results(all_no_vr)
        agg_vr = aggregate_results(all_vr)

        system = HierarchicalSystem(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'])
        plot_results(agg_no, agg_vr, all_no_vr, all_vr, system, NUM_JOBS, prefix=name)
        plot_layer0_loss(all_no_vr, all_vr, system, prefix=name)

        print(f"\n--- {name} Results ---")
        print(
            f"No VR  - Error: {np.mean(agg_no['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_no['feedback_received_mean'][-2000:]):.4f}")
        print(
            f"VR     - Error: {np.mean(agg_vr['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_vr['feedback_received_mean'][-2000:]):.4f}")

    print("\n" + "=" * 60 + "\nAll experiments completed!\n" + "=" * 60)
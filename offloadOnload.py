import numpy as np
import matplotlib.pyplot as plt
from utilities.model_performance import estimate_model_perforance, convert_offloadingCost, estimate_offloadingCost, lowest_avg_error
import pickle as pkl
import random

# read dataset
FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
with open(FILENAME, 'rb') as f:
    data = pkl.load(f)
# --- System & Task Configuration ---
# Define error rates for [client, edge, cloud]
Avg_acc, Avg_err = estimate_model_perforance(data)
ERROR_RATES = Avg_err
# ERROR_RATES = {
#     #  client   edge    cloud
#     0: [0.5, 0.3, 0.0],  # Task type 1
#     1: [0.6, 0.4, 0.0],  # Task type 2
#     2: [0.7, 0.5, 0.0],  # Task type 3
# }

# Define BASE offloading costs (actual costs will have randomness added)
OffloadCost_SCALE = 0.001
avg_offloadCost = estimate_offloadingCost(data, OffloadCost_SCALE)
# BASE_OFFLOADING_COSTS = {
#     # client_to_edge, edge_to_cloud
#     0: [1.0, 0.5],  # Task type 1
#     1: [2.0, 1.0],  # Task type 2
#     2: [3.0, 1.5],  # Task type 3
# }

NUM_TASK_TYPES = len(data['task_types'])
TASK_NAMES = data['task_types']
TOTAL_JOBS = data['total_samples']

model_keywords = [
    'openai-community__gpt2-large', # 0
    'gpt2-xl',                      # 1
    'deepseek-llm-7b',              # 2
    'deepseek-ai__deepseek-moe-16b',# 3
    'deepseek-ai__deepseek-llm-67b',# 4
    'Qwen__Qwen2-0.5B',             # 5
    'Qwen__Qwen1.5-0.5B',           # 6
    'Qwen__Qwen2-72B',              # 7
]


ENABLE_ONLOADING = True
available_client_models = [0, 2, 3, 5, 6]
available_edge_models = [1, 4, 7]
MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])  # Sizes for each model_keywords
MODEL_ONLOADING_COSTS = MODEL_SIZES
CLIENT_MEMORY_CAPACITY = 17  # Adjust based on your model sizes
EDGE_MEMORY_CAPACITY = 85
ONLOADING_EPOCH_LENGTH = 500
V_ONLOAD = 700


def initialize_onloaded_models(memory_capacity, model_ids, model_sizes):
    """Randomly selects an initial set of models that fit within the memory capacity."""
    S = set()
    remaining_capacity = memory_capacity
    available_models = list(model_ids)
    np.random.shuffle(available_models)
    for model_id in available_models:
        if model_sizes[model_id] <= remaining_capacity:
            S.add(model_id)
            remaining_capacity -= model_sizes[model_id]
    return list(S)


def enhanced_greedy_onloading(V, prev_onloaded_models,
                              memory_capacity, model_sizes, onloading_costs,
                              error_rates_dict, task_dist_estimate,
                              available_models, task_names):
    """
    Selects a set of models to onload by maximizing a submodular utility function.

    Args:
        V (float): Lyapunov parameter for onloading decisions
        prev_onloaded_models (list): Previously onloaded model IDs
        memory_capacity (float): Total memory capacity
        model_sizes (np.array): Array of model sizes indexed by position in available_models
        onloading_costs (np.array): Array of onloading costs indexed by position in available_models
        error_rates_dict (dict): ERROR_RATES dictionary structure:
                                {task_type_str: {model_id: error_rate}}
        task_dist_estimate (np.array): Estimated task distribution (length = NUM_TASK_TYPES)
        available_models (list): List of model IDs that can be onloaded (e.g., available_client_models)
        task_names (list): List of task type names (TASK_NAMES)

    Returns:
        tuple: (selected_models, total_onloading_cost)
    """
    S = set()  # Selected models
    remaining_capacity = memory_capacity
    prev_onloaded_set = set(prev_onloaded_models)

    # Create mapping from available_models indices to actual model IDs
    # available_models contains the actual model IDs we can choose from
    num_tasks = len(task_names)

    while True:
        best_model, best_ratio = -1, -np.inf

        # Get candidate models that fit in remaining capacity
        candidate_models = []
        for idx, model_id in enumerate(available_models):
            if model_id not in S:
                # Check if this model fits
                if model_sizes[idx] <= remaining_capacity:
                    candidate_models.append((idx, model_id))

        if not candidate_models:
            break

        # Calculate the expected error for current set S across all tasks
        expected_error_S = []
        for t_idx, task_type in enumerate(task_names):
            if not S:
                # No models in S, use worst-case error (1.0)
                expected_error_S.append(1.0)
            else:
                # Find minimum error among models in S for this task
                errors_for_task = []
                for model_id in S:
                    if model_id in error_rates_dict[task_type]:
                        errors_for_task.append(error_rates_dict[task_type][model_id])

                if errors_for_task:
                    expected_error_S.append(min(errors_for_task))
                else:
                    # None of the models in S have error rates for this task
                    expected_error_S.append(1.0)

        # Evaluate each candidate model
        for idx, model_id in candidate_models:
            # Calculate marginal gain for adding this model
            sum_task_acc_gain = 0.0

            # Create temporary set with candidate model
            S_plus_m = S.union({model_id})

            # Calculate expected error with S + {m} for each task
            for t_idx, task_type in enumerate(task_names):
                # Find minimum error with S + {m}
                errors_for_task_plus_m = []
                for m in S_plus_m:
                    if m in error_rates_dict[task_type]:
                        errors_for_task_plus_m.append(error_rates_dict[task_type][m])

                if errors_for_task_plus_m:
                    error_S_plus_m = min(errors_for_task_plus_m)
                else:
                    error_S_plus_m = 1.0

                # Calculate marginal accuracy gain
                # Gain = reduction in error weighted by task probability
                accuracy_gain = V * task_dist_estimate[t_idx] * (expected_error_S[t_idx] - error_S_plus_m)
                sum_task_acc_gain += accuracy_gain

            # Calculate onloading cost (only pay if model wasn't previously onloaded)
            cost_increase = onloading_costs[idx] if model_id not in prev_onloaded_set else 0

            # Calculate marginal gain (benefit - cost)
            marginal_gain = sum_task_acc_gain - cost_increase

            # Calculate ratio (gain per unit memory)
            if marginal_gain > 0:
                ratio = marginal_gain / model_sizes[idx]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_model = model_id
                    best_idx = idx

        # If no model provides positive marginal gain, stop
        if best_model == -1:
            break

        # Add best model to selection
        S.add(best_model)
        remaining_capacity -= model_sizes[best_idx]

    # Calculate total onloading cost (only for newly onloaded models)
    total_cost = 0.0
    for idx, model_id in enumerate(available_models):
        if model_id in S and model_id not in prev_onloaded_set:
            total_cost += onloading_costs[idx]

    return list(S), total_cost

def exp4_hierarchical_algorithm(num_jobs, num_experts, learning_rate, v_param,
                                cost_budget_gamma_c, cost_budget_gamma_e,
                                initial_bias_strength=0.0,
                                use_variance_reduction=False,
                                cost_std=0.05, enable_onloading=True):
    """
    Implements the hierarchical EXP4 algorithm with separate virtual queues
    and practical partial feedback (only when both client and edge offload).

    Args:
        num_jobs (int): Total number of inference jobs to simulate.
        num_experts (int): Number of experts (thresholds) for decision making.
        learning_rate (float): Learning rate (eta) for the EXP4 algorithm.
        v_param (float): Lyapunov drift-plus-penalty parameter (V).
        cost_budget_gamma_c (float): Average cost budget for client-to-edge link.
        cost_budget_gamma_e (float): Average cost budget for edge-to-cloud link.
        initial_bias_strength (float): Controls the initial bias towards offloading.
        use_variance_reduction (bool): Whether to use variance reduction technique.
        cost_std (float): Standard deviation of the Gaussian noise added to costs.

    Returns:
        dict: A dictionary containing lists of metrics over time.
    """
    # --- Initialization ---
    expert_thresholds = np.linspace(0, 1, num_experts)

    # Initialize cumulative losses with optional bias
    initial_client_losses = initial_bias_strength * (1 - expert_thresholds)
    S_client = np.tile(initial_client_losses, (NUM_TASK_TYPES, 1))
    S_edge = np.zeros((NUM_TASK_TYPES, num_experts))

    # Initialize weights uniformly
    w_client = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts
    w_edge = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts

    # Separate virtual queues for each link
    Q_c = 0.0  # Client-to-edge queue
    Q_e = 0.0  # Edge-to-cloud queue

    # Initialize onloaded models
    onloaded_client = initialize_onloaded_models(CLIENT_MEMORY_CAPACITY,
                                                 np.array(available_client_models),
                                                 MODEL_SIZES)
    onloaded_edge = initialize_onloaded_models(EDGE_MEMORY_CAPACITY,
                                               np.array(available_edge_models),
                                               MODEL_SIZES)

    if not enable_onloading:
        # Fix models to initial selection
        onloaded_client = available_client_models
        onloaded_edge = available_edge_models

    task_counts = {t: 0 for t in TASK_NAMES}

    # Track offloading probabilities for onloading decisions
    epoch_offload_c_sum = {t: 0.0 for t in TASK_NAMES}
    epoch_offload_e_sum = {t: 0.0 for t in TASK_NAMES}
    epoch_task_counts = {t: 0 for t in TASK_NAMES}
    offload_c_probs = {t: 0.5 for t in TASK_NAMES}
    offload_e_probs = {t: 0.5 for t in TASK_NAMES}

    # For variance reduction: track estimated costs and error rates
    if use_variance_reduction:
        # Initialize with true values (as specified in requirements)
        # Counters for running averages
        cost_observation_counts_c = {t: 2000 for t in TASK_NAMES}

    history = {
        'errors': [], 'costs_client': [], 'costs_edge': [], 'costs_total': [],
        'queue_client': [], 'queue_edge': [],
        'client_weight_entropy': [], 'edge_weight_entropy': [],
        'offload_client': [], 'offload_edge': [], 'feedback_received': [],
        'actual_costs_client': [], 'actual_costs_edge': [],
        'onload_costs': [],  # Add this
        'onloaded_client_history': [list(onloaded_client)],  # Add this
        'onloaded_edge_history': [list(onloaded_edge)]  # Add this
    }

    # --- Main Simulation Loop ---
    for j in range(num_jobs):

        # ----------------------onload----------------------
        if enable_onloading and j > 0 and j % ONLOADING_EPOCH_LENGTH == 0:
            # Estimate task distribution
            total_tasks = sum(task_counts.values())
            task_dist_array = np.array([task_counts[t] / (total_tasks + 1e-9) for t in TASK_NAMES])

            # Update average offloading probabilities from past epoch
            for t in TASK_NAMES:
                if epoch_task_counts[t] > 0:
                    offload_c_probs[t] = epoch_offload_c_sum[t] / epoch_task_counts[t]
                    offload_e_probs[t] = epoch_offload_e_sum[t] / epoch_task_counts[t]

            # Estimate task distribution for client (processes locally)
            task_dist_c = np.array([task_dist_array[i] * (1 - offload_c_probs[TASK_NAMES[i]])
                                    for i in range(NUM_TASK_TYPES)])

            # Client onloading decision
            client_model_sizes = MODEL_SIZES[available_client_models]
            client_onload_costs = MODEL_ONLOADING_COSTS[available_client_models]
            onloaded_client, cost_c = enhanced_greedy_onloading(
                    V=V_ONLOAD,
                    prev_onloaded_models=onloaded_client,
                    memory_capacity=CLIENT_MEMORY_CAPACITY,
                    model_sizes=client_model_sizes,
                    onloading_costs=client_onload_costs,
                    error_rates_dict=ERROR_RATES,
                    task_dist_estimate=task_dist_c,
                    available_models=available_client_models,  # [2, 3, 5, 6]
                    task_names=TASK_NAMES
                )

            # Estimate task distribution for edge (client offloads but edge processes)
            task_dist_e = np.array([task_dist_array[i] * offload_c_probs[TASK_NAMES[i]] *
                                    (1 - offload_e_probs[TASK_NAMES[i]])
                                    for i in range(NUM_TASK_TYPES)])

            # Edge onloading decision
            edge_model_sizes = MODEL_SIZES[available_edge_models]
            edge_onload_costs = MODEL_ONLOADING_COSTS[available_edge_models]

            onloaded_edge, cost_e = enhanced_greedy_onloading(
                V=V_ONLOAD,
                prev_onloaded_models=onloaded_edge,
                memory_capacity=EDGE_MEMORY_CAPACITY,
                model_sizes=edge_model_sizes,
                onloading_costs=edge_onload_costs,
                error_rates_dict=ERROR_RATES,
                task_dist_estimate=task_dist_e,
                available_models=available_edge_models,  # [0, 1, 4, 7]
                task_names=TASK_NAMES
            )

            # Record decisions
            history['onload_costs'].append(cost_c + cost_e)
            history['onloaded_client_history'].append(list(onloaded_client))
            history['onloaded_edge_history'].append(list(onloaded_edge))

            # Reset epoch counters
            epoch_offload_c_sum = {t: 0.0 for t in TASK_NAMES}
            epoch_offload_e_sum = {t: 0.0 for t in TASK_NAMES}
            epoch_task_counts = {t: 0 for t in TASK_NAMES}
        else:
            history['onload_costs'].append(0)
        # ----------onload end------------------



        # Select one value, range from 0 to TOTAL_JOBS
        idx = random.randint(0, num_jobs - 1)
        task_type = data['full_data'][idx]['category']
        task_type_idx = TASK_NAMES.index(task_type)
        epoch_task_counts[task_type] += 1
        client_err, best_client_model = lowest_avg_error(ERROR_RATES[task_type], onloaded_client)
        edge_err, best_edge_model = lowest_avg_error(ERROR_RATES[task_type], onloaded_edge)
        # cloud_err = ERROR_RATES[task_type][2]
        mean_confidence = 1.0 - client_err
        std_dev = 0.1
        confidence_Z = np.random.normal(loc=mean_confidence, scale=std_dev)
        confidence_Z = np.clip(confidence_Z, 0, 1)

        # Get costs for this task type (with added randomness)

        # Add Gaussian noise to costs
        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)

        task_counts[task_type] += 1

        # --- CLIENT DECISION ---
        # Determine which experts recommend offloading
        client_offload_experts = expert_thresholds > confidence_Z
        prob_offload_c = np.sum(w_client[task_type_idx, client_offload_experts])
        prob_offload_c = np.clip(prob_offload_c, 1e-5, 1 - 1e-5)

        epoch_offload_c_sum[task_type] += prob_offload_c



        # Sample client decision
        o_c = 1 if np.random.rand() < prob_offload_c else 0

        # --- EDGE DECISION (only if client offloads) ---
        o_e = 0
        edge_confidence_Z = 0
        if o_c == 1:
            mean_confidence = 1.0 - edge_err
            std_dev = 0.1
            edge_confidence_Z = np.random.normal(loc=mean_confidence, scale=std_dev)
            edge_confidence_Z = np.clip(edge_confidence_Z, 0, 1)
            edge_offload_experts = expert_thresholds > edge_confidence_Z
            prob_offload_e = np.sum(w_edge[task_type_idx, edge_offload_experts])
            prob_offload_e = np.clip(prob_offload_e, 1e-5, 1 - 1e-5)

            # Sample edge decision
            o_e = 1 if np.random.rand() < prob_offload_e else 0
        else:
            prob_offload_e = 0.5  # Default value when not relevant

        epoch_offload_e_sum[task_type] += prob_offload_e

        # --- DETERMINE JOB ERROR ---
        if o_c == 0:
            # Processed at client
            job_error = int(1 - data['full_data'][idx]['results'][best_client_model])
        elif o_e == 0:
            # Processed at edge
            job_error = int(1 - data['full_data'][idx]['results'][best_edge_model])
        else:
            # Processed at cloud
            job_error = 0

        # --- FEEDBACK AND LOSS COMPUTATION ---
        # Feedback is only available if BOTH client and edge offload (o_c=1 AND o_e=1)
        feedback_received = (o_c == 1) and (o_e == 1)

        # Actual error indicator
        b_c = int(1 - data['full_data'][idx]['results'][best_client_model])
        b_e = int(1 - data['full_data'][idx]['results'][best_edge_model])

        # --- UPDATE CLIENT WEIGHTS ---
        L_hat_c = np.zeros(num_experts)

        # Update estimated costs if we have feedback (for VR)
        if feedback_received and use_variance_reduction:
            # Update client cost estimate
            count_c = cost_observation_counts_c[task_type]
            avg_offloadCost[task_type] = (avg_offloadCost[task_type] * count_c + C_c) / (count_c + 1)
            cost_observation_counts_c[task_type] += 1

            # Update edge cost estimate
            avg_offloadCost[task_type] = (avg_offloadCost[task_type] * count_c + C_c) / (count_c + 1)
            cost_observation_counts_c[task_type] += 1

        # Compute loss estimator for each expert
        if use_variance_reduction:
            # VARIANCE REDUCTION: Always update, use baselines when no feedback
            for a in range(num_experts):
                o_hat_c_a = 1 if expert_thresholds[a] > confidence_Z else 0

                b_hat_c = client_err
                b_hat_e = edge_err
                C_hat_c = avg_offloadCost[task_type]
                C_hat_e = avg_offloadCost[task_type]

                # Compute expected offloading loss
                L_hat_offload = Q_c * C_hat_c + v_param * (
                            1 - prob_offload_e) * b_hat_e + prob_offload_e * Q_e * C_hat_e

                if o_hat_c_a == 0:
                    # Expert recommends processing at client
                    if feedback_received:
                        # Have observations: use residual + baseline
                        residual_error = (b_c - b_hat_c) / (prob_offload_c * prob_offload_e)
                        L_hat_c[a] = v_param * (residual_error + b_hat_c)
                    else:
                        # No observations: use baseline only
                        L_hat_c[a] = v_param * b_hat_c
                else:
                    # Expert recommends offloading to edge
                    if feedback_received:
                        # Have observations: use residual + baseline
                        L_offload_actual = Q_c * C_c + v_param * (1 - prob_offload_e) * b_e + prob_offload_e * Q_e * C_c
                        residual_offload = (L_offload_actual - L_hat_offload) / (prob_offload_c * prob_offload_e)
                        L_hat_c[a] = residual_offload + L_hat_offload
                    else:
                        # No observations: use baseline only
                        L_hat_c[a] = L_hat_offload

            # Always update with VR
            S_client[task_type_idx] += L_hat_c

        elif feedback_received:
            # ORIGINAL: Only update when we have feedback
            for a in range(num_experts):
                o_hat_c_a = 1 if expert_thresholds[a] > confidence_Z else 0

                if o_hat_c_a == 0:
                    L_ideal_c = v_param * b_c
                else:
                    L_offload = Q_c * C_c + (1 - prob_offload_e) * v_param * b_e + prob_offload_e * Q_e * C_c
                    L_ideal_c = L_offload

                L_hat_c[a] = L_ideal_c / (prob_offload_c * prob_offload_e)

            # Only update when we have feedback
            S_client[task_type_idx] += L_hat_c

        # --- UPDATE EDGE WEIGHTS (only if client offloaded) ---
        L_hat_e = np.zeros(num_experts)

        if o_c == 1:
            # Edge can update when client offloaded (edge made a decision)
            if use_variance_reduction:
                # VARIANCE REDUCTION: Always update when edge participates, use baselines when no feedback
                for k in range(num_experts):
                    o_hat_e_k = 1 if expert_thresholds[k] > edge_confidence_Z else 0

                    b_hat_e = edge_err
                    C_hat_e = avg_offloadCost[task_type]

                    if o_hat_e_k == 0:
                        # Expert recommends processing at edge
                        if feedback_received:
                            # Have observations: use residual + baseline
                            residual_error = (b_e - b_hat_e) / prob_offload_e
                            L_hat_e[k] = v_param * (residual_error + b_hat_e)
                        else:
                            # No observations: use baseline only
                            L_hat_e[k] = v_param * b_hat_e
                    else:
                        # Expert recommends offloading to cloud
                        if feedback_received:
                            # Have observations: use residual + baseline
                            residual_cost = (C_c - C_hat_e) / prob_offload_e
                            L_hat_e[k] = Q_e * (residual_cost + C_hat_e)
                        else:
                            # No observations: use baseline only
                            L_hat_e[k] = Q_e * C_hat_e

                # Always update with VR when edge participates
                S_edge[task_type_idx] += L_hat_e

            elif feedback_received:
                # ORIGINAL: Only update when we have feedback
                for k in range(num_experts):
                    o_hat_e_k = 1 if expert_thresholds[k] > edge_confidence_Z else 0

                    if o_hat_e_k == 0:
                        L_ideal_e = v_param * b_e
                    else:
                        L_ideal_e = Q_e * C_c

                    L_hat_e[k] = L_ideal_e / prob_offload_e

                # Only update when we have feedback
                S_edge[task_type_idx] += L_hat_e

        # --- UPDATE WEIGHTS USING EXP4 ---
        # Update client weights using log-sum-exp trick for numerical stability
        S_client_task = S_client[task_type_idx]
        log_w_client = -learning_rate * S_client_task
        log_w_client_max = np.max(log_w_client)
        log_w_client_shifted = log_w_client - log_w_client_max
        w_client[task_type_idx] = np.exp(log_w_client_shifted)
        w_client[task_type_idx] /= np.sum(w_client[task_type_idx])

        # Ensure no NaN values
        if np.any(np.isnan(w_client[task_type_idx])):
            w_client[task_type_idx] = np.ones(num_experts) / num_experts

        # Update edge weights (if client offloaded)
        if o_c == 1:
            S_edge_task = S_edge[task_type_idx]
            log_w_edge = -learning_rate * S_edge_task
            log_w_edge_max = np.max(log_w_edge)
            log_w_edge_shifted = log_w_edge - log_w_edge_max
            w_edge[task_type_idx] = np.exp(log_w_edge_shifted)
            w_edge[task_type_idx] /= np.sum(w_edge[task_type_idx])

            # Ensure no NaN values
            if np.any(np.isnan(w_edge[task_type_idx])):
                w_edge[task_type_idx] = np.ones(num_experts) / num_experts

        # --- UPDATE VIRTUAL QUEUES ---
        job_cost_c = o_c * C_c  # Client incurs cost only if it offloads
        job_cost_e = o_c * o_e * C_c  # Edge incurs cost only if both offload

        Q_c = max(0, Q_c + job_cost_c - cost_budget_gamma_c)
        Q_e = max(0, Q_e + job_cost_e - cost_budget_gamma_e)

        # --- RECORD HISTORY ---
        history['errors'].append(job_error)
        history['costs_client'].append(job_cost_c)
        history['costs_edge'].append(job_cost_e)
        history['costs_total'].append(job_cost_c + job_cost_e)
        history['actual_costs_client'].append(C_c if o_c == 1 else 0)
        history['actual_costs_edge'].append(C_c if o_c == 1 and o_e == 1 else 0)
        history['queue_client'].append(Q_c)
        history['queue_edge'].append(Q_e)
        history['offload_client'].append(o_c)
        history['offload_edge'].append(o_e)
        history['feedback_received'].append(int(feedback_received))

        # Compute weight entropy
        epsilon = 1e-9
        entropy_c = -np.sum(w_client * np.log2(w_client + epsilon), axis=1).mean()
        entropy_e = -np.sum(w_edge * np.log2(w_edge + epsilon), axis=1).mean()
        history['client_weight_entropy'].append(entropy_c)
        history['edge_weight_entropy'].append(entropy_e)

    return history


def aggregate_results(all_histories):
    """Averages results over multiple trials and calculates standard deviation."""
    aggregated = {}
    keys = all_histories[0].keys()

    # Define which keys should NOT be aggregated (non-numeric or variable-length lists)
    skip_keys = {'onloaded_client_history', 'onloaded_edge_history'}

    for key in keys:
        if key in skip_keys:
            # For model history, just keep the first trial's data for reference
            aggregated[key] = all_histories[0][key]
            continue

        try:
            stacked = np.array([h[key] for h in all_histories])
            aggregated[f'{key}_mean'] = np.mean(stacked, axis=0)
            aggregated[f'{key}_std'] = np.std(stacked, axis=0)
        except ValueError as e:
            # If stacking fails, skip this key
            print(f"Warning: Could not aggregate key '{key}': {e}")
            aggregated[key] = all_histories[0][key]

    return aggregated


def plot_metric(ax, x_axis, data, key, label, linestyle='-'):
    """Helper function to plot mean and shaded standard deviation."""
    ax.plot(x_axis, data[f'{key}_mean'], label=label, linestyle=linestyle)
    ax.fill_between(x_axis,
                    data[f'{key}_mean'] - data[f'{key}_std'],
                    data[f'{key}_mean'] + data[f'{key}_std'],
                    alpha=0.2)


def plot_performance_comparison(agg_no_vr, agg_vr, cost_budget_gamma_c, cost_budget_gamma_e, num_jobs):
    """Compare performance with and without variance reduction"""
    jobs_axis = np.arange(num_jobs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Variance Reduction Impact: Performance Comparison (Averaged over Trials)', fontsize=16)

    # Calculate moving averages
    avg_error_no_vr = np.cumsum(agg_no_vr['errors_mean']) / (jobs_axis + 1)
    avg_error_vr = np.cumsum(agg_vr['errors_mean']) / (jobs_axis + 1)

    avg_cost_c_no_vr = np.cumsum(agg_no_vr['costs_client_mean']) / (jobs_axis + 1)
    avg_cost_c_vr = np.cumsum(agg_vr['costs_client_mean']) / (jobs_axis + 1)

    avg_cost_e_no_vr = np.cumsum(agg_no_vr['costs_edge_mean']) / (jobs_axis + 1)
    avg_cost_e_vr = np.cumsum(agg_vr['costs_edge_mean']) / (jobs_axis + 1)

    # Row 1: Error Rate
    axes[0, 0].plot(jobs_axis, avg_error_no_vr, label='No Variance Reduction')
    axes[0, 0].plot(jobs_axis, avg_error_vr, label='With Variance Reduction', linestyle='--')
    axes[0, 0].set_ylabel('Average Error Rate')
    axes[0, 0].set_ylim(0.0, 0.5)
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()
    axes[0, 0].set_title('System Accuracy')

    # Row 1: Feedback Rate
    avg_feedback_no_vr = np.cumsum(agg_no_vr['feedback_received_mean']) / (jobs_axis + 1)
    avg_feedback_vr = np.cumsum(agg_vr['feedback_received_mean']) / (jobs_axis + 1)
    axes[0, 1].plot(jobs_axis, avg_feedback_no_vr, label='No Variance Reduction')
    axes[0, 1].plot(jobs_axis, avg_feedback_vr, label='With Variance Reduction', linestyle='--')
    axes[0, 1].set_ylabel('Feedback Rate')
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    axes[0, 1].set_title('Feedback Availability (Both Offload)')

    # Row 2: Client Cost
    axes[1, 0].plot(jobs_axis, avg_cost_c_no_vr, label='No Variance Reduction')
    axes[1, 0].plot(jobs_axis, avg_cost_c_vr, label='With Variance Reduction', linestyle='--')
    axes[1, 0].axhline(y=cost_budget_gamma_c, color='r', linestyle=':',
                       label=f'Client Budget (γc={cost_budget_gamma_c})')
    axes[1, 0].set_ylabel('Average Client Cost')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()
    axes[1, 0].set_title('Client-to-Edge Cost vs. Budget')

    # Row 2: Edge Cost
    axes[1, 1].plot(jobs_axis, avg_cost_e_no_vr, label='No Variance Reduction')
    axes[1, 1].plot(jobs_axis, avg_cost_e_vr, label='With Variance Reduction', linestyle='--')
    axes[1, 1].axhline(y=cost_budget_gamma_e, color='r', linestyle=':',
                       label=f'Edge Budget (γe={cost_budget_gamma_e})')
    axes[1, 1].set_ylabel('Average Edge Cost')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()
    axes[1, 1].set_title('Edge-to-Cloud Cost vs. Budget')

    # Row 3: Queue Sizes
    # plot_metric(axes[2, 0], jobs_axis, agg_no_vr, 'queue_client', 'No Variance Reduction', '-')
    # plot_metric(axes[2, 0], jobs_axis, agg_vr, 'queue_client', 'With Variance Reduction', '--')
    # axes[2, 0].set_xlabel('Number of Jobs')
    # axes[2, 0].set_ylabel('Client Queue Size')
    # axes[2, 0].grid(True, linestyle='--', alpha=0.6)
    # axes[2, 0].legend()
    # axes[2, 0].set_title('Client Virtual Queue Dynamics')
    #
    # plot_metric(axes[2, 1], jobs_axis, agg_no_vr, 'queue_edge', 'No Variance Reduction', '-')
    # plot_metric(axes[2, 1], jobs_axis, agg_vr, 'queue_edge', 'With Variance Reduction', '--')
    # axes[2, 1].set_xlabel('Number of Jobs')
    # axes[2, 1].set_ylabel('Edge Queue Size')
    # axes[2, 1].grid(True, linestyle='--', alpha=0.6)
    # axes[2, 1].legend()
    # axes[2, 1].set_title('Edge Virtual Queue Dynamics')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./results/performance.pdf', dpi=150, bbox_inches='tight')
    # plt.show()


def plot_learning_comparison(agg_no_vr, agg_vr, num_jobs):
    """Compare learning dynamics with and without variance reduction"""
    jobs_axis = np.arange(num_jobs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Variance Reduction Impact: Learning Dynamics (Averaged over Trials)', fontsize=16)

    # Client Policy Convergence
    plot_metric(axes[0, 0], jobs_axis, agg_no_vr, 'client_weight_entropy', 'No Variance Reduction', '-')
    plot_metric(axes[0, 0], jobs_axis, agg_vr, 'client_weight_entropy', 'With Variance Reduction', '--')
    axes[0, 0].set_ylabel('Average Entropy (bits)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()
    axes[0, 0].set_title('Client Policy Convergence (Weight Entropy)')

    # Edge Policy Convergence
    plot_metric(axes[0, 1], jobs_axis, agg_no_vr, 'edge_weight_entropy', 'No Variance Reduction', '-')
    plot_metric(axes[0, 1], jobs_axis, agg_vr, 'edge_weight_entropy', 'With Variance Reduction', '--')
    axes[0, 1].set_ylabel('Average Entropy (bits)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    axes[0, 1].set_title('Edge Policy Convergence (Weight Entropy)')

    # Client Offload Rate
    avg_offload_c_no_vr = np.cumsum(agg_no_vr['offload_client_mean']) / (jobs_axis + 1)
    avg_offload_c_vr = np.cumsum(agg_vr['offload_client_mean']) / (jobs_axis + 1)
    axes[1, 0].plot(jobs_axis, avg_offload_c_no_vr, label='No Variance Reduction')
    axes[1, 0].plot(jobs_axis, avg_offload_c_vr, label='With Variance Reduction', linestyle='--')
    axes[1, 0].set_xlabel('Number of Jobs')
    axes[1, 0].set_ylabel('Client Offload Rate')
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()
    axes[1, 0].set_title('Client Offloading Behavior')

    # Edge Offload Rate
    avg_offload_e_no_vr = np.cumsum(agg_no_vr['offload_edge_mean']) / (jobs_axis + 1)
    avg_offload_e_vr = np.cumsum(agg_vr['offload_edge_mean']) / (jobs_axis + 1)
    axes[1, 1].plot(jobs_axis, avg_offload_e_no_vr, label='No Variance Reduction')
    axes[1, 1].plot(jobs_axis, avg_offload_e_vr, label='With Variance Reduction', linestyle='--')
    axes[1, 1].set_xlabel('Number of Jobs')
    axes[1, 1].set_ylabel('Edge Offload Rate')
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()
    axes[1, 1].set_title('Edge Offloading Behavior')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('/mnt/user-data/outputs/variance_reduction_learning.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # --- Simulation Parameters ---
    NUM_TRIALS = 10
    NUM_JOBS = 40000
    NUM_EXPERTS = 50
    LEARNING_RATE_ETA = 0.01
    V_PARAM = 500
    COST_BUDGET_GAMMA_C = 0.5  # Client-to-edge budget
    COST_BUDGET_GAMMA_E = 0.5  # Edge-to-cloud budget
    INITIAL_BIAS_STRENGTH = 0.0
    COST_STD = 0.05  # Standard deviation for cost randomness

    # Set plotting parameters
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

    all_results_no_vr, all_results_vr = [], []

    print(f"Running {NUM_TRIALS} trials comparing variance reduction methods...")
    print(f"Client budget: {COST_BUDGET_GAMMA_C}, Edge budget: {COST_BUDGET_GAMMA_E}")
    print(f"V parameter: {V_PARAM}, Learning rate: {LEARNING_RATE_ETA}")
    print(f"Cost noise std: {COST_STD}\n")

    for i in range(NUM_TRIALS):
        print(f"--- Running Trial {i + 1} of {NUM_TRIALS} ---")

        # Run WITHOUT variance reduction
        results_no_vr = exp4_hierarchical_algorithm(
            num_jobs=NUM_JOBS, num_experts=NUM_EXPERTS, learning_rate=LEARNING_RATE_ETA,
            v_param=V_PARAM, cost_budget_gamma_c=COST_BUDGET_GAMMA_C,
            cost_budget_gamma_e=COST_BUDGET_GAMMA_E, initial_bias_strength=INITIAL_BIAS_STRENGTH,
            use_variance_reduction=False, cost_std=COST_STD
        )
        all_results_no_vr.append(results_no_vr)

        # Run WITH variance reduction
        results_vr = exp4_hierarchical_algorithm(
            num_jobs=NUM_JOBS, num_experts=NUM_EXPERTS, learning_rate=LEARNING_RATE_ETA,
            v_param=V_PARAM, cost_budget_gamma_c=COST_BUDGET_GAMMA_C,
            cost_budget_gamma_e=COST_BUDGET_GAMMA_E, initial_bias_strength=INITIAL_BIAS_STRENGTH,
            use_variance_reduction=True, cost_std=COST_STD
        )
        all_results_vr.append(results_vr)

    print("\n--- Aggregating results across all trials ---")
    agg_no_vr = aggregate_results(all_results_no_vr)
    agg_vr = aggregate_results(all_results_vr)

    # Print summary statistics
    # print("\n=== SUMMARY STATISTICS ===")
    # print(f"\nWithout Variance Reduction:")
    # print(f"  Final avg error rate: {np.mean(agg_no_vr['errors_mean'][-1000:]):.4f}")
    # print(f"  Final avg client cost: {np.mean(agg_no_vr['costs_client_mean'][-1000:]):.4f}")
    # print(f"  Final avg edge cost: {np.mean(agg_no_vr['costs_edge_mean'][-1000:]):.4f}")
    # print(f"  Final avg feedback rate: {np.mean(agg_no_vr['feedback_received_mean'][-1000:]):.4f}")
    # print(f"  Final client queue: {np.mean(agg_no_vr['queue_client_mean'][-1000:]):.2f}")
    # print(f"  Final edge queue: {np.mean(agg_no_vr['queue_edge_mean'][-1000:]):.2f}")
    #
    # print(f"\nWith Variance Reduction:")
    # print(f"  Final avg error rate: {np.mean(agg_vr['errors_mean'][-1000:]):.4f}")
    # print(f"  Final avg client cost: {np.mean(agg_vr['costs_client_mean'][-1000:]):.4f}")
    # print(f"  Final avg edge cost: {np.mean(agg_vr['costs_edge_mean'][-1000:]):.4f}")
    # print(f"  Final avg feedback rate: {np.mean(agg_vr['feedback_received_mean'][-1000:]):.4f}")
    # print(f"  Final client queue: {np.mean(agg_vr['queue_client_mean'][-1000:]):.2f}")
    # print(f"  Final edge queue: {np.mean(agg_vr['queue_edge_mean'][-1000:]):.2f}")

    print("\nSimulations finished. Plotting comparison results...")
    plot_performance_comparison(agg_no_vr, agg_vr, COST_BUDGET_GAMMA_C, COST_BUDGET_GAMMA_E, NUM_JOBS)
    # plot_learning_comparison(agg_no_vr, agg_vr, NUM_JOBS)


#  estimate error rates without bias
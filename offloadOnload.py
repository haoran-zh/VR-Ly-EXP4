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

# --- System & Task Configuration ---
# Define error rates for [client, edge, cloud]
Avg_acc, Avg_err = estimate_model_perforance(data)
ERROR_RATES = Avg_err  # This remains as Ground Truth (or initial warm start)

# Define BASE offloading costs
OffloadCost_SCALE = 0.001
avg_offloadCost = estimate_offloadingCost(data, OffloadCost_SCALE)

NUM_TASK_TYPES = len(data['task_types'])
TASK_NAMES = data['task_types']
TOTAL_JOBS = data['total_samples']

model_keywords = [
    'openai-community__gpt2-large',  # 0
    'gpt2-xl',  # 1
    'deepseek-llm-7b',  # 2
    'deepseek-ai__deepseek-moe-16b',  # 3
    'deepseek-ai__deepseek-llm-67b',  # 4
    'Qwen__Qwen2-0.5B',  # 5
    'Qwen__Qwen1.5-0.5B',  # 6
    'Qwen__Qwen2-72B',  # 7
]

ENABLE_ONLOADING = True
available_client_models = [0, 2, 3, 5, 6]
available_edge_models = [1, 4, 7]
MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])
MODEL_ONLOADING_COSTS = MODEL_SIZES
CLIENT_MEMORY_CAPACITY = 17
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
    Uses 'error_rates_dict' which can be dynamic estimates.
    """
    S = set()
    remaining_capacity = memory_capacity
    prev_onloaded_set = set(prev_onloaded_models)

    num_tasks = len(task_names)

    while True:
        best_model, best_ratio = -1, -np.inf

        candidate_models = []
        for idx, model_id in enumerate(available_models):
            if model_id not in S:
                if model_sizes[idx] <= remaining_capacity:
                    candidate_models.append((idx, model_id))

        if not candidate_models:
            break

        # Calculate expected error for current set S
        expected_error_S = []
        for t_idx, task_type in enumerate(task_names):
            if not S:
                expected_error_S.append(1.0)
            else:
                errors_for_task = []
                for model_id in S:
                    if model_id in error_rates_dict[task_type]:
                        errors_for_task.append(error_rates_dict[task_type][model_id])

                if errors_for_task:
                    expected_error_S.append(min(errors_for_task))
                else:
                    expected_error_S.append(1.0)

        # Evaluate each candidate model
        for idx, model_id in candidate_models:
            sum_task_acc_gain = 0.0
            S_plus_m = S.union({model_id})

            for t_idx, task_type in enumerate(task_names):
                errors_for_task_plus_m = []
                for m in S_plus_m:
                    if m in error_rates_dict[task_type]:
                        errors_for_task_plus_m.append(error_rates_dict[task_type][m])

                if errors_for_task_plus_m:
                    error_S_plus_m = min(errors_for_task_plus_m)
                else:
                    error_S_plus_m = 1.0

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

    total_cost = 0.0
    for idx, model_id in enumerate(available_models):
        if model_id in S and model_id not in prev_onloaded_set:
            total_cost += onloading_costs[idx]

    return list(S), total_cost


def exp4_hierarchical_algorithm(num_jobs, num_experts, learning_rate, v_param,
                                cost_budget_gamma_c, cost_budget_gamma_e,
                                initial_bias_strength=0.0,
                                use_variance_reduction=False,
                                cost_std=0.05, enable_onloading=True,
                                initial_error_rates=None):
    """
    Implements hierarchical EXP4 with UNBIASED ERROR RATE ESTIMATION (IPS).
    """
    # --- Initialization ---
    expert_thresholds = np.linspace(0, 1, num_experts)

    # Initialize Dynamic Error Rates (Mutable)
    if initial_error_rates is None:
        estimated_error_rates = copy.deepcopy(ERROR_RATES)
    else:
        estimated_error_rates = copy.deepcopy(initial_error_rates)

    # IPS Hyperparameters
    ERROR_LEARNING_RATE = 0.05  # Alpha for moving average of error rates
    MAX_IPS_WEIGHT = 20.0  # Clipping to prevent variance explosion

    # Initialize cumulative losses
    initial_client_losses = initial_bias_strength * (1 - expert_thresholds)
    S_client = np.tile(initial_client_losses, (NUM_TASK_TYPES, 1))
    S_edge = np.zeros((NUM_TASK_TYPES, num_experts))

    # Initialize weights
    w_client = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts
    w_edge = np.ones((NUM_TASK_TYPES, num_experts)) / num_experts

    # Virtual queues
    Q_c = 0.0
    Q_e = 0.0

    # Initialize onloaded models
    onloaded_client = initialize_onloaded_models(CLIENT_MEMORY_CAPACITY,
                                                 np.array(available_client_models),
                                                 MODEL_SIZES)
    onloaded_edge = initialize_onloaded_models(EDGE_MEMORY_CAPACITY,
                                               np.array(available_edge_models),
                                               MODEL_SIZES)

    if not enable_onloading:
        onloaded_client = available_client_models
        onloaded_edge = available_edge_models

    task_counts = {t: 0 for t in TASK_NAMES}

    # Track offloading probabilities for onloading decisions
    epoch_offload_c_sum = {t: 0.0 for t in TASK_NAMES}
    epoch_offload_e_sum = {t: 0.0 for t in TASK_NAMES}
    epoch_task_counts = {t: 0 for t in TASK_NAMES}
    offload_c_probs = {t: 0.5 for t in TASK_NAMES}
    offload_e_probs = {t: 0.5 for t in TASK_NAMES}

    if use_variance_reduction:
        cost_observation_counts_c = {t: 2000 for t in TASK_NAMES}

    history = {
        'errors': [], 'costs_client': [], 'costs_edge': [], 'costs_total': [],
        'queue_client': [], 'queue_edge': [],
        'client_weight_entropy': [], 'edge_weight_entropy': [],
        'offload_client': [], 'offload_edge': [], 'feedback_received': [],
        'actual_costs_client': [], 'actual_costs_edge': [],
        'onload_costs': [],
        'onloaded_client_history': [list(onloaded_client)],
        'onloaded_edge_history': [list(onloaded_edge)]
    }

    # --- Main Simulation Loop ---
    for j in range(num_jobs):

        # ---------------------- ONLOADING (Periodic) ----------------------
        if enable_onloading and j > 0 and j % ONLOADING_EPOCH_LENGTH == 0:
            total_tasks = sum(task_counts.values())
            task_dist_array = np.array([task_counts[t] / (total_tasks + 1e-9) for t in TASK_NAMES])

            for t in TASK_NAMES:
                if epoch_task_counts[t] > 0:
                    offload_c_probs[t] = epoch_offload_c_sum[t] / epoch_task_counts[t]
                    offload_e_probs[t] = epoch_offload_e_sum[t] / epoch_task_counts[t]

            # Client Onloading (Uses ESTIMATED error rates)
            task_dist_c = np.array([task_dist_array[i] * (1 - offload_c_probs[TASK_NAMES[i]])
                                    for i in range(NUM_TASK_TYPES)])
            client_model_sizes = MODEL_SIZES[available_client_models]
            client_onload_costs = MODEL_ONLOADING_COSTS[available_client_models]

            onloaded_client, cost_c = enhanced_greedy_onloading(
                V=V_ONLOAD,
                prev_onloaded_models=onloaded_client,
                memory_capacity=CLIENT_MEMORY_CAPACITY,
                model_sizes=client_model_sizes,
                onloading_costs=client_onload_costs,
                error_rates_dict=estimated_error_rates,  # <--- Uses dynamic estimates
                task_dist_estimate=task_dist_c,
                available_models=available_client_models,
                task_names=TASK_NAMES
            )

            # Edge Onloading (Uses ESTIMATED error rates)
            task_dist_e = np.array([task_dist_array[i] * offload_c_probs[TASK_NAMES[i]] *
                                    (1 - offload_e_probs[TASK_NAMES[i]])
                                    for i in range(NUM_TASK_TYPES)])
            edge_model_sizes = MODEL_SIZES[available_edge_models]
            edge_onload_costs = MODEL_ONLOADING_COSTS[available_edge_models]

            onloaded_edge, cost_e = enhanced_greedy_onloading(
                V=V_ONLOAD,
                prev_onloaded_models=onloaded_edge,
                memory_capacity=EDGE_MEMORY_CAPACITY,
                model_sizes=edge_model_sizes,
                onloading_costs=edge_onload_costs,
                error_rates_dict=estimated_error_rates,  # <--- Uses dynamic estimates
                task_dist_estimate=task_dist_e,
                available_models=available_edge_models,
                task_names=TASK_NAMES
            )

            history['onload_costs'].append(cost_c + cost_e)
            history['onloaded_client_history'].append(list(onloaded_client))
            history['onloaded_edge_history'].append(list(onloaded_edge))

            epoch_offload_c_sum = {t: 0.0 for t in TASK_NAMES}
            epoch_offload_e_sum = {t: 0.0 for t in TASK_NAMES}
            epoch_task_counts = {t: 0 for t in TASK_NAMES}
        else:
            history['onload_costs'].append(0)
        # ----------------------------------------------------------------

        # Select Job
        idx = random.randint(0, num_jobs - 1)
        task_type = data['full_data'][idx]['category']
        task_type_idx = TASK_NAMES.index(task_type)
        epoch_task_counts[task_type] += 1

        # Get Expected Error based on CURRENT ESTIMATES (System Belief)
        # This drives the confidence score generation (simulating local execution)
        client_err_est, best_client_model = lowest_avg_error(estimated_error_rates[task_type], onloaded_client)
        edge_err_est, best_edge_model = lowest_avg_error(estimated_error_rates[task_type], onloaded_edge)

        # Generate Confidence Score (Proxy for local error)
        mean_confidence = 1.0 - client_err_est
        std_dev = 0.1
        confidence_Z = np.random.normal(loc=mean_confidence, scale=std_dev)
        confidence_Z = np.clip(confidence_Z, 0, 1)

        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)
        task_counts[task_type] += 1

        # --- CLIENT DECISION ---
        client_offload_experts = expert_thresholds > confidence_Z
        prob_offload_c = np.sum(w_client[task_type_idx, client_offload_experts])
        prob_offload_c = np.clip(prob_offload_c, 1e-5, 1 - 1e-5)
        epoch_offload_c_sum[task_type] += prob_offload_c

        o_c = 1 if np.random.rand() < prob_offload_c else 0

        # --- EDGE DECISION ---
        o_e = 0
        edge_confidence_Z = 0
        prob_offload_e = 0.5  # Default
        if o_c == 1:
            # Generate Edge Confidence
            mean_confidence_e = 1.0 - edge_err_est
            edge_confidence_Z = np.random.normal(loc=mean_confidence_e, scale=std_dev)
            edge_confidence_Z = np.clip(edge_confidence_Z, 0, 1)

            edge_offload_experts = expert_thresholds > edge_confidence_Z
            prob_offload_e = np.sum(w_edge[task_type_idx, edge_offload_experts])
            prob_offload_e = np.clip(prob_offload_e, 1e-5, 1 - 1e-5)

            o_e = 1 if np.random.rand() < prob_offload_e else 0

        epoch_offload_e_sum[task_type] += prob_offload_e

        # --- DETERMINE GROUND TRUTH OUTCOME ---
        # We use the dataset to see if the selected model ACTUALLY failed
        if o_c == 0:
            job_error = int(1 - data['full_data'][idx]['results'][best_client_model])
        elif o_e == 0:
            job_error = int(1 - data['full_data'][idx]['results'][best_edge_model])
        else:
            job_error = 0  # Cloud assumed perfect

        # --- FEEDBACK AND IPS UPDATE ---
        feedback_received = (o_c == 1) and (o_e == 1)

        # Get actual ground truth error for the models that were active
        b_c_true = int(1 - data['full_data'][idx]['results'][best_client_model])
        b_e_true = int(1 - data['full_data'][idx]['results'][best_edge_model])

        # >>>>>> UNBIASED ERROR RATE UPDATE (IPS) <<<<<<
        if feedback_received:
            # 1. Calculate Inverse Propensity Weight (Joint Probability)
            joint_prob = prob_offload_c * prob_offload_e
            ips_weight = 1.0 / (joint_prob + 1e-9)

            # 2. Clip weight to reduce variance (Important!)
            ips_weight = min(ips_weight, MAX_IPS_WEIGHT)

            # 3. Update Client Model Estimate
            old_est_c = estimated_error_rates[task_type][best_client_model]
            # Moving average update weighted by importance sampling
            new_est_c = (1 - ERROR_LEARNING_RATE) * old_est_c + \
                        ERROR_LEARNING_RATE * (b_c_true * ips_weight)
            estimated_error_rates[task_type][best_client_model] = np.clip(new_est_c, 0.0, 1.0)

            # 4. Update Edge Model Estimate
            old_est_e = estimated_error_rates[task_type][best_edge_model]
            new_est_e = (1 - ERROR_LEARNING_RATE) * old_est_e + \
                        ERROR_LEARNING_RATE * (b_e_true * ips_weight)
            estimated_error_rates[task_type][best_edge_model] = np.clip(new_est_e, 0.0, 1.0)

        # --- UPDATE EXPERT WEIGHTS (EXP4) ---
        L_hat_c = np.zeros(num_experts)

        if feedback_received and use_variance_reduction:
            # Update cost estimates for VR
            count_c = cost_observation_counts_c[task_type]
            avg_offloadCost[task_type] = (avg_offloadCost[task_type] * count_c + C_c) / (count_c + 1)
            cost_observation_counts_c[task_type] += 1

        # Calculate Loss Estimators
        # Use ESTIMATED rates for the baselines (b_hat)
        b_hat_c = client_err_est
        b_hat_e = edge_err_est
        C_hat = avg_offloadCost[task_type]

        if use_variance_reduction:
            # VARIANCE REDUCTION LOGIC
            for a in range(num_experts):
                o_hat_c_a = 1 if expert_thresholds[a] > confidence_Z else 0

                L_hat_offload = Q_c * C_hat + v_param * (1 - prob_offload_e) * b_hat_e + prob_offload_e * Q_e * C_hat

                if o_hat_c_a == 0:
                    if feedback_received:
                        residual_error = (b_c_true - b_hat_c) / (prob_offload_c * prob_offload_e)
                        L_hat_c[a] = v_param * (residual_error + b_hat_c)
                    else:
                        L_hat_c[a] = v_param * b_hat_c
                else:
                    if feedback_received:
                        L_offload_actual = Q_c * C_c + v_param * (
                                    1 - prob_offload_e) * b_e_true + prob_offload_e * Q_e * C_c
                        residual_offload = (L_offload_actual - L_hat_offload) / (prob_offload_c * prob_offload_e)
                        L_hat_c[a] = residual_offload + L_hat_offload
                    else:
                        L_hat_c[a] = L_hat_offload

            S_client[task_type_idx] += L_hat_c

        elif feedback_received:
            # STANDARD EXP4 (Partial Feedback)
            for a in range(num_experts):
                o_hat_c_a = 1 if expert_thresholds[a] > confidence_Z else 0
                if o_hat_c_a == 0:
                    L_ideal_c = v_param * b_c_true
                else:
                    L_ideal_c = Q_c * C_c + (1 - prob_offload_e) * v_param * b_e_true + prob_offload_e * Q_e * C_c

                L_hat_c[a] = L_ideal_c / (prob_offload_c * prob_offload_e)

            S_client[task_type_idx] += L_hat_c

        # Update Edge Weights (Only if client offloaded)
        L_hat_e = np.zeros(num_experts)
        if o_c == 1:
            if use_variance_reduction:
                for k in range(num_experts):
                    o_hat_e_k = 1 if expert_thresholds[k] > edge_confidence_Z else 0

                    if o_hat_e_k == 0:
                        if feedback_received:
                            residual_error = (b_e_true - b_hat_e) / prob_offload_e
                            L_hat_e[k] = v_param * (residual_error + b_hat_e)
                        else:
                            L_hat_e[k] = v_param * b_hat_e
                    else:
                        if feedback_received:
                            residual_cost = (C_c - C_hat) / prob_offload_e
                            L_hat_e[k] = Q_e * (residual_cost + C_hat)
                        else:
                            L_hat_e[k] = Q_e * C_hat
                S_edge[task_type_idx] += L_hat_e

            elif feedback_received:
                for k in range(num_experts):
                    o_hat_e_k = 1 if expert_thresholds[k] > edge_confidence_Z else 0
                    if o_hat_e_k == 0:
                        L_ideal_e = v_param * b_e_true
                    else:
                        L_ideal_e = Q_e * C_c
                    L_hat_e[k] = L_ideal_e / prob_offload_e
                S_edge[task_type_idx] += L_hat_e

        # Normalize Weights
        S_client_task = S_client[task_type_idx]
        log_w_client = -learning_rate * S_client_task
        log_w_client_shifted = log_w_client - np.max(log_w_client)
        w_client[task_type_idx] = np.exp(log_w_client_shifted) / np.sum(np.exp(log_w_client_shifted))

        if o_c == 1:
            S_edge_task = S_edge[task_type_idx]
            log_w_edge = -learning_rate * S_edge_task
            log_w_edge_shifted = log_w_edge - np.max(log_w_edge)
            w_edge[task_type_idx] = np.exp(log_w_edge_shifted) / np.sum(np.exp(log_w_edge_shifted))

        # Update Queues
        job_cost_c = o_c * C_c
        job_cost_e = o_c * o_e * C_c
        Q_c = max(0, Q_c + job_cost_c - cost_budget_gamma_c)
        Q_e = max(0, Q_e + job_cost_e - cost_budget_gamma_e)

        # Record History
        history['errors'].append(job_error)
        history['costs_client'].append(job_cost_c)
        history['costs_edge'].append(job_cost_e)
        history['costs_total'].append(job_cost_c + job_cost_e)
        history['queue_client'].append(Q_c)
        history['queue_edge'].append(Q_e)
        history['offload_client'].append(o_c)
        history['offload_edge'].append(o_e)
        history['feedback_received'].append(int(feedback_received))

        # Calculate entropy
        epsilon = 1e-9
        entropy_c = -np.sum(w_client * np.log2(w_client + epsilon), axis=1).mean()
        entropy_e = -np.sum(w_edge * np.log2(w_edge + epsilon), axis=1).mean()
        history['client_weight_entropy'].append(entropy_c)
        history['edge_weight_entropy'].append(entropy_e)

    return history


def aggregate_results(all_histories):
    """Averages results over multiple trials."""
    aggregated = {}
    keys = all_histories[0].keys()
    skip_keys = {'onloaded_client_history', 'onloaded_edge_history'}

    for key in keys:
        if key in skip_keys:
            aggregated[key] = all_histories[0][key]
            continue
        try:
            stacked = np.array([h[key] for h in all_histories])
            aggregated[f'{key}_mean'] = np.mean(stacked, axis=0)
            aggregated[f'{key}_std'] = np.std(stacked, axis=0)
        except ValueError:
            aggregated[key] = all_histories[0][key]
    return aggregated


def plot_performance_comparison(agg_no_vr, agg_vr, cost_budget_gamma_c, cost_budget_gamma_e, num_jobs):
    """Compare performance with and without variance reduction"""
    jobs_axis = np.arange(num_jobs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Variance Reduction Impact: Performance Comparison', fontsize=16)

    # Error Rate
    avg_error_no_vr = np.cumsum(agg_no_vr['errors_mean']) / (jobs_axis + 1)
    avg_error_vr = np.cumsum(agg_vr['errors_mean']) / (jobs_axis + 1)
    axes[0, 0].plot(jobs_axis, avg_error_no_vr, label='No Variance Reduction')
    axes[0, 0].plot(jobs_axis, avg_error_vr, label='With Variance Reduction', linestyle='--')
    axes[0, 0].set_ylabel('Average Error Rate')
    axes[0, 0].legend()

    # Feedback Rate
    avg_feedback_no_vr = np.cumsum(agg_no_vr['feedback_received_mean']) / (jobs_axis + 1)
    avg_feedback_vr = np.cumsum(agg_vr['feedback_received_mean']) / (jobs_axis + 1)
    axes[0, 1].plot(jobs_axis, avg_feedback_no_vr, label='No VR')
    axes[0, 1].plot(jobs_axis, avg_feedback_vr, label='With VR', linestyle='--')
    axes[0, 1].set_ylabel('Feedback Rate')
    axes[0, 1].legend()

    # Client Cost
    avg_cost_c_no_vr = np.cumsum(agg_no_vr['costs_client_mean']) / (jobs_axis + 1)
    avg_cost_c_vr = np.cumsum(agg_vr['costs_client_mean']) / (jobs_axis + 1)
    axes[1, 0].plot(jobs_axis, avg_cost_c_no_vr, label='No VR')
    axes[1, 0].plot(jobs_axis, avg_cost_c_vr, label='With VR', linestyle='--')
    axes[1, 0].axhline(y=cost_budget_gamma_c, color='r', linestyle=':')
    axes[1, 0].set_ylabel('Avg Client Cost')
    axes[1, 0].legend()

    # Edge Cost
    avg_cost_e_no_vr = np.cumsum(agg_no_vr['costs_edge_mean']) / (jobs_axis + 1)
    avg_cost_e_vr = np.cumsum(agg_vr['costs_edge_mean']) / (jobs_axis + 1)
    axes[1, 1].plot(jobs_axis, avg_cost_e_no_vr, label='No VR')
    axes[1, 1].plot(jobs_axis, avg_cost_e_vr, label='With VR', linestyle='--')
    axes[1, 1].axhline(y=cost_budget_gamma_e, color='r', linestyle=':')
    axes[1, 1].set_ylabel('Avg Edge Cost')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    NUM_TRIALS = 5
    NUM_JOBS = 20000
    NUM_EXPERTS = 50
    LEARNING_RATE_ETA = 0.01
    V_PARAM = 500
    COST_BUDGET_GAMMA_C = 0.5
    COST_BUDGET_GAMMA_E = 0.5
    COST_STD = 0.05

    all_results_no_vr, all_results_vr = [], []

    print(f"Running {NUM_TRIALS} trials...")

    for i in range(NUM_TRIALS):
        print(f"Trial {i + 1}")

        # Pass Avg_err as the initial error estimates
        results_no_vr = exp4_hierarchical_algorithm(
            num_jobs=NUM_JOBS, num_experts=NUM_EXPERTS, learning_rate=LEARNING_RATE_ETA,
            v_param=V_PARAM, cost_budget_gamma_c=COST_BUDGET_GAMMA_C,
            cost_budget_gamma_e=COST_BUDGET_GAMMA_E, use_variance_reduction=False,
            cost_std=COST_STD, initial_error_rates=Avg_err
        )
        all_results_no_vr.append(results_no_vr)

        results_vr = exp4_hierarchical_algorithm(
            num_jobs=NUM_JOBS, num_experts=NUM_EXPERTS, learning_rate=LEARNING_RATE_ETA,
            v_param=V_PARAM, cost_budget_gamma_c=COST_BUDGET_GAMMA_C,
            cost_budget_gamma_e=COST_BUDGET_GAMMA_E, use_variance_reduction=True,
            cost_std=COST_STD, initial_error_rates=Avg_err
        )
        all_results_vr.append(results_vr)

    agg_no_vr = aggregate_results(all_results_no_vr)
    agg_vr = aggregate_results(all_results_vr)

    plot_performance_comparison(agg_no_vr, agg_vr, COST_BUDGET_GAMMA_C, COST_BUDGET_GAMMA_E, NUM_JOBS)
import matplotlib.pyplot as plt
from utilities.system import HierarchicalSystemMulti
import os
from baselines import *

estimate_sample_num = 2000

# remaining jobs: datasets (today), baselines (without loss design, Tuesday)
# TODO: define onloading cost when using merged_dataset

# read dataset FILENAME = './merged_ooo_dataset.pkl'
# FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
FILENAME = './merged_ooo_dataset.pkl'
# FILENAME = './data/ooo_dataset/ooo_dataset_pop19.pkl'
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
# ooo_dataset1.pkl
# MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])
# merged_ooo_dataset.pkl
MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0,
                        10000, # GPT4o, ignore
                        10000,
                        27, 78, 1, 7, 16, 7, 3.5, 12, 8, 32, 72, 2.2, 4.5, 1.0, 7
                        ])

# ooo_dataset_pop19
# ['llama2_7b_cpo_details_773', '13_outof_32_pruned_layers_llama3_1_8b_details_3682', 'llama_13b_details_3522', 'autotrain_llama3_70b_orpo_v1_details_1219', 'qwen1_5_1_8b_chat_details_1598', 'calme_2_2_qwen2_7b_details_3186', 'deepseek_r1_distill_qwen_14b_abliterated_v2_details_636', 'deepseek_r1_distill_qwen_32b_details_1975', 'calme_2_1_qwen2_5_72b_details_3442', 'collectivecognition_v1_1_mistral_7b_details_1811', 'merge_mixtral_prometheus_8x7b_details_802', 'dolphin_2_9_3_mistral_nemo_12b_details_786', '3prymmal_phi3_3b_slerp_details_877', 'medphi_4_14b_v1_details_1111', 'athena_gemma_2_2b_it_details_1862', 'athene_codegemma_2_7b_it_alpaca_v1_2_details_524', '4prymmal_gemma2_9b_slerp_details_2671', 'bggpt_gemma_2_27b_it_v1_0_details_1153', 'deepseek_llm_67b_chat_details_1361']
# MODEL_SIZES = np.array([
#     7,    # llama2_7b_cpo_details_773
#     8,    # llama3_1_8b (13_outof_32_pruned_layers_llama3_1_8b_details_3682)
#     13,   # llama_13b_details_3522
#     70,   # llama3_70b
#     1.8,  # qwen1_5_1_8b
#     7,    # qwen2_7b
#     14,   # qwen_14b (deepseek_r1_distill_qwen_14b...)
#     32,   # qwen_32b
#     72,   # qwen2.5_72b
#     7,    # mistral_7b
#     56,   # mixtral_8x7b  -> 8*7 = 56B effective params
#     12,   # mistral_nemo_12b
#     3,    # phi3_3b
#     14,   # phi_14b
#     2,    # gemma_2b
#     7,    # gemma_7b
#     9,    # gemma2_9b
#     27,   # gemma_27b
#     67    # deepseek_llm_67b
# ])

MODEL_ONLOADING_COSTS = MODEL_SIZES
ONLOADING_EPOCH_LENGTH = 1000
V_ONLOAD = 700
BEST_MODEL_IDX=7


def set_all_seeds(seed: int):
    # Python hash seed (affects dict/set iteration in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)


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
            {'memory': 30, 'models': list(range(25)), 'gamma': 0},
            {'memory': 100, 'models': list(range(25)), 'gamma': 0.5},
            {'gamma': 0.3},
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
            {'memory': 30, 'models': list(range(25)), 'gamma': 0},
            {'memory': 80, 'models': list(range(25)), 'gamma': 0.5},
            {'memory': 200, 'models': list(range(25)), 'gamma': 0.5},
            {'gamma': 0.3},
        ],
    }
    configs['5layer_16-8-4-2-1'] = {
        'num_layers': 5, 'nodes_per_layer': [16, 8, 4, 2, 1],
        'layer_configs': [
            {'memory': 30, 'models': list(range(25)), 'gamma': 0},
            {'memory': 80, 'models': list(range(25)), 'gamma': 0.4},
            {'memory': 150, 'models': list(range(25)), 'gamma': 0.4},
            {'memory': 200, 'models': list(range(25)), 'gamma': 0.4},
            {'gamma': 0.3},
        ],
    }
    return configs



def initialize_onloaded_models(node, diverse=False):
    memory_capacity = node.memory_capacity
    model_ids = node.available_models

    if not diverse:
        S = set()
        remaining = memory_capacity
        models = list(model_ids)
        np.random.shuffle(models)
        for m in models:
            if MODEL_SIZES[m] <= remaining:
                S.add(m)
                remaining -= MODEL_SIZES[m]
        node.onloaded_models = list(S)
    else:  # layer-diverse
        layer_num = 3
        layer_level = node.level
        # sort all models based on the model size
        model_order_small2large = np.argsort(MODEL_SIZES, kind="stable")
        parts = np.array_split(model_order_small2large, layer_num)
        models_for_this_layer = parts[layer_level]
        S = set()
        remaining = memory_capacity
        models = models_for_this_layer
        np.random.shuffle(models)
        for m in models:
            if MODEL_SIZES[m] <= remaining:
                S.add(m)
                remaining -= MODEL_SIZES[m]
        node.onloaded_models = list(S)



def enhanced_greedy_onloading(V, prev_models, memory, model_sizes, onload_costs,
                              error_dict, task_dist, available_models, task_names):
    S = []
    remaining = memory
    # print('previous set', prev_models)
    while True:
        best_model_idx, best_net = None, -1.0
        candidates = [i for i in available_models
                      if i not in S and model_sizes[i] <= remaining]
        if not candidates:
            break
        exp_err = []
        for t in task_names:
            if len(S)==0:  # if S is empty
                exp_err.append(1.0)
            else:
                errs = [error_dict[t][i] for i in S]
                exp_err.append(min(errs) if len(errs)!=0 else 1.0)
        for i in candidates:
            gain = 0.0
            S_plus = S + [i]
            for ti, t in enumerate(task_names):
                errs_plus = [error_dict[t][x] for x in S_plus]
                err_plus = min(errs_plus) if len(errs_plus)!=0 else 1.0
                gain += task_dist[ti] * (exp_err[ti] - err_plus)
            cost = onload_costs[i] if i not in prev_models else 0
            net = gain - cost * 0.01
            if net >= 0 and net >= best_net:
                best_net = net
                best_model_idx = i
        if best_model_idx is None:
            break
        S.append(best_model_idx)
        remaining -= model_sizes[best_model_idx]
    total = sum(onload_costs[i] for i, m in enumerate(available_models) if m in S and m not in prev_models)
    # print('onloaded models', S)
    return S, total


def multi_layer_exp4(system, num_jobs, num_experts, learning_rate, v_param,
                     use_variance_reduction=False, enable_onloading=True, initial_error_rates=None,
                     exploration_gamma=0.05, loss_mode='recursive', diverse=False):
    """
    Multi-layer EXP4 (bandits with expert advice) for *multi-parent* offloading.

    Key extension vs. the tree case:
      - Each node n (non-cloud) has K_n candidate parents (upper-layer nodes).
      - Action set size at node n: K_n + 1 (offload-to-parent-k, or stay).
      - Experts are indexed by (k, a): parent-index k and threshold-index a.
        Expert (k, a) chooses:
            offload to parent k  if theta_a > z
            stay (process locally) otherwise
      - Probability over actions:
            p_k   = sum_a w_{k,a} * 1{theta_a > z},  k=0..K_n-1
            p_stay= sum_{k,a} w_{k,a} * 1{theta_a <= z}
        then mix-in exploration: p <- (1-gamma)*p + gamma*Unif(K_n+1)

    Importance sampling:
      - Let P_cloud(n) be the probability that a job starting at node n eventually reaches cloud
        under the current (stochastic) policy. In a layered DAG this is computed recursively:
            P_cloud(cloud)=1
            P_cloud(n)=sum_{k} p_k(n) * P_cloud(parent_k)
      - Cloud feedback arrives iff the realized trajectory reaches cloud; IPS uses 1/P_cloud(start).

    The loss recursion also generalizes by using the *expected* parent-Q and expected parent-exp-loss
    under the conditional distribution over parents given offload.
    """
    oracle_avg_error = 0.0

    expert_thresholds = np.linspace(0, 1, num_experts)

    ERROR_LEARNING_RATE = 1/estimate_sample_num
    MAX_IPS_WEIGHT = 20.0

    # -----------------------
    # Initialize per-node EXP4 tensors
    # -----------------------
    for node in system.get_non_cloud_nodes():
        # K_n parents; experts are (k,a) pairs
        if node.level == system.num_layers - 1:
            continue
        K_n = len(getattr(node, 'parents', []))
        node.K_parents = K_n
        node.num_actions = K_n + 1  # K parents + stay
        node.S = np.zeros((NUM_TASK_TYPES, K_n, num_experts))  # cumulative loss per (k,a)
        node.w = np.ones((NUM_TASK_TYPES, K_n, num_experts)) / (node.num_actions * num_experts + 1e-12)

        initialize_onloaded_models(node, diverse=diverse)

    task_counts = {t: 0 for t in TASK_NAMES}
    node_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    # Track empirical offload probability (sum over parents) per node-task
    node_offload_probs = {n.node_id: {t: 0.5 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_offload_sum = {n.node_id: {t: 0.0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}
    epoch_task_counts = {n.node_id: {t: 0 for t in TASK_NAMES} for n in system.get_non_cloud_nodes()}

    estimated_error_rates_per_node = {n.node_id: copy.deepcopy(initial_error_rates) for n in system.get_non_cloud_nodes()}

    # Variance reduction control variates
    if use_variance_reduction:
        avg_cost_estimate = {t: avg_offloadCost[t] for t in TASK_NAMES}
        cost_obs_count = {t: estimate_sample_num for t in TASK_NAMES}

    history = {
        'errors': [], 'costs_total': [],
        'node_costs': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'node_queues': {n.node_id: [] for n in system.nodes.values() if n.Q is not None},
        'node_offload_decisions': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        'feedback_received': [], 'onload_costs': [], 'execution_layer': [],
        'node_loss_values': {n.node_id: [] for n in system.get_non_cloud_nodes()},
        "oracle_avg_error": 0,
    }

    # Helper: compute action distribution at node given confidence z
    def compute_action_probs(node, task_type_idx, z):
        K = node.K_parents  # how many parents
        if K == 0:
            # No parents (shouldn't happen except cloud). Force stay.
            return np.array([1.0])
        offload_mask = (expert_thresholds > z)  # shape (A,)
        # p_k: sum_a w_{k,a} 1{theta_a > z}
        w_k_a = node.w[task_type_idx]  # ignore the last prob (stay)
        p_parents = np.sum(w_k_a[:, offload_mask], axis=1)  # (K+1,)
        # p_stay: sum_{k,a} w_{k,a} 1{theta_a <= z}
        p_stay = float(np.sum(w_k_a[:, ~offload_mask]))
        p = np.concatenate([p_parents, np.array([p_stay])])
        p = np.clip(p, 1e-12, None)
        p = p / np.sum(p)
        # exploration
        p = (1.0 - exploration_gamma) * p + exploration_gamma * (1.0 / (K + 1))  # the function is defined inside so exploration_gamma is okay
        p = p / np.sum(p)
        return p  # p is a vector

    # Helper: compute P_cloud for all nodes (dynamic programming over levels)
    def compute_prob_to_cloud(all_node_action_probs):
        P_cloud = {}
        cloud = system.get_cloud_node()
        P_cloud[cloud.node_id] = 1.0
        # process from top-1 down to 0
        for level in range(system.num_layers - 2, -1, -1):
            for node in [n for n in system.nodes.values() if n.level == level]:
                p = all_node_action_probs[node.node_id]  # length K+1
                K = node.K_parents
                prob = 0.0
                for k in range(K):
                    parent = node.parents[k]
                    prob += p[k] * P_cloud[parent.node_id]  # conditional probabilty
                P_cloud[node.node_id] = max(prob, 1e-12)
        return P_cloud

    for j in range(num_jobs):
        if j % 5000 == 0 and j > 0:
            print(f"  Job {j}: err={np.mean(history['errors'][-1000:]):.3f}, "
                  f"fb={np.mean(history['feedback_received'][-1000:]):.3f}")

        # -----------------------
        # Onloading (unchanged)
        # -----------------------
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
                local_dist = np.array([task_dist[i] * (1 - node_offload_probs[node.node_id][TASK_NAMES[i]])
                                       for i in range(NUM_TASK_TYPES)])
                new_models, cost = enhanced_greedy_onloading(
                    V_ONLOAD, node.onloaded_models, node.memory_capacity,
                    MODEL_SIZES[node.available_models], MODEL_ONLOADING_COSTS[node.available_models],
                    ERROR_RATES_GT, local_dist, node.available_models, TASK_NAMES
                )
                node.onloaded_models = new_models
                total_onload_cost += cost
            history['onload_costs'].append(total_onload_cost)
            for node in system.get_non_cloud_nodes():
                epoch_offload_sum[node.node_id] = {t: 0.0 for t in TASK_NAMES}
                epoch_task_counts[node.node_id] = {t: 0 for t in TASK_NAMES}
        else:
            history['onload_costs'].append(0)

        # -----------------------
        # Sample a job and starting node
        # -----------------------
        idx = random.randint(0, TOTAL_JOBS - 1)
        task_type = data['full_data'][idx]['category']
        task_type_idx = TASK_NAMES.index(task_type)
        task_counts[task_type] += 1

        oracle_avg_error += np.min(1-np.array(data['full_data'][idx]['results']))

        start_node = random.choice(system.get_leaf_nodes())

        # -----------------------
        # Pre-compute model confidence + action distributions for *all* non-cloud nodes
        # (so we can compute P_cloud via DP)
        # -----------------------
        all_node_best_models = {}
        all_node_confidences = {}
        all_node_action_probs = {}

        for node in system.get_non_cloud_nodes():
            # model error estimate for confidence
            if len(node.onloaded_models) == 0:
                err_exp, best_model = 1.0, None
            else:
                err_exp, best_model = lowest_avg_error(ERROR_RATES_GT[task_type], node.onloaded_models)
                if err_exp is None:
                    err_exp, best_model = 1.0, None
            all_node_best_models[node.node_id] = best_model
            confidence = np.clip(np.random.normal(1.0 - err_exp, 0.1), 0, 1)
            all_node_confidences[node.node_id] = confidence
            node.confidence = confidence  # temp

            # action distribution at this node for this task
            p = compute_action_probs(node, task_type_idx, confidence)
            all_node_action_probs[node.node_id] = p

            node.p_offload = float(np.sum(p[:-1]))  # offload probability (sum over parents)

        # Probability-to-cloud for IPS (under the stochastic policy)
        P_cloud = compute_prob_to_cloud(all_node_action_probs)

        # -----------------------
        # Realized traversal under sampled actions
        # -----------------------
        current_node = start_node
        visited_path = [current_node]
        offload_decisions = {}  # store chosen action index (0..K for each visited node)
        executed_at_node = None

        while True:
            if current_node.level == system.num_layers - 1:
                executed_at_node = current_node
                break

            node_task_counts[current_node.node_id][task_type] += 1
            epoch_task_counts[current_node.node_id][task_type] += 1
            epoch_offload_sum[current_node.node_id][task_type] += float(np.sum(all_node_action_probs[current_node.node_id][:-1]))

            p = all_node_action_probs[current_node.node_id]
            K = current_node.K_parents
            action = int(np.random.choice(np.arange(K + 1), p=p))
            offload_decisions[current_node.node_id] = action

            if action == K:  # stay
                executed_at_node = current_node
                break
            else:
                # offload to selected parent
                next_node = current_node.parents[action]
                current_node.to = next_node
                current_node = next_node
                visited_path.append(current_node)

        # -----------------------
        # Determine job error & feedback
        # -----------------------
        if executed_at_node.level == system.num_layers - 1:
            job_error = 0  # execute at cloud
        else:
            best_model = all_node_best_models.get(executed_at_node.node_id)
            job_error = 1 if best_model is None else int(1 - data['full_data'][idx]['results'][best_model])

        feedback_received = (executed_at_node.level == system.num_layers - 1)
        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)

        # -----------------------
        # IPS update for estimated per-node error rates (only when cloud feedback is observed)
        # -----------------------
        if feedback_received:
            joint_prob = P_cloud[start_node.node_id]
            ips_weight = min(1.0 / (joint_prob + 1e-12), MAX_IPS_WEIGHT)

            for node in system.get_non_cloud_nodes():
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

        # -----------------------
        # EXP4 update set:
        #   - VR: update all non-cloud nodes (learn even without feedback)
        #   - no-VR: update only when feedback_received (cloud)
        # -----------------------
        if use_variance_reduction:
            nodes_to_update = [n for n in system.get_non_cloud_nodes()]
        elif feedback_received:
            nodes_to_update = [n for n in system.get_non_cloud_nodes()]
        else:
            nodes_to_update = []

        # Process higher levels first so parent.exp_loss is available for children
        nodes_to_update = sorted(nodes_to_update, key=lambda n: n.level, reverse=True)

        # Pre-set exp_loss at cloud = 0 for convenience
        system.get_cloud_node().exp_loss = 0.0

        for node in nodes_to_update:
            K = node.K_parents  # how many parents
            node.exp_loss = None
            node.actual_loss = np.zeros((K, num_experts))

            node.p_to_cloud = P_cloud[node.node_id]
            best_model = all_node_best_models.get(node.node_id)
            if best_model is None:
                node.b_hat = 1.0
                node.b_true = 1.0 if feedback_received else 1.0
            else:
                node.b_hat = estimated_error_rates_per_node[node.node_id][task_type][best_model]
                node.b_true = int(1 - data['full_data'][idx]['results'][best_model]) if feedback_received else node.b_hat

            node.confidence = all_node_confidences[node.node_id]
            node.C_true = C_c
            node.C_hat = (avg_cost_estimate[task_type] if use_variance_reduction else C_c)

            # Compute exp_loss once per node (uses node.p_offload + parent expectations)
            loss_recursive_exp_multi(node, v_param, system, all_node_action_probs,
                                     use_variance_reduction, if_feedback=feedback_received, loss_mode=loss_mode)

            # Compute per-expert losses (k,a)
            for k in range(K):
                for a in range(num_experts):
                    loss_recursive_actual_multi(node, v_param, system, k, a, expert_thresholds,
                                                all_node_action_probs, use_variance_reduction,
                                                if_feedback=feedback_received, loss_mode=loss_mode)

            # Accumulate loss and update weights (flatten over (k,a))
            node.S[task_type_idx] += node.actual_loss

            history['node_loss_values'][node.node_id].append(float(np.sum(node.actual_loss)))

            # Normalize weights
            log_w = -learning_rate * node.S[task_type_idx].reshape(-1)
            log_w -= np.max(log_w)
            w_flat = np.exp(log_w)
            w_flat /= np.sum(w_flat)
            node.w[task_type_idx] = w_flat.reshape(K, num_experts)

            # clear temp
        for node in nodes_to_update:
            node.b_true = None
            node.b_hat = None
            node.C_true = None
            node.C_hat = None
            node.actual_loss = None
            node.p_to_cloud = None
            node.confidence = None
            node.exp_loss = None

        # -----------------------
        # Update queues and record metrics (same structure as original)
        # Queue updates only apply to non-initial, non-cloud nodes (Q != None).
        # For multi-parent, the queue update is applied to each *visited* parent node that received the job.
        # -----------------------
        total_job_cost = 0.0
        # record decisions
        for node in system.get_non_cloud_nodes():
            if node.node_id in offload_decisions:
                history['node_offload_decisions'][node.node_id].append(1 if offload_decisions[node.node_id] != node.K_parents else 0)
            else:
                history['node_offload_decisions'][node.node_id].append(0)

        # total cost only when a hop happens
        # Apply cost to each visited transition (node -> chosen parent)
        for t_idx in range(len(visited_path) - 1):
            parent = visited_path[t_idx + 1]
            if parent.Q is not None:
                parent.Q = max(0, parent.Q + C_c - parent.cost_budget_gamma)
            total_job_cost += C_c

        # Record per-node queues and costs
        for node in system.get_non_cloud_nodes():
            history['node_costs'][node.node_id].append(total_job_cost)  # keep as in original (per-job total)
        for node in system.nodes.values():
            if node.Q is not None:
                history['node_queues'][node.node_id].append(node.Q)

        history['errors'].append(job_error)
        history['costs_total'].append(total_job_cost)
        history['feedback_received'].append(1 if feedback_received else 0)
        history['execution_layer'].append(executed_at_node.level)
    history['oracle_avg_error'] = oracle_avg_error/num_jobs

    return history


def loss_recursive_exp_multi(node, v_param, system, all_node_action_probs, use_variance_reduction, if_feedback, loss_mode):
    """
    Expected loss at node under the *current stochastic policy* (not conditioned on any expert).
    With multiple parents, the offload term uses the expected parent Q and expected parent exp_loss
    under the conditional distribution over parents given offload.
    """
    if loss_mode == 'local': # set node.exp_loss to 0
        node.exp_loss = 0.0
        return

    K = node.K_parents
    # action distribution at this node (length K+1)
    p = all_node_action_probs[node.node_id]
    p_offload = float(np.sum(p[:-1]))
    p_offload = max(p_offload, 1e-12)

    # Conditional distribution over parents given offload
    cond = p[:-1] / p_offload  # (K,)
    exp_parent_Q = 0.0
    exp_parent_loss = 0.0
    for k in range(K):
        parent = node.parents[k]
        exp_parent_Q += cond[k] * parent.Q
        exp_parent_loss += cond[k] * parent.exp_loss

    if use_variance_reduction:
        residual_b = (node.b_true - node.b_hat) / (node.p_to_cloud + 1e-12) * if_feedback
        residual_C = (node.C_true - node.C_hat) / (node.p_to_cloud + 1e-12) * if_feedback
        node.exp_loss = (
            v_param * (1 - p_offload) * (residual_b + node.b_hat)
            + p_offload * (exp_parent_Q * (residual_C + node.C_hat) + exp_parent_loss)
        )
    else:
        node.exp_loss = (
            v_param * (1 - p_offload) * node.b_true
            + p_offload * (exp_parent_Q * node.C_true + exp_parent_loss)
        )


def loss_recursive_actual_multi(node, v_param, system, k, a, expert_thresholds,
                               all_node_action_probs, use_variance_reduction, if_feedback, loss_mode):
    """
    Loss for expert (k,a): offload to parent k iff theta_a > confidence, else stay.
    """
    o_hat = 1 if expert_thresholds[a] > node.confidence else 0
    parent = node.parents[k]  # specific parent for this expert group
    if loss_mode == 'local':  # parent.exp_loss should be 0
        assert parent.exp_loss == 0.0

    if use_variance_reduction:
        residual_b = (node.b_true - node.b_hat) / (node.p_to_cloud + 1e-12) * if_feedback
        residual_C = (node.C_true - node.C_hat) / (node.p_to_cloud + 1e-12) * if_feedback
        node.actual_loss[k, a] = (
            v_param * (1 - o_hat) * (residual_b + node.b_hat)
            + o_hat * (parent.Q * (residual_C + node.C_hat)
                       + parent.exp_loss)
        )
    else:
        node.actual_loss[k, a] = (
            v_param * (1 - o_hat) * node.b_true
            + o_hat * ((parent.Q if parent.Q is not None else 0.0) * node.C_true
                       + parent.exp_loss)
        )


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
    to_run = ['3layer_4-2-1']
    # , '5layer_16-8-4-2-1', '3layer_4-2-1', '4layer_1-1-1-1', '4layer_8-4-2-1'

    for name in to_run:
        cfg = configs[name]
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")

        all_no_vr = []
        all_vr = []
        all_no_exp = []
        all_no_exp_no_vr = []
        all_local = []
        all_unif = []
        all_rr = []
        onload = False
        diverse = False
        for trial in range(NUM_TRIALS):
            print(f"Trial {trial + 1}/{NUM_TRIALS}")
            # no variance reduction, with loss expectation
            seed = 11 + trial
            set_all_seeds(seed)
            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'], multi_parent=True)
            res_no = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM, enable_onloading=onload,
                                      use_variance_reduction=False, initial_error_rates=Avg_err, exploration_gamma=0.1, loss_mode='recursive', diverse=diverse)
            all_no_vr.append(res_no)
            print('oracle_avg_error', res_no['oracle_avg_error'])
            # with variance reduction, with loss expectation
            seed = 11 + trial
            set_all_seeds(seed)
            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'], multi_parent=True)
            res_vr = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM, enable_onloading=onload,
                                      use_variance_reduction=True, initial_error_rates=Avg_err, exploration_gamma=0.1, loss_mode='recursive', diverse=diverse)
            all_vr.append(res_vr)
            # with variance reduction, no loss expectation
            seed = 11 + trial
            set_all_seeds(seed)
            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'],
                                             multi_parent=True)
            res_no_exp = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM, enable_onloading=onload,
                                      use_variance_reduction=True, initial_error_rates=Avg_err, exploration_gamma=0.1, loss_mode='local', diverse=diverse)
            all_no_exp.append(res_no_exp)
            # no variance reduction, no loss expectation
            seed = 11 + trial
            set_all_seeds(seed)
            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'],
                                             multi_parent=True)
            res_no_exp_no_vr = multi_layer_exp4(system, NUM_JOBS, NUM_EXPERTS, LEARNING_RATE, V_PARAM, enable_onloading=onload,
                                      use_variance_reduction=False, initial_error_rates=Avg_err, exploration_gamma=0.1, loss_mode='local', diverse=diverse)
            all_no_exp_no_vr.append(res_no_exp_no_vr)

            # baselines
            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'],
                                             multi_parent=True)
            res1 = baseline_all_local(system, NUM_JOBS, data, TASK_NAMES, ERROR_RATES_GT, initial_error_rates=Avg_err, diverse=diverse)
            all_local.append(res1)

            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'],
                                             multi_parent=True)
            res2 = baseline_uniform_random(system, NUM_JOBS, data, TASK_NAMES, ERROR_RATES_GT, p_off=0.3,
                                           initial_error_rates=Avg_err, diverse=diverse)
            all_unif.append(res2)

            system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'],
                                             multi_parent=True)
            res3 = baseline_round_robin(system, NUM_JOBS, data, TASK_NAMES, ERROR_RATES_GT, p_off=0.3,
                                        initial_error_rates=Avg_err, diverse=diverse)
            all_rr.append(res3)

        agg_no = aggregate_results(all_no_vr)
        agg_vr = aggregate_results(all_vr)
        agg_no_exp = aggregate_results(all_no_exp)
        agg_no_exp_no_vr = aggregate_results(all_no_exp_no_vr)
        agg_local = aggregate_results(all_local)
        agg_unif = aggregate_results(all_unif)
        agg_rr = aggregate_results(all_rr)


        # plot
        system = HierarchicalSystemMulti(cfg['num_layers'], cfg['nodes_per_layer'], cfg['layer_configs'], multi_parent=True)
        plot_results(agg_no, agg_vr, all_no_vr, all_vr, system, NUM_JOBS, prefix=name)
        plot_layer0_loss(all_no_vr, all_vr, system, prefix=name)



        print(f"\n--- {name} Results ---")
        print(
            f"No VR  - Error: {np.mean(agg_no['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_no['feedback_received_mean'][-2000:]):.4f}")
        print(
            f"VR     - Error: {np.mean(agg_vr['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_vr['feedback_received_mean'][-2000:]):.4f}")
        print(
            f"No Exp - Error: {np.mean(agg_no_exp['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_no_exp['feedback_received_mean'][-2000:]):.4f}")
        print(
            f"No Exp No VR - Error: {np.mean(agg_no_exp_no_vr['errors_mean'][-2000:]):.4f}, Feedback: {np.mean(agg_no_exp_no_vr['feedback_received_mean'][-2000:]):.4f}")
        print(f"Local  - Error: {np.mean(agg_local['errors_mean'][-2000:]):.4f}")
        print(f"Uniform - Error: {np.mean(agg_unif['errors_mean'][-2000:]):.4f}")
        print(f"Round Robin - Error: {np.mean(agg_rr['errors_mean'][-2000:]):.4f}")
    print("\n" + "=" * 60 + "\nAll experiments completed!\n" + "=" * 60)
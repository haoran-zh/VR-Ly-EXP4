"""
Non-learning routing baselines for hierarchical offloading system.

Implements:
  1) All-Local (no offloading)
  2) Uniform Random Offloading
  3) Round-Robin Offloading

These baselines preserve:
  - Traversal logic
  - Queue updates
  - Onloading optimization
  - Logging format

No learning / no EXP4 updates are performed.
"""

import numpy as np
import random
import copy

# -------------------------------------------------
# REQUIRED IMPORTS FROM YOUR PROJECT
# -------------------------------------------------
# from system import *
from utilities.model_performance import *
from offloadOnload_multi import (
    initialize_onloaded_models,
    enhanced_greedy_onloading,
)

# -------------------------------------------------
# GLOBAL PARAMETERS (can be overridden externally)
# -------------------------------------------------

# Offloading probability for random / RR baselines
DEFAULT_P_OFF = 0.3

# Onloading update frequency
ONLOADING_EPOCH_LENGTH = 500

# Onloading weight (same meaning as your EXP4 code)
V_ONLOAD = 700

# Cost scaling
OffloadCost_SCALE = 0.001

# MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0,
#                         0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0,
#                         0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0])
MODEL_SIZES = np.array([0.7, 1.5, 7.0, 16.0, 67.0, 0.5, 0.5, 72.0,
                        10000, # GPT4o, ignore
                        10000,
                        27, 78, 1, 7, 16, 7, 3.5, 12, 8, 32, 72, 2.2, 4.5, 1.0, 7
                        ])
MODEL_ONLOADING_COSTS = MODEL_SIZES

# =================================================
# INTERNAL UTILITIES
# =================================================

def _init_nodes(system, enable_onloading=True):
    """Initialize per-node fields required by baselines."""
    for node in system.get_non_cloud_nodes():

        if node.level == system.num_layers - 1:
            continue

        K = len(getattr(node, "parents", []))
        node.K_parents = K
        node.num_actions = K + 1

        if enable_onloading and len(node.available_models) > 0:
            node.onloaded_models = initialize_onloaded_models(
                node.memory_capacity,
                node.available_models,
            )
        else:
            node.onloaded_models = list(node.available_models)

        # Round-robin pointer
        if not hasattr(node, "rr_ptr"):
            node.rr_ptr = 0


def _make_history(system):
    """Create history dictionary matching EXP4 format."""
    return {
        "errors": [],
        "costs_total": [],
        "node_costs": {n.node_id: [] for n in system.get_non_cloud_nodes()},
        "node_queues": {n.node_id: [] for n in system.nodes.values() if n.Q is not None},
        "node_offload_decisions": {n.node_id: [] for n in system.get_non_cloud_nodes()},
        "feedback_received": [],
        "onload_costs": [],
        "execution_layer": [],
        "node_loss_values": {n.node_id: [] for n in system.get_non_cloud_nodes()},
    }


# =================================================
# CORE SIMULATION LOOP (SHARED BY ALL BASELINES)
# =================================================

def _run_baseline(
    system,
    num_jobs,
    data,
    TASK_NAMES,
    ERROR_RATES_GT,
    enable_onloading=True,
    initial_error_rates=None,
    p_off=DEFAULT_P_OFF,
    policy="uniform_random",
):
    """
    Common simulation loop for non-learning baselines.

    policy ∈ {"all_local", "uniform_random", "round_robin"}
    """

    _init_nodes(system, enable_onloading)
    history = _make_history(system)

    NUM_TASK_TYPES = len(TASK_NAMES)
    TOTAL_JOBS = len(data["full_data"])

    node_task_counts = {
        n.node_id: {t: 0 for t in TASK_NAMES}
        for n in system.get_non_cloud_nodes()
    }

    epoch_offload_sum = copy.deepcopy(node_task_counts)
    epoch_task_counts = copy.deepcopy(node_task_counts)

    node_offload_probs = {
        n.node_id: {t: 0.5 for t in TASK_NAMES}
        for n in system.get_non_cloud_nodes()
    }

    estimated_error_rates_per_node = {
        n.node_id: copy.deepcopy(initial_error_rates)
        for n in system.get_non_cloud_nodes()
    }

    # =================================================
    # MAIN LOOP
    # =================================================
    for j in range(num_jobs):

        # -------------------------
        # ONLOADING UPDATE
        # -------------------------
        if enable_onloading and j > 0 and j % ONLOADING_EPOCH_LENGTH == 0:

            total_onload_cost = 0.0

            for node in system.get_non_cloud_nodes():

                if len(node.available_models) == 0:
                    continue

                total_tasks = sum(node_task_counts[node.node_id].values())
                task_dist = np.array([
                    node_task_counts[node.node_id][t] / (total_tasks + 1e-9)
                    for t in TASK_NAMES
                ])

                for t in TASK_NAMES:
                    if epoch_task_counts[node.node_id][t] > 0:
                        node_offload_probs[node.node_id][t] = (
                            epoch_offload_sum[node.node_id][t]
                            / epoch_task_counts[node.node_id][t]
                        )

                local_dist = np.array([
                    task_dist[i] * (1.0 - node_offload_probs[node.node_id][TASK_NAMES[i]])
                    for i in range(NUM_TASK_TYPES)
                ])

                new_models, cost = enhanced_greedy_onloading(
                    V_ONLOAD,
                    node.onloaded_models,
                    node.memory_capacity,
                    MODEL_SIZES[node.available_models],
                    MODEL_ONLOADING_COSTS[node.available_models],
                    estimated_error_rates_per_node[node.node_id],
                    local_dist,
                    node.available_models,
                    TASK_NAMES,
                )

                node.onloaded_models = new_models
                total_onload_cost += cost

            history["onload_costs"].append(total_onload_cost)

            # reset epoch stats
            for node in system.get_non_cloud_nodes():
                epoch_offload_sum[node.node_id] = {t: 0.0 for t in TASK_NAMES}
                epoch_task_counts[node.node_id] = {t: 0 for t in TASK_NAMES}

        else:
            history["onload_costs"].append(0.0)

        # -------------------------
        # SAMPLE JOB
        # -------------------------
        idx = random.randint(0, TOTAL_JOBS - 1)
        task_type = data["full_data"][idx]["category"]

        start_node = random.choice(system.get_leaf_nodes())

        current_node = start_node
        visited = [current_node]
        offload_decisions = {}

        # -------------------------
        # ROUTING LOOP
        # -------------------------
        while True:

            if current_node.level == system.num_layers - 1:
                executed_node = current_node
                break

            node_task_counts[current_node.node_id][task_type] += 1
            epoch_task_counts[current_node.node_id][task_type] += 1

            K = current_node.K_parents

            # ===== SELECT ACTION =====
            if policy == "all_local" or K == 0:
                action = K

            elif policy == "uniform_random":
                if np.random.rand() < p_off:
                    action = np.random.randint(0, K)
                else:
                    action = K

            elif policy == "round_robin":
                if np.random.rand() < p_off:
                    action = current_node.rr_ptr % K
                    current_node.rr_ptr = (current_node.rr_ptr + 1) % K
                else:
                    action = K

            else:
                raise ValueError("Unknown baseline policy")

            epoch_offload_sum[current_node.node_id][task_type] += float(action != K)
            offload_decisions[current_node.node_id] = action

            if action == K:
                executed_node = current_node
                break
            else:
                next_node = current_node.parents[action]
                current_node = next_node
                visited.append(current_node)

        # -------------------------
        # EXECUTION ERROR
        # -------------------------
        if executed_node.level == system.num_layers - 1:
            job_error = 0
        else:
            err_exp, best_model = lowest_avg_error(
                ERROR_RATES_GT[task_type],
                executed_node.onloaded_models,
            )
            if best_model is None:
                job_error = 1
            else:
                job_error = int(
                    1 - data["full_data"][idx]["results"][best_model]
                )

        feedback = int(executed_node.level == system.num_layers - 1)

        # -------------------------
        # COST + QUEUE UPDATE
        # -------------------------
        C_c = convert_offloadingCost(data, sample_idx=idx, scale=OffloadCost_SCALE)
        total_cost = 0.0

        for t in range(len(visited) - 1):
            parent = visited[t + 1]
            if parent.Q is not None:
                parent.Q = max(0.0, parent.Q + C_c - parent.cost_budget_gamma)
            total_cost += C_c

        # -------------------------
        # LOGGING
        # -------------------------
        for node in system.get_non_cloud_nodes():
            history["node_costs"][node.node_id].append(total_cost)
            history["node_loss_values"][node.node_id].append(0.0)

            if node.node_id in offload_decisions:
                history["node_offload_decisions"][node.node_id].append(
                    int(offload_decisions[node.node_id] != node.K_parents)
                )
            else:
                history["node_offload_decisions"][node.node_id].append(0)

        for node in system.nodes.values():
            if node.Q is not None:
                history["node_queues"][node.node_id].append(node.Q)

        history["errors"].append(job_error)
        history["costs_total"].append(total_cost)
        history["feedback_received"].append(feedback)
        history["execution_layer"].append(executed_node.level)

    return history


# =================================================
# PUBLIC BASELINE APIS
# =================================================

def baseline_all_local(system, num_jobs, data, TASK_NAMES, ERROR_RATES_GT,
                       enable_onloading=True, initial_error_rates=None):
    return _run_baseline(
        system,
        num_jobs,
        data,
        TASK_NAMES,
        ERROR_RATES_GT,
        enable_onloading,
        initial_error_rates,
        p_off=0.0,
        policy="all_local",
    )


def baseline_uniform_random(system, num_jobs, data, TASK_NAMES, ERROR_RATES_GT,
                            p_off=DEFAULT_P_OFF,
                            enable_onloading=True,
                            initial_error_rates=None):
    return _run_baseline(
        system,
        num_jobs,
        data,
        TASK_NAMES,
        ERROR_RATES_GT,
        enable_onloading,
        initial_error_rates,
        p_off=p_off,
        policy="uniform_random",
    )


def baseline_round_robin(system, num_jobs, data, TASK_NAMES, ERROR_RATES_GT,
                         p_off=DEFAULT_P_OFF,
                         enable_onloading=True,
                         initial_error_rates=None):
    return _run_baseline(
        system,
        num_jobs,
        data,
        TASK_NAMES,
        ERROR_RATES_GT,
        enable_onloading,
        initial_error_rates,
        p_off=p_off,
        policy="round_robin",
    )

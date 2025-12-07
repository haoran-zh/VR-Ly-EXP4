import pickle as pkl
import random
import numpy as np

#FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
# with open(FILENAME, 'rb') as f:
#     data = pkl.load(f)

def estimate_model_perforance(data):
    """
    Estimates model performance based on a random sample of data.
    Ensures ALL task categories exist in the output, defaulting to 0.5 if not sampled.
    """
    full_data = data['full_data']
    total_sample_num = data['total_samples']
    all_indices = list(range(total_sample_num))
    model_num = len(data['available_models'])
    all_categories = data['task_types']  # Get list of all possible task types

    # Randomly select 500 samples
    sample_num = 500
    sampled_indices = random.sample(all_indices, sample_num)

    # Initialize accumulators for ALL categories to ensure coverage
    # We use temporary dictionaries to hold sums and counts
    category_sums = {cat: np.zeros(model_num) for cat in all_categories}
    category_counts = {cat: 0 for cat in all_categories}

    # Process the sampled data
    for sample_idx in sampled_indices:
        query = full_data[sample_idx]
        category = query['category']
        results = np.array(query['results'])

        # Accumulate data if the category is recognized
        if category in category_sums:
            category_sums[category] += results
            category_counts[category] += 1

    model_avg_performance = {}
    model_err_performance = {}

    # Calculate final averages with fallback logic
    for cat in all_categories:
        if category_counts[cat] > 0:
            # If we have samples, calculate the real average
            avg_acc = category_sums[cat] / category_counts[cat]
        else:
            # If we missed this category in sampling, default to 0.5
            avg_acc = np.ones(model_num) * 0.5

        model_avg_performance[cat] = avg_acc
        model_err_performance[cat] = 1.0 - avg_acc

    return model_avg_performance, model_err_performance


def convert_offloadingCost(data, sample_idx, scale):
    full_data = data['full_data']
    return full_data[sample_idx]['length'] * scale


def estimate_offloadingCost(data, scale):
    """
    Estimates offloading cost based on a random sample.
    If a category is missing in the sample, defaults to the average cost of all observed samples.
    """
    full_data = data['full_data']
    total_sample_num = data['total_samples']
    all_indices = list(range(total_sample_num))
    all_categories = data['task_types']  # Ensure we know all possible categories

    # Randomly select 500 samples
    sample_num = 500
    sampled_indices = random.sample(all_indices, sample_num)

    # Initialize accumulators for ALL categories
    category_sums = {cat: 0.0 for cat in all_categories}
    category_counts = {cat: 0 for cat in all_categories}

    # Track global stats for the fallback value
    total_observed_cost = 0.0
    total_observed_count = 0

    for sample_idx in sampled_indices:
        query = full_data[sample_idx]
        category = query['category']
        cost = query['length'] * scale

        # Accumulate per-category
        if category in category_sums:
            category_sums[category] += cost
            category_counts[category] += 1

        # Accumulate global (for fallback)
        total_observed_cost += cost
        total_observed_count += 1

    # Calculate fallback cost (average of all observed data)
    if total_observed_count > 0:
        fallback_cost = total_observed_cost / total_observed_count
    else:
        fallback_cost = 0.0  # Should not happen with 500 samples

    avg_offloadCost = {}

    # Calculate final averages
    for cat in all_categories:
        if category_counts[cat] > 0:
            avg_offloadCost[cat] = category_sums[cat] / category_counts[cat]
        else:
            # Use the global average for missing categories
            avg_offloadCost[cat] = fallback_cost

    return avg_offloadCost

def lowest_avg_error(error_rates, available_client_models):
    lowest_error_rate = 1
    lowest_error_model = None
    for model in available_client_models:
        if error_rates[model] <= lowest_error_rate:
            lowest_error_rate = error_rates[model]
            lowest_error_model = model
    return lowest_error_rate, lowest_error_model
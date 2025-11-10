import pickle as pkl
import random
import numpy as np

#FILENAME = './data/ooo_dataset/ooo_dataset1.pkl'
# with open(FILENAME, 'rb') as f:
#     data = pkl.load(f)

def estimate_model_perforance(data):
    # estimate the average performance of each model for each task
    full_data = data['full_data']
    total_sample_num = data['total_samples']
    all_indices = list(range(total_sample_num))
    model_num = len(data['available_models'])
    # randomly select 2000 samples
    sample_num = 2000
    # create 2000 random indices, range from 0 to total_sample_num
    sampled_indices = random.sample(all_indices, sample_num)
    model_avg_performance = {}
    task_counts = {}
    for sample_idx in sampled_indices:
        query = full_data[sample_idx]
        category = query['category']
        # check if this category has existed in model_avg_performance
        if category not in model_avg_performance: # if we meet this category for the first time
            model_avg_performance[category] = np.zeros(model_num)
            avg_performance = np.array(query['results'])
            model_avg_performance[category] += avg_performance
            task_counts[category] = 1
        else:
            model_avg_performance[category] += np.array(query['results'])
            task_counts[category] += 1
    # average acc for each task-model
    # print(f'total categories: {len(model_avg_performance)}')
    for category in model_avg_performance.keys():
        model_avg_performance[category] /= task_counts[category]
        # print(f'category: {category}, performance: {model_avg_performance[category]}')
    # convert the avg acc (1 is the best) to avg error rate (0 is the best)
    model_err_performance = {}
    for category in model_avg_performance.keys():
        model_err_performance[category] = 1 - model_avg_performance[category]
    return model_avg_performance, model_err_performance


def convert_offloadingCost(data, sample_idx, scale):
    full_data = data['full_data']
    return full_data[sample_idx]['length'] * scale



def estimate_offloadingCost(data, scale):
    # estimate the average performance of each model for each task
    full_data = data['full_data']
    total_sample_num = data['total_samples']
    all_indices = list(range(total_sample_num))
    # randomly select 2000 samples
    sample_num = 2000
    # create 2000 random indices, range from 0 to total_sample_num
    sampled_indices = random.sample(all_indices, sample_num)
    avg_offloadCost = {}
    task_counts = {}
    for sample_idx in sampled_indices:
        query = full_data[sample_idx]
        category = query['category']
        # check if this category has existed in model_avg_performance
        if category not in avg_offloadCost: # if we meet this category for the first time
            avg_offloadCost[category] = 0
            cost = query['length']
            avg_offloadCost[category] += cost * scale
            task_counts[category] = 1
        else:
            avg_offloadCost[category] += query['length'] * scale
            task_counts[category] += 1
    # average acc for each task-model
    # print(f'total categories: {len(avg_offloadCost)}')
    for category in avg_offloadCost.keys():
        avg_offloadCost[category] /= task_counts[category]
        # print(f'category: {category}, performance: {avg_offloadCost[category]}')
    # convert the avg acc (1 is the best) to avg error rate (0 is the best)
    return avg_offloadCost

def lowest_avg_error(error_rates, available_client_models):
    lowest_error_rate = 1
    lowest_error_model = None
    for model in available_client_models:
        if error_rates[model] <= lowest_error_rate:
            lowest_error_rate = error_rates[model]
            lowest_error_model = model
    return lowest_error_rate, lowest_error_model
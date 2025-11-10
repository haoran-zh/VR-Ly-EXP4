import pickle
import numpy as np
# from utils import convert_arrays_to_shapes # Not used in this snippet
from huggingface_hub import hf_hub_download
import os
import zipfile

# (Your unzip_file function and commented-out code remain the same)
# ...

# ----------------reorganize data----------------

# (Your data_options and model_keywords lists remain the same)
data_options = ['bbh', 'gpqa', 'musr', 'math_lv_5', 'mmlu_pro']
model_keywords = ['openai-community__gpt2-details',
                  'openai-community__gpt2-large-details',
                  'openai-community__gpt2-xl-details',
                  'deepseek-ai__deepseek-llm-7b',
                  'deepseek-ai__deepseek-moe-16b',
                  'deepseek-ai__deepseek-llm-67b',
                  'Qwen__Qwen2.5-0.5B-Instruct-details',
                  'Qwen__Qwen1.5-0.5B-Chat',
                  'Qwen__Qwen2.5-32B',
                  'Qwen__Qwen2.5-72B',
                  ]

# No longer needed, as this logic is now inside the loop
# def get_model_result(model_file, dataset_folder, subdataset, idx):
#     ...

ooo_dataset = {}
ooo_dataset['available_models'] = model_keywords
ooo_dataset['available_datasets'] = data_options
ooo_dataset['full_data'] = []
task_types = []

for dataset_idx in data_options:
    print(f"Processing dataset: {dataset_idx}")
    data_folder = f'./data/ooo_dataset/full_data/{dataset_idx}/'
    models_list = os.listdir(data_folder)

    # --- 1. Pre-load all model data for this dataset ---
    # This loop runs ONCE per dataset, loading all models into RAM
    print(f"  Loading all model files for {dataset_idx} into memory...")
    loaded_models_data = {}
    model_file_map = {}  # Map keyword to the actual file name found

    for model_idx in model_keywords:
        try:
            # Find the corresponding filename
            model_file = next(
                filename for filename in models_list if model_idx in filename)
            model_file_map[model_idx] = model_file

            # Load the file ONCE and store it in the dictionary
            file_path = os.path.join(data_folder, model_file)
            with open(file_path, 'rb') as f:
                loaded_models_data[model_file] = pickle.load(f)

        except StopIteration:
            print(f"  ⚠️ Warning: No file found for keyword '{model_idx}' in '{dataset_idx}'")
        except Exception as e:
            print(f"  ❌ Error loading {model_file}: {e}")

    print(f"  ✅ All model data for {dataset_idx} loaded.")

    # --- 2. Get subdataset keys from the first loaded model ---
    if not model_file_map:
        print(f"  ❌ No models loaded for {dataset_idx}, skipping.")
        continue

    # Get the filename of the first successfully loaded model
    first_model_key = model_keywords[0]
    first_model_file = model_file_map.get(first_model_key)

    if not first_model_file:  # Fallback if first model in list wasn't found
        first_model_file = list(loaded_models_data.keys())[0]

    first_model_result = loaded_models_data[first_model_file]
    subdataset_keys = list(first_model_result.keys())
    task_types.extend(subdataset_keys)

    # --- 3. Iterate through subdatasets ---
    for subdata_idx in subdataset_keys:
        num_samples = len(first_model_result[subdata_idx]['doc_id'])
        print(f'  Processing subdataset {subdata_idx} ({num_samples} samples)')

        # --- 4. Pre-process results for this ENTIRE subdataset ---
        # This dictionary will hold the np.array of results for each model
        # e.g., processed_results['deepseek-ai__deepseek-llm-7b'] = [1, 0, 1, ..., 1]
        processed_results = {}

        for model_idx, model_file in model_file_map.items():
            model_result = loaded_models_data[model_file]

            # This logic is from your get_model_result function
            acc_key = list(model_result[subdata_idx])[-1]
            try:
                # Get the *entire* array of results at once
                results_array = np.array(model_result[subdata_idx][acc_key])
            except Exception as e:
                print(f"    Error converting array for {model_file}, {subdata_idx}: {e}")
                # Fallback for complex object arrays
                results_array = np.array(model_result[subdata_idx][acc_key], dtype=object)

            results_array[results_array == True] = 1
            results_array[results_array == False] = 0
            results_array[results_array == 'none'] = 0

            # Store the cleaned, full array
            processed_results[model_idx] = results_array.astype(int)

        # --- 5. Iterate through samples (now very fast) ---
        # This loop now only does dictionary and array lookups (RAM access)
        for sample_idx in range(num_samples):
            query = {}
            query['category'] = dataset_idx + '-' + subdata_idx
            query['idx'] = sample_idx

            model_results = []
            for model_idx in model_keywords:
                if model_idx in processed_results:
                    # Just get the result from the pre-processed array
                    model_results.append(processed_results[model_idx][sample_idx])
                else:
                    # Handle case where model was missing for this dataset
                    model_results.append(None)  # Or 0, depending on preference

            query['results'] = model_results
            ooo_dataset['full_data'].append(query)

    # --- 6. Clear memory for the next dataset ---
    del loaded_models_data
    del processed_results
    print(f"  Finished dataset {dataset_idx}, cleared from memory.")

ooo_dataset['task_types'] = task_types
ooo_dataset['total_samples'] = len(ooo_dataset['full_data'])

# pkl save ooo_dataset
print("Saving final aggregated dataset...")
with open(f'data/ooo_dataset/ooo_dataset1.pkl', 'wb') as f:
    pickle.dump(ooo_dataset, f)

print("✅ All done.")
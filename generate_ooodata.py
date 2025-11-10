import pickle
import numpy as np
# from utils import convert_arrays_to_shapes # Not used in this snippet
from huggingface_hub import hf_hub_download
import os
import zipfile


def unzip_file(zip_path, extract_to_dir):
    """Unzips a single file to a specified directory."""
    try:
        print(f"  Unzipping {os.path.basename(zip_path)} to {extract_to_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(extract_to_dir)
        print("  ✅ Unzip successful.")

        os.remove(zip_path)
        print("  🗑️ Zip file removed.")

    except FileNotFoundError:
        print(f"  ❌ Error: ZIP file not found at {zip_path}")
    except zipfile.BadZipFile:
        print(f"  ❌ Error: File is not a valid ZIP file: {zip_path}")
    except Exception as e:
        print(f"  ❌ An unexpected error occurred during unzipping: {e}")

#
# # display the pre-built router dataset
# # print(convert_arrays_to_shapes(router_dataset))
#
# print(f"Keys: {router_dataset.keys()}")

# print(f"data: {router_dataset['easy'][5]['strong_to_weak']['data']}")
# print(f"model: {router_dataset['easy'][5]['strong_to_weak']['model']}")

# print(f"model: {router_dataset['hard'][1000]['strong_to_weak']['model']}")
# print(f"model: {router_dataset['split_index']}")  # how they split the dataset into train, val, test
# print(f"model: {router_dataset['embedding']['train_embed']}")
# print(f"model: {router_dataset['prompt']['train_prompt']}")
#

# data_options = ['arc_challenge', 'hellaswag', 'mmlu', 'winogrande',
#                 'gsm8k', 'truthfulqa', 'ifeval', 'bbh', 'gpqa',
#                 'musr', 'math_lv_5', 'mmlu_pro']

# for 'ifeval', 'bbh', 'gpqa', 'musr', 'math_lv_5', 'mmlu_pro'



def download_data():
    data_options = ['ifeval', 'bbh', 'gpqa', 'musr', 'math_lv_5', 'mmlu_pro']
    model_keywords = ['openai-community__gpt2-details',
                    'openai-community__gpt2-large',
                    'gpt2-xl',
                        'deepseek-llm-7b',
                      'deepseek-ai__deepseek-moe-16b',
                      'deepseek-ai__deepseek-llm-67b',
                      'Qwen__Qwen2-0.5B',
                      'Qwen__Qwen1.5-0.5B',
                      'Qwen__Qwen2-72B',
                      ]

    # load the pre-built router dataset
    with open(f'data/model_list_per_dataset.pkl', 'rb') as f:
        model_list_per_dataset = pickle.load(f)
    # ----------------download data----------------
        repo_id = "linggm/RouterEval"
        for dataset_idx in data_options:
            for model_keyword in model_keywords:
                try:
                    target_file = next(filename for filename in model_list_per_dataset[dataset_idx] if model_keyword in filename)
                except:
                    print(f"No file found for keyword '{model_keyword}' in '{dataset_idx}'")
                    exit()

                file_name = f"full_data/{dataset_idx}/{target_file}"
                target_directory = "./data/ooo_dataset"

                # 2. Use the local_dir parameter to specify the download location
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    repo_type="dataset",
                    local_dir=target_directory
                )
                extraction_dir = os.path.dirname(local_path)
                unzip_file(local_path, extraction_dir)



# ----------------reorganize data----------------

# (Your data_options and model_keywords lists remain the same)
def organize_data():
    data_options = ['arc_challenge', 'hellaswag', 'mmlu', 'winogrande',
                    'gsm8k', 'ifeval', 'bbh', 'gpqa',
                    'musr', 'math_lv_5', 'mmlu_pro']
    model_keywords = [
                    'openai-community__gpt2-large',
                    'gpt2-xl',
                        'deepseek-llm-7b',
                      'deepseek-ai__deepseek-moe-16b',
                      'deepseek-ai__deepseek-llm-67b',
                      'Qwen__Qwen2-0.5B',
                      'Qwen__Qwen1.5-0.5B',
                      'Qwen__Qwen2-72B',
                      ]
    # No longer needed, as this logic is now inside the loop
    # def get_model_result(model_file, dataset_folder, subdataset, idx):
    #     ...

    ooo_dataset = {}
    ooo_dataset['available_models'] = model_keywords
    ooo_dataset['available_datasets'] = data_options
    ooo_dataset['full_data'] = []
    task_types = []

    for cnt, dataset_idx in enumerate(data_options):
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

        if not first_model_file:  # Fallback if the first model in the list wasn't found
            first_model_file = list(loaded_models_data.keys())[0]

        first_model_result = loaded_models_data[first_model_file]
        subdataset_keys = list(first_model_result.keys())
        task_types.extend(subdataset_keys)

        # --- 3. Iterate through subdatasets ---
        for subdata_idx in subdataset_keys:
            if cnt > 4:
                num_samples = len(first_model_result[subdata_idx]['doc_id'])
            else:
                num_samples = len(first_model_result[subdata_idx]['metrics'])
                # print(ooo_dataset['harness|hellaswag|10']['metrics'])
            print(f'  Processing subdataset {subdata_idx} ({num_samples} samples)')

            # --- 4. Pre-process results for this ENTIRE subdataset ---
            # This dictionary will hold the np.array of results for each model

            processed_results = {}

            for model_idx, model_file in model_file_map.items():
                model_result = loaded_models_data[model_file]

                # This logic is from your get_model_result function
                if cnt > 4:
                    if subdata_idx == 'ifeval':
                        acc_key = 'prompt_level_loose_acc'
                    else:
                        acc_key = list(model_result[subdata_idx])[-1]
                else:
                    acc_key = 'metrics'
                # try:
                    # Get the *entire* array of results at once
                if cnt > 4:
                    results_array = np.array(model_result[subdata_idx][acc_key])
                else:
                    if model_file == 'details_Rachneet__gpt2-xl-alpaca.pkl':
                        results_array = np.array(model_result[subdata_idx]['acc'])
                    else:
                        results_array = np.array([model_result[subdata_idx][acc_key][i]['acc'] for i in range(num_samples)])
                # except Exception as e:
                #     print(f"    Error converting array for {model_file}, {subdata_idx}: {e}")
                #     # Fallback for complex object arrays
                #     results_array = np.array(model_result[subdata_idx][acc_key], dtype=object)

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

# download_data()
# organize_data()
# read ooo_dataset
# /home/hz8556/llmooo/data/ooo_dataset/full_data/arc_challenge/details_deepseek-ai__deepseek-moe-16b-base.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/ifeval/deepseek-ai__deepseek-llm-67b-chat-details_1361.pkl
with open(f'data/ooo_dataset/ooo_dataset1.pkl', 'rb') as f:
    ooo_dataset = pickle.load(f)
# dict_keys(['choices', 'cont_tokens', 'example', 'full_prompt', 'gold',
# 'gold_index', 'input_tokens', 'instruction', 'metrics', 'num_asked_few_shots',
# 'num_effective_few_shots', 'padded', 'pred_logits', 'predictions', 'truncated'])
# print(ooo_dataset['harness|truthfulqa:mc|0']['predictions'])
print(len(ooo_dataset['task_types']))
print(ooo_dataset['total_samples'])
print(ooo_dataset['full_data'][28769])
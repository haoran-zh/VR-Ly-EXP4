import pickle
import numpy as np
from utils import convert_arrays_to_shapes
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

# load the pre-built router dataset
# with open(f'data/model_list_per_dataset.pkl', 'rb') as f:
#     model_list_per_dataset = pickle.load(f)


# ----------------download data----------------
# repo_id = "linggm/RouterEval"
# for dataset_idx in data_options:
#     for model_keyword in model_keywords:
#         target_file = next(filename for filename in model_list_per_dataset[dataset_idx] if model_keyword in filename)
#
#         file_name = f"full_data/{dataset_idx}/{target_file}"
#         target_directory = "./data/ooo_dataset"
#
#         # 2. Use the local_dir parameter to specify the download location
#         local_path = hf_hub_download(
#             repo_id=repo_id,
#             filename=file_name,
#             repo_type="dataset",
#             local_dir=target_directory
#         )
#         extraction_dir = os.path.dirname(local_path)
#         unzip_file(local_path, extraction_dir)
# ------------------------------------------------


# ----------------reorganize data----------------
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/Qwen__Qwen2.5-32B-Instruct-details_1488.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/openai-community__gpt2-xl-details_754.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/deepseek-ai__deepseek-llm-7b-base-details_2827.pkl
# for 'ifeval', 'bbh', 'gpqa', 'musr', 'math_lv_5', 'mmlu_pro'
# with open(f'./data/ooo_dataset/full_data/mmlu_pro/Qwen__Qwen2.5-32B-details_2769.pkl', 'rb') as f:
#     deepseek_ai = pickle.load(f)
# keys = list(deepseek_ai.keys()) # convert to list
# print(deepseek_ai.keys())
# print(deepseek_ai[keys[0]].keys())
# subkeys = list(deepseek_ai[keys[0]].keys())
# print(deepseek_ai[keys[0]][subkeys[-1]])
# dict_keys(['doc_id', 'doc', 'target', 'arguments', 'resps', 'filtered_resps', 'doc_hash', 'prompt_hash', 'target_hash', 'acc_norm'])
# use last key as acc
#

def get_model_result(model_file, dataset_folder, subdataset, idx):
    with open(os.path.join(dataset_folder, model_file), 'rb') as f:
         model_result = pickle.load(f)
    acc_key = list(model_result[subdataset])[-1]
    # model_result[subdataset][acc_key][idx] can be 0 or 1, or True or False. If get True or False, convert to 1 and 0.
    try:
        model_result[subdataset][acc_key] = np.array(model_result[subdataset][acc_key])
    except:
        print(model_result[subdataset][acc_key])
    model_result[subdataset][acc_key][model_result[subdataset][acc_key] == True] = 1
    model_result[subdataset][acc_key][model_result[subdataset][acc_key] == False] = 0
    # if contains none, also set to 0
    model_result[subdataset][acc_key][model_result[subdataset][acc_key] == 'none'] = 0
    return model_result[subdataset][acc_key][idx]


ooo_dataset = {}
ooo_dataset['available_models'] = model_keywords
ooo_dataset['available_datasets'] = data_options
ooo_dataset['full_data'] = []
task_types = []
for dataset_idx in data_options:
    data_folder = f'./data/ooo_dataset/full_data/{dataset_idx}/'
    models_list = os.listdir(data_folder)
    first_model = os.path.join(data_folder, models_list[0])
    with open(first_model, 'rb') as f:
         first_model_result = pickle.load(f)
    subdataset_keys = list(first_model_result.keys())
    # add task types
    task_types.extend(subdataset_keys)
    for subdata_idx in subdataset_keys:
        num_samples = len(first_model_result[subdata_idx]['doc_id'])
        print(f'subdataset {subdata_idx} contain {num_samples} samples')
        for sample_idx in range(num_samples):
            query = {}
            query['category'] = dataset_idx + '-' + subdata_idx
            query['idx'] = sample_idx
            model_results = []
            for model_idx in model_keywords:
                # search the model_filename from model_list (if model_keywords contains in the list)
                model_file = next(
                    filename for filename in models_list if model_idx in filename)
                model_results.append(get_model_result(model_file, data_folder, subdataset=subdata_idx, idx=sample_idx))
            query['results'] = model_results
            ooo_dataset['full_data'].append(query)


ooo_dataset['task_types'] = task_types
ooo_dataset['total_samples'] = len(ooo_dataset['full_data'])
# pkl save ooo_dataset
with open(f'data/ooo_dataset/ooo_dataset1.pkl', 'wb') as f:
    pickle.dump(ooo_dataset, f)



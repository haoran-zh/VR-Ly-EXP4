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

data_options = ['ifeval', 'bbh', 'gpqa', 'musr', 'math_lv_5', 'mmlu_pro']
model_keywords = ['openai-community__gpt2-details',
                'openai-community__gpt2-large-details',
                'openai-community__gpt2-xl-details',
                    'deepseek-ai__deepseek-llm-7b',
                  'deepseek-ai__deepseek-moe-16b',
                  'deepseek-ai__deepseek-llm-67b',
                  'mistral-community__Mistral-7B-v0.2-details',
                  'mistral-community__Mixtral-8x22B-v0.1-details',
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
# reorganize data
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/Qwen__Qwen2.5-32B-Instruct-details_1488.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/openai-community__gpt2-xl-details_754.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/deepseek-ai__deepseek-llm-7b-base-details_2827.pkl
# /home/hz8556/llmooo/data/ooo_dataset/full_data/bbh/mistral-community__Mixtral-8x22B-v0.1-details_3466.pkl (evaluate not all)
with open(f'./data/ooo_dataset/full_data/math_lv_5/Qwen__Qwen2.5-32B-Instruct-details_1488.pkl', 'rb') as f:
    deepseek_ai = pickle.load(f)
print(deepseek_ai['math_algebra_hard']['exact_match'])



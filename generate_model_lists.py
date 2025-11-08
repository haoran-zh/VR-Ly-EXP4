import pickle
import numpy as np
from utils import convert_arrays_to_shapes

# choose to_handle_datasets from below
# ['arc', 'hellaswag', 'mmlu', 'winogrande', 'gsm8k']
# ['ifeval', 'bbh', 'gpqa', 'musr', 'math', 'mmlu_pro']
to_handle_dataset = 'gsm8k'

# load the pre-built router dataset
# with open(f'data/router_dataset/{to_handle_dataset}_router_dataset.pkl', 'rb') as f:
#     router_dataset = pickle.load(f)
#
# # display the pre-built router dataset
# # print(convert_arrays_to_shapes(router_dataset))
#
#
#
# print(f"Keys: {router_dataset.keys()}")

# print(f"data: {router_dataset['easy'][5]['strong_to_weak']['data']}")
# print(f"model: {router_dataset['easy'][5]['strong_to_weak']['model']}")

# print(f"model: {router_dataset['hard'][1000]['strong_to_weak']['model']}")
# print(f"model: {router_dataset['split_index']}")  # how they split the dataset into train, val, test
# print(f"model: {router_dataset['embedding']['train_embed']}")
# print(f"model: {router_dataset['prompt']['train_prompt']}")

from huggingface_hub import list_repo_files

# full data folders
data_options = ['arc_challenge', 'hellaswag', 'mmlu', 'winogrande',
                'gsm8k', 'truthfulqa', 'ifeval', 'bbh', 'gpqa',
                'musr', 'math_lv_5', 'mmlu_pro']


# generate model list per dataset
model_list_per_dataset = {}

for dataset_idx in data_options:
    repo_id = "linggm/RouterEval"
    folder_prefix = f"full_data/{dataset_idx}/"

    # Get a list of all file paths in the dataset repository
    all_file_paths = list_repo_files(repo_id, repo_type="dataset")

    # Filter the list to only include files that start with the folder path
    file_names = []
    for path in all_file_paths:
        if path.startswith(folder_prefix) and path != folder_prefix:
            # Extract just the file name by removing the folder_prefix
            file_names.append(path.replace(folder_prefix, ""))

    print(f"Dataset {dataset_idx}:")
    print(f'total count of models: {len(file_names)}')
    model_list_per_dataset[dataset_idx] = file_names

# pickle save model_lists_per_dataset
with open(f'data/model_list_per_dataset.pkl', 'wb') as f:
    pickle.dump(model_list_per_dataset, f)
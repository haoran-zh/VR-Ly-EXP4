#!/usr/bin/env python3
"""
Generate ooo_dataset using 19 selected popular models (canonical IDs).

Update (per request):
- If a selected model does not exist for a dataset (or fails to load), its per-sample result is
  set to 0 (i.e., always "error") instead of None.

This version uses the SAME 11 datasets as your original organize_data() and preserves the original
metrics-vs-doc_id logic.

Usage:
  python generate_ooodata_pop19_all11_missing0.py --root ./data --local_dir ./data/ooo_dataset --do_download --do_organize
"""

import os
import re
import pickle
import zipfile
import argparse
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "linggm/RouterEval"

SELECTED_CANON_MODELS = [
    "llama2_7b_cpo_details_773",
    "13_outof_32_pruned_layers_llama3_1_8b_details_3682",
    "llama_13b_details_3522",
    "autotrain_llama3_70b_orpo_v1_details_1219",
    "qwen1_5_1_8b_chat_details_1598",
    "calme_2_2_qwen2_7b_details_3186",
    "deepseek_r1_distill_qwen_14b_abliterated_v2_details_636",
    "deepseek_r1_distill_qwen_32b_details_1975",
    "calme_2_1_qwen2_5_72b_details_3442",
    "collectivecognition_v1_1_mistral_7b_details_1811",
    "merge_mixtral_prometheus_8x7b_details_802",
    "dolphin_2_9_3_mistral_nemo_12b_details_786",
    "3prymmal_phi3_3b_slerp_details_877",
    "medphi_4_14b_v1_details_1111",
    "athena_gemma_2_2b_it_details_1862",
    "athene_codegemma_2_7b_it_alpaca_v1_2_details_524",
    "4prymmal_gemma2_9b_slerp_details_2671",
    "bggpt_gemma_2_27b_it_v1_0_details_1153",
    "deepseek_llm_67b_chat_details_1361",
]

DATASETS = [
    "ifeval", "bbh", "gpqa", "musr", "math_lv_5", "mmlu_pro"
]

METRICS_STYLE = {"arc_challenge", "hellaswag", "mmlu", "winogrande", "gsm8k"}
DOCID_STYLE   = set(DATASETS) - METRICS_STYLE


def canonicalize(name: str) -> str:
    s = os.path.basename(name.strip())
    if s in {"", ".gitkeep"}:
        return ""
    s = re.sub(r"\.(zip|tar\.gz|tgz|gguf|bin|pt|pth|safetensors|pkl)$", "", s, flags=re.I)
    s = re.sub(r"^(details_|detail_|model_|models_)", "", s, flags=re.I)
    if "__" in s:
        s = s.split("__", 1)[1]
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_canon_to_filename(file_list):
    canon_map = {}
    collisions = {}
    for fn in file_list:
        c = canonicalize(fn)
        if not c:
            continue
        if c in canon_map and canon_map[c] != fn:
            collisions.setdefault(c, set()).update([canon_map[c], fn])
            continue
        canon_map[c] = fn
    if collisions:
        print("⚠️ Canonical collisions (same canon -> multiple files). Using first seen:")
        shown = 0
        for c, fns in collisions.items():
            print(" ", c, "=>", list(fns)[:3], ("..." if len(fns) > 3 else ""))
            shown += 1
            if shown >= 10:
                break
    return canon_map


def unzip_file(zip_path, extract_to_dir):
    print(f"  Unzipping {os.path.basename(zip_path)} to {extract_to_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)
    os.remove(zip_path)


def download_data(root: str, local_dir: str):
    model_list_path = os.path.join(root, "model_list_per_dataset.pkl")
    if not os.path.exists(model_list_path):
        raise FileNotFoundError(f"model_list_per_dataset.pkl not found at: {model_list_path}")

    with open(model_list_path, "rb") as f:
        model_list_per_dataset = pickle.load(f)

    for dataset in DATASETS:
        print(f"\n=== Downloading dataset: {dataset} ===")
        file_list = model_list_per_dataset[dataset]
        canon_map = build_canon_to_filename(file_list)

        for canon_model in SELECTED_CANON_MODELS:
            target_file = canon_map.get(canon_model)
            if not target_file:
                continue

            filename = f"full_data/{dataset}/{target_file}"
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
            )
            unzip_file(local_path, os.path.dirname(local_path))

    print("\n✅ Download done.")


def _clean_to_int_array(arr: np.ndarray) -> np.ndarray:
    """Convert bool/'none'/ints to {0,1} int array."""
    arr = np.array(arr, copy=False)
    arr[arr == True] = 1
    arr[arr == False] = 0
    arr[arr == "none"] = 0
    return arr.astype(int)


def organize_data(local_dir: str, out_pkl: str):
    ooo_dataset = {
        "available_models": SELECTED_CANON_MODELS,
        "available_datasets": DATASETS,
        "full_data": [],
    }
    task_types = []

    for dataset_idx in DATASETS:
        print(f"\nProcessing dataset: {dataset_idx}")
        data_folder = os.path.join(local_dir, "full_data", dataset_idx)
        if not os.path.isdir(data_folder):
            print(f"  ❌ Missing folder: {data_folder}. Did you run --do_download? Skipping.")
            continue

        models_list = os.listdir(data_folder)
        canon_map = build_canon_to_filename(models_list)

        # Load model files that exist; missing models will be handled as all-zeros later.
        loaded_models_data = {}
        model_file_map = {}  # canon_model -> actual filename

        print(f"  Loading model files for {dataset_idx} into memory...")
        for canon_model in SELECTED_CANON_MODELS:
            model_file = canon_map.get(canon_model)
            if not model_file:
                continue
            model_file_map[canon_model] = model_file
            try:
                with open(os.path.join(data_folder, model_file), "rb") as f:
                    loaded_models_data[model_file] = pickle.load(f)
            except Exception as e:
                print(f"    ⚠️ Failed to load {model_file} ({canon_model}): {e}")
                model_file_map.pop(canon_model, None)
        print(f"  ✅ Loaded {len(model_file_map)}/{len(SELECTED_CANON_MODELS)} models for {dataset_idx}.")

        if not loaded_models_data:
            print(f"  ❌ No models loaded for {dataset_idx}, skipping.")
            continue

        first_model_file = next(iter(loaded_models_data.keys()))
        first_model_result = loaded_models_data[first_model_file]
        subdataset_keys = list(first_model_result.keys())
        task_types.extend(subdataset_keys)

        for subdata_idx in subdataset_keys:
            if dataset_idx in DOCID_STYLE:
                num_samples = len(first_model_result[subdata_idx]["doc_id"])
            else:
                num_samples = len(first_model_result[subdata_idx]["metrics"])

            print(f"  Processing subdataset {subdata_idx} ({num_samples} samples)")

            # Precompute arrays for models that exist; missing ones will be zeros.
            processed_results = {}

            for canon_model, model_file in model_file_map.items():
                model_result = loaded_models_data[model_file]

                if dataset_idx in DOCID_STYLE:
                    if subdata_idx == "ifeval":
                        acc_key = "prompt_level_loose_acc"
                    else:
                        acc_key = list(model_result[subdata_idx])[-1]
                    arr = np.array(model_result[subdata_idx][acc_key])
                    processed_results[canon_model] = _clean_to_int_array(arr)
                else:
                    acc_key = "metrics"
                    # Keep your special-case, but robust:
                    if "gpt2-xl" in model_file and ("alpaca" in model_file or "Rachneet" in model_file):
                        arr = np.array(model_result[subdata_idx].get("acc", [0] * num_samples))
                        processed_results[canon_model] = _clean_to_int_array(arr)
                    else:
                        arr = np.array([model_result[subdata_idx][acc_key][i]["acc"] for i in range(num_samples)])
                        processed_results[canon_model] = _clean_to_int_array(arr)

            # Per-sample loop
            for sample_idx in range(num_samples):
                q = {"category": subdata_idx, "idx": sample_idx}

                if dataset_idx in DOCID_STYLE:
                    if dataset_idx == "ifeval":
                        q["length"] = len(first_model_result[subdata_idx]["doc"][sample_idx]["prompt"])
                    elif dataset_idx == "bbh":
                        q["length"] = len(first_model_result[subdata_idx]["doc"][sample_idx]["input"])
                    elif dataset_idx == "gpqa":
                        q["length"] = len(first_model_result[subdata_idx]["doc"][sample_idx]["Pre-Revision Question"])
                    elif dataset_idx == "musr":
                        q["length"] = (
                            len(first_model_result[subdata_idx]["doc"][sample_idx]["narrative"])
                            + len(first_model_result[subdata_idx]["doc"][sample_idx]["question"])
                        )
                    elif dataset_idx == "math_lv_5":
                        q["length"] = len(first_model_result[subdata_idx]["doc"][sample_idx]["problem"])
                    elif dataset_idx == "mmlu_pro":
                        q["length"] = len(first_model_result[subdata_idx]["doc"][sample_idx]["question"])
                    else:
                        q["length"] = 0
                else:
                    q["length"] = len(first_model_result[subdata_idx]["full_prompt"][sample_idx])

                # IMPORTANT CHANGE: missing model => 0
                res = []
                for canon_model in SELECTED_CANON_MODELS:
                    arr = processed_results.get(canon_model)
                    res.append(int(arr[sample_idx]) if arr is not None else 0)
                q["results"] = res

                ooo_dataset["full_data"].append(q)

            del processed_results

        del loaded_models_data
        print(f"  Finished dataset {dataset_idx}, cleared from memory.")

    ooo_dataset["task_types"] = task_types
    ooo_dataset["total_samples"] = len(ooo_dataset["full_data"])

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(ooo_dataset, f)

    print(f"\n✅ All done. Saved: {out_pkl}")
    print(f"total_samples = {ooo_dataset['total_samples']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data", help="Root dir containing model_list_per_dataset.pkl")
    ap.add_argument("--local_dir", type=str, default="./data/ooo_dataset", help="Download/unzip dir (contains full_data/)")
    ap.add_argument("--out_pkl", type=str, default="./data/ooo_dataset/ooo_dataset_pop19.pkl", help="Output PKL")
    ap.add_argument("--do_download", action="store_true")
    ap.add_argument("--do_organize", action="store_true")
    args = ap.parse_args()

    if not args.do_download and not args.do_organize:
        ap.error("Nothing to do. Add --do_download and/or --do_organize")

    if args.do_download:
        download_data(root=args.root, local_dir=args.local_dir)
    if args.do_organize:
        organize_data(local_dir=args.local_dir, out_pkl=args.out_pkl)


if __name__ == "__main__":
    main()

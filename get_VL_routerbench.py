#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import tarfile
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


# -----------------------------
# Download / extract
# -----------------------------
def download_and_extract_vl_routerbench(
    out_dir: str = "./vl_routerbench_data",
    repo_id: str = "KinghtH/VL-RouterBench",
    filename: str = "vlm_router_data.tar.gz",
) -> str:
    """
    Downloads vlm_router_data.tar.gz from HF and extracts it under out_dir.
    Returns the extracted root directory path (…/vlm_router_data).
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tar_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=out_dir,
    )
    print(f"[download] tar.gz saved to: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)

    extracted_root = os.path.join(out_dir, "vlm_router_data")
    if not os.path.isdir(extracted_root):
        raise FileNotFoundError(
            f"Expected extracted folder {extracted_root} not found. "
            f"Check extracted contents under {out_dir}."
        )

    print(f"[extract] extracted to: {extracted_root}")
    return extracted_root


# -----------------------------
# Helpers: infer columns
# -----------------------------
_UID_CANDIDATES = [
    "uid", "id", "index", "idx", "question_id", "doc_id", "sample_id", "qid", "pid"
]

_TEXT_CANDIDATES = [
    "prompt", "question", "query", "input", "instruction", "text", "problem",
    "stem", "caption"
]

_ACC_CANDIDATES = [
    "acc", "accuracy", "correct", "is_correct", "score", "match", "hit", "judge",
    "auxmatch", "result"
]


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", "", str(c).strip().lower())


def infer_uid_column(df: pd.DataFrame) -> Optional[str]:
    cols = { _norm_col(c): c for c in df.columns }
    for k in _UID_CANDIDATES:
        nk = _norm_col(k)
        if nk in cols:
            return cols[nk]
    return None


def infer_text_column(df: pd.DataFrame) -> Optional[str]:
    cols = { _norm_col(c): c for c in df.columns }
    for k in _TEXT_CANDIDATES:
        nk = _norm_col(k)
        if nk in cols:
            return cols[nk]
    # fallback: first object column with non-trivial average length
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    best = None
    best_len = -1
    for c in obj_cols:
        s = df[c].astype(str).fillna("")
        avg = float(s.str.len().mean())
        if avg > best_len:
            best_len = avg
            best = c
    if best_len >= 5:
        return best
    return None


def _is_boolish_series(s: pd.Series) -> bool:
    # Accept {0,1}, {True,False}, {"true","false"}, etc.
    if s.dtype == bool:
        return True
    v = s.dropna().unique()
    if len(v) == 0:
        return False
    # numeric boolish
    if np.issubdtype(s.dtype, np.number):
        return set(map(int, v)).issubset({0, 1})
    # string boolish
    vs = set(str(x).strip().lower() for x in v)
    return vs.issubset({"0", "1", "true", "false", "yes", "no"})


def infer_acc_column(df: pd.DataFrame) -> Optional[str]:
    cols = { _norm_col(c): c for c in df.columns }
    # 1) direct name match
    for k in _ACC_CANDIDATES:
        nk = _norm_col(k)
        if nk in cols:
            return cols[nk]

    # 2) best boolish column
    boolish = []
    for c in df.columns:
        try:
            if _is_boolish_series(df[c]):
                boolish.append(c)
        except Exception:
            pass
    if len(boolish) == 1:
        return boolish[0]
    if len(boolish) > 1:
        # prefer columns with fewer missing values
        boolish.sort(key=lambda c: df[c].isna().mean())
        return boolish[0]

    return None


def to_int_correct(s: pd.Series) -> np.ndarray:
    """
    Convert a correctness column to int {0,1}. Unknown -> 0.
    """
    if s.dtype == bool:
        return s.fillna(False).astype(int).to_numpy()

    if np.issubdtype(s.dtype, np.number):
        # assume 0/1 or floats; clamp
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy()
        x = (x > 0.5).astype(int)
        return x

    # strings
    ss = s.astype(str).fillna("").str.strip().str.lower()
    m = ss.isin(["1", "true", "yes"])
    return m.astype(int).to_numpy()


# -----------------------------
# Task name parsing
# -----------------------------
def infer_task_name(model_name: str, xlsx_path: Path) -> str:
    """
    Example: deepseek_vl2_AI2D_TEST_openai_result.xlsx -> AI2D_TEST
    """
    name = xlsx_path.stem  # without .xlsx
    # strip model prefix if present
    if name.startswith(model_name + "_"):
        name = name[len(model_name) + 1 :]

    # strip common suffixes
    name = re.sub(r"(_openai_result|_openai_results)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(_result|_results)$", "", name, flags=re.IGNORECASE)

    return name


# -----------------------------
# Read one model's evaluation folder
# -----------------------------
def read_one_model_eval(model_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Return: task -> dataframe
    """
    out = {}
    for p in sorted(model_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(p)
            task = infer_task_name(model_dir.name, p)
            out[task] = df
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    return out


# -----------------------------
# Align across models and build ooo-style dataset
# -----------------------------
def build_vl_ooo_dataset(
    eval_root: Path,
    prefer_uid: bool = True,
    strict_task_intersection: bool = True,
) -> Dict:
    """
    Scans eval_root = .../VLMEvalKit_evaluation
    Builds:
      {
        'available_models': [...],
        'available_tasks': [...],
        'full_data': [
            {
              'category': <task>,
              'idx': <row index within aligned table>,
              'uid': <uid or None>,
              'length': <int>,
              'results': [0/1/None per model order]
            }, ...
        ],
        'task_types': [...],   # same as tasks
        'total_samples': int,
        'meta': {...}
      }
    """
    eval_root = Path(eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root not found: {eval_root}")

    model_dirs = [p for p in eval_root.iterdir() if p.is_dir()]
    model_dirs.sort(key=lambda p: p.name)

    available_models = [p.name for p in model_dirs]
    print(f"[info] found {len(available_models)} models under {eval_root}")

    # load all model dataframes
    model_task_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for md in model_dirs:
        model_task_dfs[md.name] = read_one_model_eval(md)

    # tasks
    task_sets = [set(d.keys()) for d in model_task_dfs.values()]
    if len(task_sets) == 0:
        raise RuntimeError("No model folders / no xlsx found.")

    if strict_task_intersection:
        tasks = sorted(set.intersection(*task_sets))
    else:
        tasks = sorted(set.union(*task_sets))

    print(f"[info] tasks = {len(tasks)} (strict_intersection={strict_task_intersection})")

    ooo = {
        "available_models": available_models,
        "available_tasks": tasks,
        "full_data": [],
        "task_types": tasks,
        "meta": {
            "prefer_uid": prefer_uid,
            "strict_task_intersection": strict_task_intersection,
            "eval_root": str(eval_root),
        },
    }

    # build per-task aligned tables
    for task in tasks:
        # choose a reference df: first model that has this task
        ref_model = next((m for m in available_models if task in model_task_dfs[m]), None)
        if ref_model is None:
            continue
        ref_df = model_task_dfs[ref_model][task].copy()

        uid_col = infer_uid_column(ref_df) if prefer_uid else None
        text_col = infer_text_column(ref_df)
        acc_col_ref = infer_acc_column(ref_df)

        if acc_col_ref is None:
            print(f"[warn] task={task}: cannot infer acc column in reference model={ref_model}. Skipping task.")
            continue

        # Build reference ordering
        if uid_col is not None:
            ref_uids = ref_df[uid_col].tolist()
        else:
            ref_uids = [None] * len(ref_df)

        # length vector from reference
        if text_col is not None:
            ref_lengths = ref_df[text_col].astype(str).fillna("").str.len().astype(int).to_numpy()
        else:
            ref_lengths = np.zeros(len(ref_df), dtype=int)

        # Prepare result arrays per model aligned to reference
        per_model_results: List[Optional[np.ndarray]] = []
        for m in available_models:
            dfm = model_task_dfs[m].get(task, None)
            if dfm is None:
                per_model_results.append(None)
                continue

            acc_col = infer_acc_column(dfm)
            if acc_col is None:
                print(f"[warn] task={task}: cannot infer acc column for model={m}; will fill None")
                per_model_results.append(None)
                continue

            corr = to_int_correct(dfm[acc_col])

            if uid_col is not None:
                # align by uid if possible (use uid col name from THIS df if exists; else fallback row order)
                uid_col_m = infer_uid_column(dfm)
                if uid_col_m is None:
                    # fallback: row order
                    aligned = corr
                else:
                    uid_to_idx = {uid: i for i, uid in enumerate(dfm[uid_col_m].tolist())}
                    aligned = np.array([corr[uid_to_idx.get(u, -1)] if u in uid_to_idx else -1 for u in ref_uids])
                    # map -1 to None later
                per_model_results.append(aligned)
            else:
                # row-order align
                if len(corr) != len(ref_df):
                    # pad/trim to match
                    L = len(ref_df)
                    tmp = np.full(L, -1, dtype=int)
                    tmp[: min(L, len(corr))] = corr[: min(L, len(corr))]
                    corr = tmp
                per_model_results.append(corr)

        # Emit samples
        num_samples = len(ref_df)
        for i in range(num_samples):
            results_i = []
            for arr in per_model_results:
                if arr is None:
                    results_i.append(None)
                else:
                    v = int(arr[i])
                    if v < 0:
                        results_i.append(None)
                    else:
                        results_i.append(v)

            ooo["full_data"].append(
                {
                    "category": task,
                    "idx": i,
                    "uid": ref_uids[i],
                    "length": int(ref_lengths[i]),
                    "results": results_i,
                }
            )

        print(f"[ok] task={task}: samples={num_samples}")

    ooo["total_samples"] = len(ooo["full_data"])
    return ooo


# -----------------------------
# Saving
# -----------------------------
def save_pickle(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save] pickle -> {path}")


def save_jsonl(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in obj["full_data"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[save] jsonl -> {path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./vl_routerbench_data",
                    help="Where to download/extract vlm_router_data.tar.gz")
    ap.add_argument("--repo_id", type=str, default="KinghtH/VL-RouterBench")
    ap.add_argument("--filename", type=str, default="vlm_router_data.tar.gz")

    ap.add_argument("--eval_subdir", type=str, default="VLMEvalKit_evaluation",
                    help="Subdir under extracted root that contains per-model xlsx folders")
    ap.add_argument("--save_dir", type=str, default="./vl_routerbench_data/processed",
                    help="Where to write processed dataset")
    ap.add_argument("--name", type=str, default="vl_ooo_dataset",
                    help="Output base name (without extension)")

    ap.add_argument("--no_download", action="store_true",
                    help="Skip download/extract; assume out_dir already has vlm_router_data/")
    ap.add_argument("--no_uid_align", action="store_true",
                    help="Disable UID-based alignment (always use row order)")
    ap.add_argument("--task_union", action="store_true",
                    help="Use union of tasks across models (default: intersection)")

    ap.add_argument("--write_jsonl", action="store_true",
                    help="Also write a JSONL (one sample per line)")
    args = ap.parse_args()

    if args.no_download:
        extracted_root = os.path.join(os.path.abspath(args.out_dir), "vlm_router_data")
        if not os.path.isdir(extracted_root):
            raise FileNotFoundError(f"--no_download set but not found: {extracted_root}")
    else:
        extracted_root = download_and_extract_vl_routerbench(
            out_dir=args.out_dir,
            repo_id=args.repo_id,
            filename=args.filename,
        )

    eval_root = Path(extracted_root) / args.eval_subdir
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root not found: {eval_root}")

    ooo = build_vl_ooo_dataset(
        eval_root=eval_root,
        prefer_uid=(not args.no_uid_align),
        strict_task_intersection=(not args.task_union),
    )

    save_dir = Path(args.save_dir)
    pkl_path = save_dir / f"{args.name}.pkl"
    save_pickle(ooo, pkl_path)

    if args.write_jsonl:
        jsonl_path = save_dir / f"{args.name}.jsonl"
        save_jsonl(ooo, jsonl_path)

    print(f"[done] total_samples={ooo['total_samples']} models={len(ooo['available_models'])} tasks={len(ooo['available_tasks'])}")


# if __name__ == "__main__":
#     main()
# read vl_routerbench_data/processed/vl_ooo_dataset.pkl
with open("vl_routerbench_data/processed/vl_ooo_dataset.pkl", "rb") as f:
    ooo = pickle.load(f)
# print(ooo.keys())  # dict_keys(['available_models', 'available_tasks', 'full_data', 'task_types', 'meta', 'total_samples'])
# print(ooo['full_data'][1]['results'])  # dict_keys(['category', 'idx', 'uid', 'length', 'results'])
print(ooo['available_models'])
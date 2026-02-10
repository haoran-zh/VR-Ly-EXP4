#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save] {path}")


def sample_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    Canonical key for matching the same sample across datasets.
    Uses UID if present, else falls back to idx.
    Return a tuple (category, key_type, key_value_as_int_or_hash).
    """
    cat = row.get("category")
    uid = row.get("uid", None)
    if uid is not None:
        # uid might be string/int; normalize to string to avoid 1 vs "1"
        return (cat, "uid", hash(str(uid)))
    return (cat, "idx", int(row.get("idx")))


def unique_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def merge_two(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    m1 = d1["available_models"]
    m2 = d2["available_models"]
    merged_models = unique_preserve_order(m1 + m2)

    # mapping from model name -> index in each dataset
    idx1 = {m: i for i, m in enumerate(m1)}
    idx2 = {m: i for i, m in enumerate(m2)}
    midx = {m: i for i, m in enumerate(merged_models)}

    tasks = sorted(set(d1.get("available_tasks", d1.get("task_types", []))) |
                   set(d2.get("available_tasks", d2.get("task_types", []))))

    # Index both datasets by sample key
    map1 = {}
    for row in d1["full_data"]:
        map1[sample_key(row)] = row

    map2 = {}
    for row in d2["full_data"]:
        map2[sample_key(row)] = row

    all_keys = list(map1.keys() | map2.keys())

    merged_full = []
    for k in all_keys:
        r1 = map1.get(k, None)
        r2 = map2.get(k, None)

        # choose a "base" row to carry metadata (category/idx/uid/length)
        base = r1 if r1 is not None else r2

        new_row = {
            "category": base.get("category"),
            "idx": base.get("idx"),
            "uid": base.get("uid", None),
            "length": base.get("length", 0),
            "results": [None] * len(merged_models),
        }

        if r1 is not None:
            res1 = r1["results"]
            for model_name, j in idx1.items():
                new_row["results"][midx[model_name]] = res1[j]

        if r2 is not None:
            res2 = r2["results"]
            for model_name, j in idx2.items():
                new_row["results"][midx[model_name]] = res2[j]

        merged_full.append(new_row)

    # Optional: sort for stability (by category then idx then uid-hash key)
    merged_full.sort(key=lambda r: (str(r["category"]), int(r.get("idx", 0)), str(r.get("uid", ""))))

    merged = {
        "available_models": merged_models,
        "available_tasks": tasks,
        "task_types": tasks,
        "full_data": merged_full,
        "total_samples": len(merged_full),
        "meta": {
            "merged_from": ["dataset1", "dataset2"],
            "models_1": m1,
            "models_2": m2,
        },
    }
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl1", type=str, required=True)
    ap.add_argument("--pkl2", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    d1 = load_pkl(args.pkl1)
    d2 = load_pkl(args.pkl2)
    merged = merge_two(d1, d2)
    save_pkl(merged, args.out)

    print("[done] models:", len(merged["available_models"]),
          "tasks:", len(merged["available_tasks"]),
          "samples:", merged["total_samples"])


if __name__ == "__main__":
    main()

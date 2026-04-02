import os
import time
import gc
from statistics import mean
from typing import Dict, List, Optional

import pandas as pd
import psutil
from radon.metrics import mi_visit
from mmdet.apis import init_detector, inference_detector


# =========================================================
# USER SETTINGS
# =========================================================

# 1) IMAGE TEST
TEST_IMAGE = r"C:\Users\gelob\Desktop\final eggplant dataset\test\51933218-3dfb-4628-b16c-68151b2cace6_jpeg.rf.2fb2c08f6dfe09a02cb6034558598a51.jpg"

# 2) 3 MODELS
MODELS = [
    {
        "name": "Mask R-CNN",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\mask_rcnn\mask_rcnn_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\mask_rcnn_eggplant\epoch_30.pth",
        "bbox_map": 0.732,
        "code_files": [
            r"C:\Users\gelob\Desktop\mmdetection\configs\mask_rcnn\mask_rcnn_eggplant.py",
        ],
    },
    {
        "name": "CondInst",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\condinst\condinst_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\condinst_eggplant\iter_7400.pth",
        "bbox_map": 0.725,
        "code_files": [
            r"C:\Users\gelob\Desktop\mmdetection\configs\condinst\condinst_eggplant.py",
        ],
    },
    {
        "name": "YOLACT",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\yolact\yolact_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\yolact_eggplant\epoch_30.pth",
        "bbox_map": 0.670,
        "code_files": [
            r"C:\Users\gelob\Desktop\mmdetection\configs\yolact\yolact_eggplant.py",
        ],
    },
]

# 3) Inference settings
DEVICE = "cuda:0"   # use "cpu" if needed
WARMUP_RUNS = 5
MEASURE_RUNS = 10

# 4) Output file
OUTPUT_CSV = r"C:\Users\gelob\Desktop\design_constraints_results.csv"

def file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def compute_error_rate(bbox_map: float) -> float:
    return 1.0 - bbox_map


def compute_maintainability_index(code_files: List[str]) -> Optional[float]:
    scores = []

    for file_path in code_files:
        if not os.path.isfile(file_path):
            print(f"[WARN] Maintainability file not found: {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            score = mi_visit(code, multi=True)
            scores.append(score)
        except Exception as e:
            print(f"[WARN] Failed MI for {file_path}: {e}")

    if not scores:
        return None

    return mean(scores)


def measure_inference_and_ram(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    device: str = "cuda:0",
    warmup_runs: int = 5,
    measure_runs: int = 10,
) -> Dict[str, float]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")

    process = psutil.Process(os.getpid())

    print(f"\n[INFO] Loading model:")
    print(f"       Config     : {config_path}")
    print(f"       Checkpoint : {checkpoint_path}")

    # Baseline RAM before model load
    gc.collect()
    baseline_ram = process.memory_info().rss / (1024 * 1024)

    # Load model
    model = init_detector(config_path, checkpoint_path, device=device)

    # RAM after model load
    ram_after_load = process.memory_info().rss / (1024 * 1024)

    # Warmup
    for _ in range(warmup_runs):
        _ = inference_detector(model, image_path)

    # Measure inference time + peak RAM while model is loaded
    times_ms = []
    peak_ram = ram_after_load

    for _ in range(measure_runs):
        start = time.perf_counter()
        _ = inference_detector(model, image_path)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        times_ms.append(elapsed_ms)

        current_ram = process.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram:
            peak_ram = current_ram

    total_ram_used = max(0.0, peak_ram - baseline_ram)

    # Clean
    del model
    gc.collect()

    return {
        "avg_inference_ms": mean(times_ms),
        "total_ram_used_mb": total_ram_used,
        "baseline_ram_mb": baseline_ram,
        "ram_after_load_mb": ram_after_load,
        "peak_ram_mb": peak_ram,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    results = []

    print("\n========== DESIGN CONSTRAINTS MEASUREMENT ==========")

    for model_info in MODELS:
        name = model_info["name"]
        config = model_info["config"]
        checkpoint = model_info["checkpoint"]
        bbox_map = model_info["bbox_map"]
        code_files = model_info.get("code_files", [])

        print(f"\n==================== {name} ====================")

        # 1) Model Size
        size_mb = file_size_mb(checkpoint)
        print(f"[OK] Model Size (MB): {size_mb:.3f}")

        # 2) Inference Speed + 3) RAM Consumption
        perf = measure_inference_and_ram(
            config_path=config,
            checkpoint_path=checkpoint,
            image_path=TEST_IMAGE,
            device=DEVICE,
            warmup_runs=WARMUP_RUNS,
            measure_runs=MEASURE_RUNS,
        )
        inference_ms = perf["avg_inference_ms"]
        ram_mb = perf["total_ram_used_mb"]

        print(f"[OK] Avg Inference Speed (ms): {inference_ms:.3f}")
        print(f"[OK] Total RAM Used (MB): {ram_mb:.3f}")
        print(f"[INFO] Baseline RAM (MB): {perf['baseline_ram_mb']:.3f}")
        print(f"[INFO] RAM After Load (MB): {perf['ram_after_load_mb']:.3f}")
        print(f"[INFO] Peak RAM (MB): {perf['peak_ram_mb']:.3f}")

        # 4) Maintainability Index
        mi_score = compute_maintainability_index(code_files)
        if mi_score is None:
            print("[WARN] Maintainability Index could not be computed.")
        else:
            print(f"[OK] Maintainability Index: {mi_score:.3f}")

        # 5) Error Rate
        error_rate = compute_error_rate(bbox_map)
        print(f"[OK] Error Rate: {error_rate:.3f}")

        results.append({
            "Model": name,
            "Model Size (MB)": round(size_mb, 3),
            "Inference Speed (ms)": round(inference_ms, 3),
            "RAM Consumption (MB)": round(ram_mb, 3),
            "Maintainability Index": round(mi_score, 3) if mi_score is not None else None,
            "bbox_mAP": round(bbox_map, 3),
            "Error Rate": round(error_rate, 3),
        })

    df = pd.DataFrame(results)

    print("\n========== FINAL RESULTS ==========")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
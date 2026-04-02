import os
import sys
import time
import json
import subprocess
from statistics import mean

import pandas as pd


# =========================================================
# USER SETTINGS
# =========================================================

TEST_IMAGE = r"C:\Users\gelob\Desktop\final eggplant dataset\test\51933218-3dfb-4628-b16c-68151b2cace6_jpeg.rf.2fb2c08f6dfe09a02cb6034558598a51.jpg"

MODELS = [
    {
        "name": "Mask R-CNN",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\mask_rcnn\mask_rcnn_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\mask_rcnn_eggplant\epoch_30.pth",
        "bbox_map": 0.732,
        "code_file": r"C:\Users\gelob\Desktop\mmdetection\configs\mask_rcnn\mask_rcnn_eggplant.py",
    },
    {
        "name": "CondInst",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\condinst\condinst_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\condinst_eggplant\iter_7400.pth",
        "bbox_map": 0.725,
        "code_file": r"C:\Users\gelob\Desktop\mmdetection\configs\condinst\condinst_eggplant.py",
    },
    {
        "name": "YOLACT",
        "config": r"C:\Users\gelob\Desktop\mmdetection\configs\yolact\yolact_eggplant.py",
        "checkpoint": r"C:\Users\gelob\Desktop\mmdetection\work_dirs\yolact_eggplant\epoch_30.pth",
        "bbox_map": 0.670,
        "code_file": r"C:\Users\gelob\Desktop\mmdetection\configs\yolact\yolact_eggplant.py",
    },
]

DEVICE = "cuda:0"
WARMUP_RUNS = 5
MEASURE_RUNS = 10

OUTPUT_CSV = r"C:\Users\gelob\Desktop\design_constraints_results.csv"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def compute_error_rate(bbox_map: float) -> float:
    return 1.0 - bbox_map


def compute_maintainability_index(code_path: str):
    try:
        from radon.metrics import mi_visit
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        return mi_visit(code, multi=True)
    except Exception as e:
        print(f"[WARN] Failed MI for {code_path}: {e}")
        return None


def run_subprocess_measurement(model_info):
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        model_info["config"],
        model_info["checkpoint"],
        TEST_IMAGE,
        DEVICE,
        str(WARMUP_RUNS),
        str(MEASURE_RUNS),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Measurement failed for {model_info['name']}")

    lines = result.stdout.strip().splitlines()
    json_line = lines[-1]
    return json.loads(json_line)


# =========================================================
# WORKER MODE
# =========================================================

def worker_mode():
    import gc
    import time
    import psutil
    import torch
    from mmdet.apis import init_detector, inference_detector

    config_path = sys.argv[2]
    checkpoint_path = sys.argv[3]
    image_path = sys.argv[4]
    device = sys.argv[5]
    warmup_runs = int(sys.argv[6])
    measure_runs = int(sys.argv[7])

    process = psutil.Process(os.getpid())

    gc.collect()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    baseline_ram = process.memory_info().rss / (1024 * 1024)

    model = init_detector(config_path, checkpoint_path, device=device)

    ram_after_load = process.memory_info().rss / (1024 * 1024)
    peak_ram = ram_after_load

    # warmup
    for _ in range(warmup_runs):
        _ = inference_detector(model, image_path)

    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times_ms = []

    for _ in range(measure_runs):
        start = time.perf_counter()
        _ = inference_detector(model, image_path)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

        current_ram = process.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram:
            peak_ram = current_ram

    total_ram_used = max(0.0, peak_ram - baseline_ram)

    gpu_mem_mb = 0.0
    if torch.cuda.is_available() and "cuda" in device:
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    output = {
        "avg_inference_ms": mean(times_ms),
        "ram_used_mb": total_ram_used,
        "gpu_mem_mb": gpu_mem_mb,
        "baseline_ram_mb": baseline_ram,
        "ram_after_load_mb": ram_after_load,
        "peak_ram_mb": peak_ram,
    }

    print(json.dumps(output))


# =========================================================
# MAIN
# =========================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_mode()
        return

    results = []

    print("\n========== DESIGN CONSTRAINTS MEASUREMENT ==========")

    for model_info in MODELS:
        print(f"\n==================== {model_info['name']} ====================")

        model_size = file_size_mb(model_info["checkpoint"])
        perf = run_subprocess_measurement(model_info)
        mi_score = compute_maintainability_index(model_info["code_file"])
        error_rate = compute_error_rate(model_info["bbox_map"])

        print(f"[OK] Model Size (MB): {model_size:.3f}")
        print(f"[OK] Avg Inference Speed (ms): {perf['avg_inference_ms']:.3f}")
        print(f"[OK] RAM Consumption (MB): {perf['ram_used_mb']:.3f}")
        print(f"[OK] GPU Memory (MB): {perf['gpu_mem_mb']:.3f}")
        print(f"[INFO] Baseline RAM (MB): {perf['baseline_ram_mb']:.3f}")
        print(f"[INFO] RAM After Load (MB): {perf['ram_after_load_mb']:.3f}")
        print(f"[INFO] Peak RAM (MB): {perf['peak_ram_mb']:.3f}")
        print(f"[OK] Maintainability Index: {mi_score:.3f}" if mi_score is not None else "[WARN] Maintainability Index unavailable")
        print(f"[OK] Error Rate: {error_rate:.3f}")

        results.append({
            "Model": model_info["name"],
            "Model Size (MB)": round(model_size, 3),
            "Inference Speed (ms)": round(perf["avg_inference_ms"], 3),
            "RAM Consumption (MB)": round(perf["ram_used_mb"], 3),
            "GPU Memory (MB)": round(perf["gpu_mem_mb"], 3),
            "Maintainability Index": round(mi_score, 3) if mi_score is not None else None,
            "bbox_mAP": round(model_info["bbox_map"], 3),
            "Error Rate": round(error_rate, 3),
        })

    df = pd.DataFrame(results)

    print("\n========== FINAL RESULTS ==========")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
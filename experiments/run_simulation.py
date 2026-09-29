"""
Phase 5: Flower FL Simulation tren PC.
Toan bo cau hinh doc tu configs/experiment.yaml.

Chay:
    python -m experiments.run_simulation --all
    python -m experiments.run_simulation --num_clients 3 --alpha 1.0 --strategy fedavg
"""

import argparse
import json
import time
from pathlib import Path

import yaml
import torch
import flwr as fl

from datasets.medmnist_code import get_bloodmnist_datasets
from datasets.partition import load_partition
from models.cnn import SimpleCNN, get_parameters
from client.client import make_client_fn
from server.server import get_strategy, EarlyStoppingCallback, GlobalLRScheduler


def get_gpu_info() -> dict:
    """Thu thap thong tin GPU truoc khi chay."""
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_name": "N/A (CPU only)",
            "gpu_total_memory_gb": 0.0,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_available":        True,
        "gpu_name":             props.name,
        "gpu_total_memory_gb":  round(props.total_memory / (1024**3), 2),
        "cuda_version":         torch.version.cuda,
        "cudnn_version":        torch.backends.cudnn.version(),
    }


def get_gpu_usage() -> dict:
    """Doc muc su dung GPU tai thoi diem hien tai va peak."""
    if not torch.cuda.is_available():
        return {"peak_gpu_memory_mb": 0.0, "current_gpu_memory_mb": 0.0}
    return {
        "peak_gpu_memory_mb":    round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "current_gpu_memory_mb": round(torch.cuda.memory_allocated(0)     / (1024**2), 1),
    }

# === Doc cau hinh tu experiment.yaml (1 nguon su that duy nhat) ===
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"

with open(_CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Lay cac gia tri tu CFG
DEFAULT_NUM_CLIENTS = CFG["num_clients_options"]        # [3, 5, 10]
DEFAULT_ALPHAS      = CFG["alpha_values"]               # [1.0, 0.3, 0.1]
DEFAULT_STRATEGIES  = CFG["fl_strategies"]              # list 11 strategies
DEFAULT_SEED        = CFG["seed"]                       # 42
PARTITION_DIR       = Path(CFG["partition_save_dir"])   # data/partitions
RESULTS_DIR         = Path(CFG["results_dir"])          # results/federated
LOCAL_EPOCHS        = CFG["local_epochs"]               # 5
NUM_ROUNDS          = CFG["fl_rounds"]                  # 50
LR                  = CFG["learning_rate"]              # 0.001

ES_PATIENCE         = CFG["early_stopping"]["patience"]    # 5
ES_MIN_DELTA        = CFG["early_stopping"]["min_delta"]   # 0.001

LR_FACTOR           = CFG["lr_scheduler"]["factor"]        # 0.5
LR_PATIENCE         = CFG["lr_scheduler"]["patience"]      # 3
LR_MIN              = CFG["lr_scheduler"]["min_lr"]        # 0.000001

CLIENT_RESOURCES    = CFG["client_resources"]              # {num_cpus: 2, num_gpus: 0.5}


class EarlyStopException(Exception):
    """Exception de ep Flower dung vong lap."""
    pass

def calc_model_size_bytes(model) -> int:
    return sum(arr.nbytes for arr in get_parameters(model))


def run_one(num_clients: int, alpha: float,
            strategy_name: str, device: torch.device) -> dict:

    print(f"\n{'='*65}")
    print(f"  FL | clients={num_clients} | alpha={alpha}"
          f" | strategy={strategy_name.upper()}")
    print(f"{'='*65}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # 1. Dataset
    train_ds, val_ds, test_ds, num_classes = get_bloodmnist_datasets()

    # 2. Partition (da chia o Phase 4)
    json_file = (PARTITION_DIR /
                 f"partition_seed{DEFAULT_SEED}_alpha{alpha}"
                 f"_clients{num_clients}.json")
    if not json_file.exists():
        print(f"  [SKIP] Khong tim thay {json_file}")
        return None

    partition = load_partition(str(json_file))

    # 3. Model goc
    init_model  = SimpleCNN(in_channels=3, num_classes=num_classes).to(device)
    model_bytes = calc_model_size_bytes(init_model)
    model_kb    = model_bytes / 1024
    init_params = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    print(f"  Model size: {model_kb:.1f} KB")

    # 4. Strategy
    strategy = get_strategy(strategy_name, num_clients, init_params)

    # 5. Client factory
    client_fn = make_client_fn(
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        partition=partition, local_epochs=LOCAL_EPOCHS, lr=LR, device=device,
    )

    # 6. Callbacks (dung CFG thay vi hardcode)
    early_stopper = EarlyStoppingCallback(
        patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
    lr_scheduler  = GlobalLRScheduler(
        initial_lr=LR, factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN)

    history_acc  = []
    round_times  = []
    round_start  = [time.time()]
    actual_rounds = [0]

    # 7. Monkey-patch callbacks
    orig_agg = strategy.aggregate_evaluate

    def agg_with_tracking(server_round, results, failures):
        t_round = time.time() - round_start[0]
        round_times.append(t_round)
        round_start[0] = time.time()
        actual_rounds[0] = server_round

        agg = orig_agg(server_round, results, failures)
        if agg is not None:
            _, metrics_agg = agg
            acc = metrics_agg.get("accuracy",  0.0)
            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)
            history_acc.append(acc)
            new_lr = lr_scheduler.step(acc)
            print(f"  [Round {server_round:3d}] "
                  f"Acc={acc:.4f} Pre={pre:.4f} "
                  f"Rec={rec:.4f} F1={f1:.4f} "
                  f"LR={new_lr:.6f} t={t_round:.1f}s")
            if not early_stopper.update(server_round, acc):
                raise EarlyStopException()
        return agg

    strategy.aggregate_evaluate = agg_with_tracking
    strategy.on_fit_config_fn   = lambda _round: lr_scheduler.get_config()

    # 8. Chay
    t0 = time.time()
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources=CLIENT_RESOURCES,
        )
    except Exception as e:
        # Flower sẽ bọc lỗi của chúng ta vào trong e.__cause__
        if isinstance(e.__cause__, EarlyStopException):
            print("\n  >> Da chu dong ngat Flower Simulation vi kich hoat Early Stopping!")
        else:
            raise e  # Nếu là lỗi thật sự (code sai, hết RAM...) thì văng ra để biết
    total_time     = time.time() - t0
    actual_n       = actual_rounds[0]

    # 9. Communication
    bytes_per_round   = 2 * model_bytes * num_clients
    total_comm_mb     = (bytes_per_round * actual_n) / (1024 ** 2)
    comm_per_round_kb = bytes_per_round / 1024
    convergence_round = (history_acc.index(max(history_acc)) + 1
                         if history_acc else 0)

    # 10. Luu
    result = {
        "strategy": strategy_name, "num_clients": num_clients, "alpha": alpha,
        "local_epochs": LOCAL_EPOCHS, "fl_rounds_target": NUM_ROUNDS,
        "fl_rounds_actual": actual_n,
        "best_accuracy": float(max(history_acc)) if history_acc else 0.0,
        "final_accuracy": float(history_acc[-1]) if history_acc else 0.0,
        "convergence_round": convergence_round,
        "accuracy_history": history_acc,
        "total_time_s": round(total_time, 1),
        "avg_time_per_round_s": round(
            sum(round_times) / len(round_times) if round_times else 0, 2),
        "model_size_kb": round(model_kb, 2),
        "comm_per_round_kb": round(comm_per_round_kb, 2),
        "total_comm_mb": round(total_comm_mb, 3),
        # --- GPU metrics (moi them) ---
        **get_gpu_usage(),   # peak_gpu_memory_mb, current_gpu_memory_mb
    }

    save_dir = RESULTS_DIR / f"clients{num_clients}_alpha{alpha}_{strategy_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"\n  >> BestAcc={result['best_accuracy']:.4f} | "
          f"Round={convergence_round} | "
          f"Time={total_time:.0f}s | Comm={total_comm_mb:.1f}MB")
    return result


def generate_comparison_table(all_results: list):
    """In bang so sanh va luu CSV day du sau khi chay xong tat ca thi nghiem."""
    if not all_results:
        return

    # Kiem tra co GPU khong de quyet dinh hien thi cot PeakGPU
    has_gpu = any(r.get("peak_gpu_memory_mb", 0) > 0 for r in all_results)

    # --- In terminal ---
    if has_gpu:
        header = (f"{'Strategy':<20} {'Clients':>7} {'Alpha':>6} "
                  f"{'BestAcc':>8} {'FinalAcc':>9} {'ConvRnd':>8} "
                  f"{'Rounds':>7} {'Time(s)':>8} "
                  f"{'ModelKB':>8} {'CommMB':>7} {'PeakGPU(MB)':>12}")
    else:
        header = (f"{'Strategy':<20} {'Clients':>7} {'Alpha':>6} "
                  f"{'BestAcc':>8} {'FinalAcc':>9} {'ConvRnd':>8} "
                  f"{'Rounds':>7} {'Time(s)':>8} "
                  f"{'ModelKB':>8} {'CommMB':>7} {'PeakGPU(MB)':>12}")

    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("  BANG SO SANH - Centralized Baseline: Acc=90.41%  F1=89.08%")
    print(f"{'='*len(header)}\n{header}\n{sep}")

    for r in sorted(all_results, key=lambda x: x["best_accuracy"], reverse=True):
        peak_gpu = r.get("peak_gpu_memory_mb", 0.0)
        print(f"{r['strategy']:<20} "
              f"{r['num_clients']:>7} "
              f"{r['alpha']:>6.1f} "
              f"{r['best_accuracy']:>8.4f} "
              f"{r['final_accuracy']:>9.4f} "
              f"{r['convergence_round']:>8} "
              f"{r['fl_rounds_actual']:>7} "
              f"{r['total_time_s']:>8.1f} "
              f"{r['model_size_kb']:>8.1f} "
              f"{r['total_comm_mb']:>7.1f} "
              f"{peak_gpu:>12.1f}")
    print(sep)

    # --- Luu CSV ---
    # Tat ca cac truong scalar tu result dict, bo qua accuracy_history (la list)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keys = [
        # Tham so thi nghiem
        "strategy",
        "num_clients",
        "alpha",
        "local_epochs",
        "fl_rounds_target",
        "fl_rounds_actual",
        # Hieu nang
        "best_accuracy",
        "final_accuracy",
        "convergence_round",
        # Thoi gian
        "total_time_s",
        "avg_time_per_round_s",
        # Giao tiep
        "model_size_kb",
        "comm_per_round_kb",
        "total_comm_mb",
        # GPU
        "peak_gpu_memory_mb",
        "current_gpu_memory_mb",
    ]

    csv_path = RESULTS_DIR / "comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"\n  CSV da luu: {csv_path}")
    print(f"  Tong so thi nghiem: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int,   default=None)
    parser.add_argument("--alpha",       type=float, default=None)
    parser.add_argument("--strategy",    type=str,   default=None)
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info()
    print(f"Device  : {device}")
    print(f"GPU     : {gpu_info['gpu_name']}")
    if gpu_info["gpu_available"]:
        print(f"VRAM    : {gpu_info['gpu_total_memory_gb']} GB")
        print(f"CUDA    : {gpu_info['cuda_version']}")
    print(f"Config  : {_CFG_PATH}")
    # Luu gpu_info vao 1 file chung
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gpu_info.json", "w") as f:
        json.dump(gpu_info, f, indent=4)


    all_results = []
    if args.all or not any([args.num_clients, args.alpha, args.strategy]):
        total = len(DEFAULT_NUM_CLIENTS)*len(DEFAULT_ALPHAS)*len(DEFAULT_STRATEGIES)
        print(f"Chay {total} thi nghiem...")
        for nc in DEFAULT_NUM_CLIENTS:
            for al in DEFAULT_ALPHAS:
                for st in DEFAULT_STRATEGIES:
                    r = run_one(nc, al, st, device)
                    if r: all_results.append(r)
    else:
        r = run_one(args.num_clients or 5,
                    args.alpha or 0.3,
                    args.strategy or "fedavg", device)
        if r: all_results.append(r)

    generate_comparison_table(all_results)
    print("\nHoan thanh!")

if __name__ == "__main__":
    main()
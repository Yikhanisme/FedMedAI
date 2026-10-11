"""
Chạy FL simulation giống setup của FedSLD paper (ISBI 2022).
Sử dụng SimpleCNN ~20K params thay vì model của paper (~570K).

Chạy:
    # Chạy tất cả (2 dataset x 2 partition x các strategy)
    python -m experiments.run_fedsld_repro --all

    # Chạy cụ thể
    python -m experiments.run_fedsld_repro \
        --dataset pathmnist \
        --partition practical_noniid \
        --strategy fedavg

    # Chạy nhanh để verify (khuyến nghị bắt đầu từ đây)
    python -m experiments.run_fedsld_repro \
        --dataset organmnist_axial \
        --partition practical_noniid \
        --strategy fedavg
"""

import argparse
import csv
import json
import time
from pathlib import Path
import yaml
import torch
import flwr as fl

from datasets.medmnist_code import get_pathmnist_datasets, get_organmnist_datasets
from datasets.partition import load_partition
from models.cnn import SimpleCNN, get_parameters
from models.cnn_570k import CNN570K
from client.client import make_client_fn, make_fedbn_client_fn
from server.server import get_strategy, EarlyStoppingCallback

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
FS = CFG["fedsld_setup"]   # fedsld_setup block

DATASET_FNS = {
    "pathmnist":        (get_pathmnist_datasets,  9),
    "organmnist_axial": (get_organmnist_datasets, 11),
}
PARTITION_DIR = Path("data/partitions")
RESULTS_DIR   = Path(FS["results_dir"])


# ── Metrics helper ────────────────────────────────────────────────────────────
class EarlyStopException(Exception):
    pass


def run_one(dataset_name, partition_method, strategy_name, device, model_type="20k"):
    get_fn, num_classes = DATASET_FNS[dataset_name]
    num_rounds = FS["partition_settings"][partition_method]["fl_rounds"]

    model_label = "~570K params (FedSLD Paper)" if model_type == "570k" else "~20K params (Ours)"
    print(f"\n{'='*65}")
    print(f"  DATASET  : {dataset_name.upper()}")
    print(f"  PARTITION: {partition_method}")
    print(f"  STRATEGY : {strategy_name.upper()}")
    print(f"  MODEL    : {model_type.upper()} ({model_label})")
    print(f"  CLIENTS  : {FS['num_clients']} | BATCH SIZE: {FS['batch_size']}")
    print(f"  ROUNDS   : {num_rounds} | LR: {FS['learning_rate']} ({FS['optimizer'].upper()})")
    print(f"{'='*65}")

    # Dataset
    train_ds, val_ds, test_ds, _ = get_fn(download=False)

    # Partition
    json_file = PARTITION_DIR / f"{partition_method}_{dataset_name}_seed42_clients12.json"
    if not json_file.exists():
        print(f"  [SKIP] Partition file không tồn tại: {json_file}")
        print(f"  -> Chạy trước: python -m datasets.partition_fedsld")
        return None
    partition = load_partition(str(json_file))

    # Model
    if model_type == "570k":
        net = CNN570K(in_channels=3, num_classes=num_classes).to(device)
    else:
        net = SimpleCNN(in_channels=3, num_classes=num_classes).to(device)
    init_params = fl.common.ndarrays_to_parameters(get_parameters(net))

    # Strategy
    strategy = get_strategy(strategy_name, FS["num_clients"], init_params)

    # Client factory (chọn make_fedbn_client_fn nếu strategy là fedbn)
    client_fn_maker = make_fedbn_client_fn if strategy_name == "fedbn" else make_client_fn
    client_fn = client_fn_maker(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        partition=partition,
        local_epochs=FS["local_epochs"],
        lr=FS["learning_rate"],
        device=device,
        num_classes=num_classes,
        batch_size=FS["batch_size"],
        optimizer_type=FS["optimizer"],
        momentum=FS["momentum"],
        model_type=model_type,
    )

    # Tracking metrics
    history_bta   = []   # Best Test Accuracy per round (weighted avg)
    history_bmcta = []   # Best Mean Client Test Accuracy per round (simple mean)
    round_times   = []
    actual_rounds = [0]
    round_start   = [time.time()]
    es_cfg        = FS.get("early_stopping", {"enabled": False})

    early_stopper = EarlyStoppingCallback(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.001)
    ) if es_cfg.get("enabled", False) else None

    orig_agg = strategy.aggregate_evaluate

    def agg_with_tracking(server_round, results, failures):
        t = time.time() - round_start[0]
        round_times.append(t)
        round_start[0] = time.time()
        actual_rounds[0] = server_round

        agg = orig_agg(server_round, results, failures)
        if agg is not None and results:
            _, metrics_agg = agg

            # BTA: weighted average accuracy (toàn bộ test set)
            bta = metrics_agg.get("accuracy", 0.0)
            history_bta.append(bta)

            # BMCTA: simple mean accuracy các clients (không trọng số)
            accs = [eval_res.metrics.get("accuracy", 0.0) for _, eval_res in results]
            bmcta = sum(accs) / len(accs) if accs else 0.0
            history_bmcta.append(bmcta)

            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)
            print(f"  [Round {server_round:3d}/{num_rounds}] "
                  f"BTA={bta:.4f} BMCTA={bmcta:.4f} "
                  f"Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f} ({t:.1f}s)")

            if early_stopper and not early_stopper.update(server_round, bta):
                raise EarlyStopException()

        return agg

    strategy.aggregate_evaluate = agg_with_tracking
    strategy.on_fit_config_fn   = lambda _: {"lr": FS["learning_rate"]}

    t0 = time.time()
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=FS["num_clients"],
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0},
        )
    except Exception as e:
        if not isinstance(e.__cause__, EarlyStopException):
            raise

    num_params = sum(p.numel() for p in net.parameters())
    model_size_mb = round(num_params * 4 / (1024 * 1024), 3)
    # Tổng băng thông truyền thông FL (upload + download) cho tất cả clients qua các rounds
    comm_per_round_mb = (num_params * 4 * 2 * FS["num_clients"]) / (1024 * 1024)
    total_comm_mb = round(comm_per_round_mb * actual_rounds[0], 1)
    total_time_min = round(total_time / 60.0, 1)

    result = {
        "dataset":           dataset_name,
        "partition_method":  partition_method,
        "strategy":          strategy_name,
        "model":             f"{model_type} ({'~570k' if model_type == '570k' else '~20k'})",
        "num_params":        num_params,
        "model_size_mb":     model_size_mb,
        "total_comm_mb":     total_comm_mb,
        "total_time_min":    total_time_min,
        "total_time_s":      total_time,
        "num_clients":       FS["num_clients"],
        "fl_rounds_target":  num_rounds,
        "fl_rounds_actual":  actual_rounds[0],
        "optimizer":         FS["optimizer"],
        "learning_rate":     FS["learning_rate"],
        "batch_size":        FS["batch_size"],
        # Metrics chính (theo paper)
        "BTA":               round(max(history_bta),   4) if history_bta   else 0.0,
        "BMCTA":             round(max(history_bmcta), 4) if history_bmcta else 0.0,
        # Lịch sử từng round
        "bta_per_round":     [round(x, 4) for x in history_bta],
        "bmcta_per_round":   [round(x, 4) for x in history_bmcta],
        "round_times_s":     [round(x, 1) for x in round_times],
    }

    # Lưu kết quả ra file JSON
    run_name = f"{dataset_name}_{partition_method}_{strategy_name}_model{model_type}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n  [COMPLETED] -> BTA = {result['BTA']:.4f} | BMCTA = {result['BMCTA']:.4f}")
    print(f"  [SAVED]     -> {out_dir / 'results.json'}")
    return result


def main():
    valid_strategies = list(set(FS.get("fl_strategies", ["fedavg", "fedprox"]) + ["fedbn"]))

    parser = argparse.ArgumentParser(description="FedSLD Reproduction Simulation")
    parser.add_argument("--dataset",   choices=list(DATASET_FNS.keys()), default="organmnist_axial")
    parser.add_argument("--partition", choices=["practical_noniid", "pathological_noniid"], default="practical_noniid")
    parser.add_argument("--strategy",  choices=valid_strategies, default="fedavg")
    parser.add_argument("--model",     choices=["20k", "570k"], default="20k", help="Model architecture: 20k (ours) or 570k (FedSLD paper)")
    parser.add_argument("--all",       action="store_true", help="Run all experiment scenarios")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    all_results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        for ds in list(DATASET_FNS.keys()):
            for pt in ["practical_noniid", "pathological_noniid"]:
                for st in FS.get("fl_strategies", ["fedavg", "fedprox"]):
                    r = run_one(ds, pt, st, device, model_type=args.model)
                    if r:
                        all_results.append(r)
    else:
        r = run_one(args.dataset, args.partition, args.strategy, device, model_type=args.model)
        if r:
            all_results.append(r)

    # Xuất CSV tổng hợp
    if all_results:
        keys = [
            "dataset", "partition_method", "strategy", "model",
            "num_params", "model_size_mb", "total_comm_mb",
            "total_time_min", "total_time_s",
            "BTA", "BMCTA", "fl_rounds_actual", "optimizer", "learning_rate"
        ]
        file_exists = csv_path.exists()
        with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            w.writerows(all_results)
        print(f"\n  [CSV EXPORT] -> {csv_path}")


if __name__ == "__main__":
    main()

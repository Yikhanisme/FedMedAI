"""
Quét toàn bộ thư mục results/simulate_federated,
đọc từng file results.json và tạo lại comparison_table.csv.

Dùng khi: quá trình --all bị crash giữa chừng, muốn tạo CSV từ kết quả đã có.

Chạy:
    python -m experiments.build_csv
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "simulate_federated"

KEYS = [
    "strategy", "num_clients", "alpha",
    "best_accuracy", "final_accuracy", "convergence_round",
    "fl_rounds_actual", "fl_rounds_target",
    "total_time_s", "avg_time_per_round_s",
    "model_size_kb", "comm_per_round_kb", "total_comm_mb",
    "peak_gpu_memory_mb", "current_gpu_memory_mb",
]

def main():
    result_files = sorted(RESULTS_DIR.glob("*/results.json"))

    if not result_files:
        print(f"Không tìm thấy file results.json nào trong: {RESULTS_DIR}")
        return

    print(f"Tìm thấy {len(result_files)} kết quả:")
    all_results = []
    for f in result_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        all_results.append(data)
        print(f"  ✓ {f.parent.name}  →  Acc={data.get('best_accuracy', 'N/A'):.4f}")

    csv_path = RESULTS_DIR / "comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(KEYS) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(k, "")) for k in KEYS) + "\n")

    print(f"\n✅ CSV đã lưu: {csv_path}")
    print(f"   Tổng số thí nghiệm: {len(all_results)}")

if __name__ == "__main__":
    main()
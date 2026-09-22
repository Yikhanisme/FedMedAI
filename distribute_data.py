"""
Script tự động phân phối dữ liệu phân vùng (data partition) từ Laptop đến 10 thiết bị Jetson qua mạng Wi-Fi/LAN (SCP).

Cấu hình cụm:
- 5 NVIDIA Jetson Orin (Client 0 -> 4)
- 5 NVIDIA Jetson Nano (Client 5 -> 9)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi mã hóa cp1252
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# CẤU HÌNH DANH SÁCH 10 THIẾT BỊ JETSON (5 ORIN + 5 NANO)
# Bạn hãy cập nhật đúng IP, username và đường dẫn thư mục lưu trên máy Jetson
# ==============================================================================
JETSON_CLIENTS = [
    # --- 5 NVIDIA Jetson Orin (Clients 0 - 4) ---
    {
        "client_id": 0,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.50",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 1,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.51",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 2,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.52",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 3,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.53",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 4,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.54",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },

    # --- 5 NVIDIA Jetson Nano (Clients 5 - 9) ---
    {
        "client_id": 5,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.60",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 6,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.61",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 7,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.62",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 8,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.63",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 9,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.64",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
]


def check_ping(ip: str) -> bool:
    """Kiểm tra xem thiết bị có đang online trong mạng LAN/Wi-Fi không."""
    param = "-n 1" if sys.platform.startswith("win") else "-c 1"
    timeout_param = "-w 1000" if sys.platform.startswith("win") else "-W 1"
    command = f"ping {param} {timeout_param} {ip}"
    result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def push_data_to_client(client_info: dict, local_data_root: Path) -> bool:
    """Gửi dữ liệu phân vùng của 1 client cụ thể sang máy Jetson tương ứng."""
    cid = client_info["client_id"]
    ip = client_info["ip"]
    user = client_info["username"]
    dev_type = client_info["device_type"]
    remote_dir = client_info["remote_dir"]

    # Đường dẫn thư mục dữ liệu cục bộ trên Laptop
    local_client_data = local_data_root / f"client_{cid}"

    print(f"\n" + "=" * 65)
    print(f"📡 Client #{cid} | Loại: {dev_type} | Đích: {user}@{ip}")
    print("=" * 65)

    # 1. Kiểm tra thư mục dữ liệu trên Laptop
    if not local_client_data.exists():
        print(f"❌ LỖI: Không tìm thấy thư mục dữ liệu cục bộ: {local_client_data}")
        print(f"👉 Vui lòng chạy script phân chia dữ liệu (datasets/partition.py) trước!")
        return False

    # 2. Kiểm tra kết nối mạng
    print(f"🔍 Đang kiểm tra kết nối mạng tới {ip}...")
    if not check_ping(ip):
        print(f"⚠️ CẢNH BÁO: Không ping được tới {ip}. Máy có thể đang tắt hoặc sai IP!")
        choice = input("Bạn có muốn tiếp tục thử gửi qua SCP không? (y/n): ").strip().lower()
        if choice != 'y':
            print("⏭️ Đã bỏ qua thiết bị này.")
            return False

    # 3. Tạo thư mục đích trên máy Jetson qua SSH (nếu chưa có)
    ssh_mkdir_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "mkdir -p {remote_dir}"'
    subprocess.run(ssh_mkdir_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Đẩy dữ liệu qua SCP
    remote_target = f"{user}@{ip}:{remote_dir}"
    scp_cmd = f"scp -r {local_client_data} {remote_target}"

    print(f"🚀 Đang truyền {local_client_data.name} sang {remote_target} ...")
    ret = subprocess.run(scp_cmd, shell=True)

    if ret.returncode == 0:
        print(f"✅ Đã gửi thành công dữ liệu cho Client #{cid} ({dev_type})!")
        return True
    else:
        print(f"❌ Lỗi khi gửi dữ liệu sang {ip} (Mã lỗi: {ret.returncode}).")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tự động phân phối dữ liệu phân vùng FL tới 10 thiết bị Jetson qua Wi-Fi.")
    parser.add_argument(
        "--client_id", 
        type=int, 
        default=None, 
        help="Chỉ đẩy dữ liệu cho 1 client cụ thể (0 đến 9). Mặc định là đẩy cho tất cả 10 máy."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data", 
        help="Đường dẫn tới thư mục chứa dữ liệu phân vùng trên Laptop (Mặc định: 'data')."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    local_data_root = project_root / args.data_dir

    print("\n" + "#" * 65)
    print("      HỆ THỐNG PHÂN PHỐI DỮ LIỆU FEDMEDAI QUA MẠNG WI-FI")
    print(f"      Tổng số thiết bị: {len(JETSON_CLIENTS)} (5 Orin + 5 Nano)")
    print("#" * 65)

    if args.client_id is not None:
        target_clients = [c for c in JETSON_CLIENTS if c["client_id"] == args.client_id]
        if not target_clients:
            print(f"❌ Client ID không hợp lệ: {args.client_id}. Phải từ 0 đến {len(JETSON_CLIENTS)-1}.")
            return
    else:
        target_clients = JETSON_CLIENTS

    success_count = 0
    total_count = len(target_clients)

    for client in target_clients:
        if push_data_to_client(client, local_data_root):
            success_count += 1

    print("\n" + "#" * 65)
    print(f"🏁 KẾT QUẢ: Đã hoàn tất {success_count}/{total_count} thiết bị thành công!")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()


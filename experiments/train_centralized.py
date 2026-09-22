# File chạy Baseline tập trung (Centralized Baseline)

import os
import json
import torch
import psutil
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.cnn import SimpleCNN
from datasets.medmnist import  get_bloodmnist_datasets, get_bloodmnist_dataloaders
from client.train import train
from client.evaluate import evaluate
import datetime 
def get_resource_usage():
    """
    Lấy thông tin tài nguyên phần cứng hiện tại (RAM, CPU, GPU).
    """
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    gpu_memory_allocated = 0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Chuyển đổi sang MB
    return{
        
        "ram_usage": ram_usage,
        "cpu_usage": cpu_usage,
        "gpu_memory_allocated_MB": gpu_memory_allocated
    }

def plot_and_save_result(train_metrics, eval_metrics, save_dir):
    """Hàm vẽ biểu đồ và lưu ảnh"""
    history = train_metrics['history']
    epochs = range(1, len(history['loss']) + 1)

    #loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', marker='o', label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', marker='s', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Đồ thị Accuracy (Train vs Val)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', marker='o', label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r-', marker='s', label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    # 2. Vẽ Confusion Matrix
    # [IMPROVE 1] Thêm tên class thật của BloodMNIST vào trục X/Y thay vì chỉ để số
    class_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG',
                   'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
    cm = np.array(eval_metrics['confusion_matrix'])
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    
    # 3. Vẽ Precision, Recall, F1-Score cho từng class
    precision_class = eval_metrics['precision_class']
    recall_class = eval_metrics['recall_class']
    f1_score_class = eval_metrics['f1_score_class']
    
    num_classes = len(f1_score_class)
    classes = np.arange(num_classes)
    width = 0.25  # Độ rộng của mỗi cột
    
    plt.figure(figsize=(12, 6))
    
    # Vẽ 3 cột đứng cạnh nhau cho mỗi class
    plt.bar(classes - width, precision_class, width, label='Precision', color='lightcoral')
    plt.bar(classes, recall_class, width, label='Recall', color='lightgreen')
    plt.bar(classes + width, f1_score_class, width, label='F1-Score', color='skyblue')
    
    plt.title('Precision, Recall, and F1-Score per Class')
    plt.xlabel('Class ID')
    plt.ylabel('Score')
    plt.xticks(classes)
    
    # Kéo bảng chú thích (Legend) ra góc ngoài để không che biểu đồ
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Đổi tên file lưu thành per_class_metrics.png cho ý nghĩa
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'))
    plt.close()

def plot_summary_card(train_metrics, eval_metrics, resource_before, resource_after, dataset_info, save_dir):
    """
    Vẽ một ảnh tổng hợp tất cả thông số quan trọng (Summary Card).
    Không bao gồm chỉ số từng class vì đã có biểu đồ riêng.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    # Màu nền của từng nhóm
    color_title   = '#2C3E50'
    color_model   = '#1A5276'
    color_train   = '#145A32'
    color_resource= '#6E2F7E'
    color_data    = '#7D6608'
    color_row_a   = '#EBF5FB'
    color_row_b   = '#FDFEFE'

    y = 0.98  # Vị trí bắt đầu từ trên xuống

    def draw_section_header(ax, y, title, color):
        """Vẽ tiêu đề của từng nhóm"""
        ax.add_patch(plt.Rectangle((0, y - 0.025), 1, 0.03,
                                   transform=ax.transAxes, color=color, zorder=2))
        ax.text(0.5, y - 0.01, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white', zorder=3)
        return y - 0.025

    def draw_row(ax, y, label, value, bg_color):
        """Vẽ một dòng thông số"""
        ax.add_patch(plt.Rectangle((0, y - 0.025), 1, 0.026,
                                   transform=ax.transAxes, color=bg_color, zorder=1))
        ax.text(0.05, y - 0.012, label, transform=ax.transAxes,
                ha='left', va='center', fontsize=10, color='#2C3E50')
        ax.text(0.95, y - 0.012, value, transform=ax.transAxes,
                ha='right', va='center', fontsize=10, fontweight='bold', color='#1A1A1A')
        return y - 0.026

    # ── TIÊU ĐỀ CHÍNH ──
    ax.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.045,
                               transform=ax.transAxes, color=color_title, zorder=2))
    ax.text(0.5, y - 0.018, 'CENTRALIZED BASELINE — RESULT SUMMARY',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=3)
    y -= 0.05

    # ── NHÓM 1: THÔNG TIN DỮ LIỆU ──
    y = draw_section_header(ax, y, 'DATASET INFO', color_data)
    rows = [
        ('Train Samples',   f"{dataset_info['train_samples']:,}"),
        ('Val Samples',     f"{dataset_info['val_samples']:,}"),
        ('Test Samples',    f"{dataset_info['test_samples']:,}"),
        ('Num Classes',     f"{dataset_info['num_classes']}"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # ── NHÓM 2: KẾT QUẢ MÔ HÌNH (TỔNG THỂ) ──
    y = draw_section_header(ax, y, 'MODEL METRICS (Test Set)', color_model)
    rows = [
        ('Loss',              f"{eval_metrics['loss']:.4f}"),
        ('Accuracy',          f"{eval_metrics['accuracy']:.4f}  ({eval_metrics['accuracy']*100:.2f}%)"),
        ('Precision (macro)', f"{eval_metrics['precision']:.4f}"),
        ('Recall    (macro)', f"{eval_metrics['recall']:.4f}"),
        ('F1-Score  (macro)', f"{eval_metrics['f1_score']:.4f}"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # ── NHÓM 3: THÔNG SỐ HUẤN LUYỆN ──
    history = train_metrics['history']
    total_epochs = len(history['loss'])
    y = draw_section_header(ax, y, 'TRAINING INFO', color_train)
    rows = [
        ('Total Epochs Run',              f"{total_epochs}"),
        ('Best Epoch (saved weights)',    f"{train_metrics.get('best_epoch', 'N/A')}"),
        ('Best Val Loss  (best weights)', f"{train_metrics['best_val_loss']:.4f}"
                                          if isinstance(train_metrics.get('best_val_loss'), float) else 'N/A'),
        ('Best Val Acc   (best weights)', f"{train_metrics['best_val_acc']:.4f}  ({train_metrics['best_val_acc']*100:.2f}%)"
                                          if isinstance(train_metrics.get('best_val_acc'), float) else 'N/A'),
        ('Final Train Loss (last epoch)', f"{train_metrics['final_loss']:.4f}"),
        ('Training Time',                 f"{train_metrics['training_time']:.1f} s  ({train_metrics['training_time']/60:.1f} min)"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    y -= 0.01

    # ── NHÓM 4: THÔNG SỐ PHẦN CỨNG ──
    y = draw_section_header(ax, y, 'HARDWARE RESOURCES', color_resource)
    rows = [
        ('CPU Usage (before)',      f"{resource_before['cpu_usage']:.1f}%"),
        ('CPU Usage (after)',       f"{resource_after['cpu_usage']:.1f}%"),
        ('RAM Usage (before)',      f"{resource_before['ram_usage']:.1f}%"),
        ('RAM Usage (after)',       f"{resource_after['ram_usage']:.1f}%"),
        ('GPU Memory (before)',     f"{resource_before['gpu_memory_allocated_MB']:.1f} MB"),
        ('GPU Memory (after)',      f"{resource_after['gpu_memory_allocated_MB']:.1f} MB"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = draw_row(ax, y, lbl, val, color_row_a if i % 2 == 0 else color_row_b)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_card.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Summary card saved: summary_card.png")

def run_centralized():
    print("Starting centralized training...")
    batch_size = 32
    epochs = 50
    learning_rate = 0.001


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/centralized/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    #load dataset
    print("Loading BloodMNIST dataset...")
    # [FIX 3] Sửa "num_casses" thành "num_classes" (typo)
    # [FIX 4] Xóa "batch_size=batch_size" khỏi get_bloodmnist_datasets vì hàm đó không nhận tham số batch_size
    train_data, val_data, test_data, num_classes = get_bloodmnist_datasets(download=True)
    train_loader, val_loader, test_loader, num_classes = get_bloodmnist_dataloaders(batch_size=batch_size)
    print(f"Số lượng mẫu tập Train: {len(train_data)}")
    print(f"Số lượng mẫu tập Val: {len(val_data)}")
    print(f"Số lượng mẫu tập Test: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    resource_usage_before = get_resource_usage()

    print("Starting training...")
    training_metrics = train(model, train_loader, optimizer, epochs, device, val_loader=val_loader)

    resource_usage_after = get_resource_usage()

    print("Evaluating on test_set...")
    loss, eval_metrics = evaluate(model, test_loader, device)

    # [IMPROVE 2] In tóm tắt kết quả đầy đủ thay vì chỉ in 1 dòng
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS ON TEST SET")
    print(f"{'='*55}")
    print(f"  Loss            : {loss:.4f}")
    print(f"  Accuracy        : {eval_metrics['accuracy']:.4f}")
    print(f"  Precision(macro): {eval_metrics['precision']:.4f}")
    print(f"  Recall   (macro): {eval_metrics['recall']:.4f}")
    print(f"  F1-Score (macro): {eval_metrics['f1_score']:.4f}")
    print(f"  Train Time      : {training_metrics['training_time']:.1f}s")
    print(f"{'='*55}")

    plot_and_save_result(training_metrics, eval_metrics, save_dir)

    plot_summary_card(
        train_metrics=training_metrics,
        eval_metrics=eval_metrics,
        resource_before=resource_usage_before,
        resource_after=resource_usage_after,
        dataset_info={
            "train_samples": len(train_data),
            "val_samples":   len(val_data),
            "test_samples":  len(test_data),
            "num_classes":   num_classes
        },
        save_dir=save_dir
    )

    results = {
        # Bổ sung thêm thông tin dữ liệu
        "dataset_info": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "num_classes": num_classes
        },
        "model_metrics": eval_metrics,
        "resource_metrics": {
            "training_time_seconds": training_metrics["training_time"],
            "hardware_before_train": resource_usage_before,
            "hardware_after_train": resource_usage_after
        }
    }

    with open(os.path.join(save_dir, "centralized_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4,ensure_ascii=False)

    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    run_centralized() #python -m experiments.train_centralized
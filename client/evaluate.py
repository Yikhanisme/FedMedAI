"""Chứa vòng lặp đánh giá cục bộ (local evaluation loop) trên tập validation/test của client để tính toán loss, accuracy."""


import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate(model, test_loader, device):
    """
    Đánh giá mô hình trên tập dữ liệu test.
    Tính toán các chỉ số: Loss, Accuracy, Precision, Recall, F1-score và Confusion Matrix.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions =[]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    # Tính toán Precision, Recall, F1-score tổng thể
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)

    #Tính riêng cho từng class
    precision_class, recall_class, f1_score_class, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)


    conf_matrix = confusion_matrix(all_labels, all_predictions)

    eval_metrics ={
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "precision_class": precision_class.tolist(),
        "recall_class": recall_class.tolist(),
        "f1_score_class": f1_score_class.tolist(),
        "confusion_matrix": conf_matrix.tolist()
    }
    return avg_loss, eval_metrics


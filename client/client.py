"""FL Client - Wrapper Flower cho SimpleCNN."""

import torch
import flwr as fl
from models.cnn import get_parameters, set_parameters, SimpleCNN
from flwr.common import Context
from client.train import train
from client.evaluate import evaluate


def make_client_fn(train_dataset, val_dataset, test_dataset,
                   partition, local_epochs, lr, device):
    """
    Factory function: trả về hàm client_fn dùng cho Flower Simulation.

    Flower sẽ gọi client_fn(cid) mỗi khi cần tạo 1 client ảo.
    partition: List[List[int]] — danh sách indices cho từng client.
    """
    from torch.utils.data import DataLoader, Subset

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])

        # Tạo Subset từ indices đã chia ở Phase 4
        train_subset  = Subset(train_dataset, partition[client_id])
        train_loader  = DataLoader(train_subset, batch_size=32,
                                   shuffle=True, num_workers=0)

        # Val loader: dùng chung 1 val set (mỗi client đánh giá trên val gốc)
        val_loader    = DataLoader(val_dataset,  batch_size=32,
                                   shuffle=False, num_workers=0)

        test_loader   = DataLoader(test_dataset, batch_size=32,
                                   shuffle=False, num_workers=0)

        # Mỗi client có model riêng (sẽ nhận trọng số từ Server khi fit)
        net = SimpleCNN(in_channels=3, num_classes=8).to(device)

        return FlowerClient(
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            local_epochs=local_epochs,
            lr=lr,
            device=device
        ).to_client()

    return client_fn


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader,
                 test_loader, local_epochs, lr, device):
        self.net          = net
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.local_epochs = local_epochs
        self.lr           = lr
        self.device       = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Nhan trong so tu Server -> train cuc bo -> gui ve."""
        set_parameters(self.net, parameters)

        # Doc LR tu config Server (neu Server gui xuong GlobalLRScheduler)
        # Neu khong co, dung LR mac dinh khi khoi tao
        current_lr = config.get("lr", self.lr)
    
        optimizer = torch.optim.Adam(self.net.parameters(), lr=current_lr)
        metrics = train(
            model=self.net,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            val_loader=self.val_loader
        )
        return get_parameters(self.net), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Nhan trong so tu Server -> danh gia tren test set."""
        set_parameters(self.net, parameters)
        loss, eval_metrics = evaluate(self.net, self.test_loader, self.device)
        # Chi lay cac gia tri scalar (float) ma Flower co the aggregate duoc.
        # Cac gia tri dang List (precision_class, confusion_matrix)
        # Flower khong ho tro aggregate truc tiep -> bo qua o day,
        # se xu ly rieng trong run_simulation.py neu can.
        return loss, len(self.test_loader.dataset), {
            "accuracy":  float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall":    float(eval_metrics["recall"]),
            "f1_score":  float(eval_metrics["f1_score"]),
        }   
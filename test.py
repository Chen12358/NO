import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ====== IMPORTANT FOR CLUSTER (NO DISPLAY) ======
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
# ===============================================


class NeuralOperator(nn.Module):
    def __init__(self, coord_dim=1, input_dim=1, output_dim=1, J=64, L=3, hidden_dim=64, lr=0.1):
        super().__init__()
        self.J = J
        self.L = L
        self.coord_dim = coord_dim

        # 初始化网络层和位移参数
        self.nets = nn.ModuleList([
            self._build_network(coord_dim, J + 1, hidden_dim)
            for _ in range(L)
        ])
        self.ccs = nn.ParameterList([
            nn.Parameter(torch.randn(J, coord_dim) * 0.1)
            for _ in range(L)
        ])

        # 注册坐标范围参数
        self.register_buffer('x_min', torch.tensor(0.0))
        self.register_buffer('x_max', torch.tensor(1.0))

    def _build_network(self, in_dim, out_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, out_dim),
        )

    def _interpolate(self, u, xx):
        """使用grid_sample进行二维双线性插值"""
        # u: (batch, input_dim, num_points)
        # xx: (batch, J, num_points, coord_dim=2)
        batch_size, J, num_points, coord_dim = xx.shape
        assert coord_dim == 2, "仅支持二维坐标"

        # 提取空间维度
        H = W = int(num_points ** 0.5)

        # 调整输入到4D格式 (batch*J, input_dim, H, W)
        u_grid = u.unsqueeze(1)  # (batch, 1, input_dim, num_points)
        u_grid = u_grid.expand(-1, J, -1, -1)  # (batch, J, input_dim, num_points)
        u_grid = u_grid.reshape(batch_size * J, -1, H, W)  # (batch*J, input_dim, H, W)

        # 调整坐标到5D格式 (batch, J, H, W, 2)
        xx_grid = xx.reshape(batch_size, J, H, W, 2)

        # 归一化坐标到[-1, 1]范围
        xx_normalized = (xx_grid - self.x_min) / (self.x_max - self.x_min) * 2 - 1

        # 合并batch和J维度以适应grid_sample
        xx_normalized = xx_normalized.reshape(batch_size * J, H, W, 2)

        # 执行双线性插值
        interpolated = torch.nn.functional.grid_sample(
            u_grid,
            xx_normalized,
            align_corners=True,
            mode='bilinear'
        )  # 输出形状: (batch*J, input_dim, H, W)

        # 恢复原始维度
        interpolated = interpolated.reshape(batch_size, J, -1, H, W)
        interpolated = interpolated.reshape(batch_size, J, -1, num_points)

        return interpolated  # (batch, J, input_dim, num_points)

    def forward(self, a_batch, x_coord):
        """
        a_batch: (batch, num_points, input_dim)
        x_coord: (batch, num_points, coord_dim) - 已包含批次维度
        """
        u = a_batch

        for net, cc in zip(self.nets, self.ccs):
            # 生成特征
            features = net(x_coord)  # (batch, num_points, J+1)
            features = features.permute(0, 2, 1)  # (batch, J+1, num_points)
            f, b = features[:, :self.J], features[:, self.J]  # f:(batch,J,num_points), b:(batch,num_points)

            # 计算位移坐标 (保持4D形状)
            xx = x_coord.unsqueeze(1) + cc.view(1, self.J, 1, -1)  # (batch, J, num_points, coord_dim)

            # 插值计算
            Uj = self._interpolate(u, xx)  # (batch, J, input_dim, num_points)

            # 加权求和并激活
            s = torch.einsum('bjdn,bjdn->bdn', Uj, f.unsqueeze(2)) + b.unsqueeze(1)
            u = nn.Sigmoid()(s)

        return u.permute(0, 2, 1)  # (batch, num_points, input_dim)


class OperatorTrainer:
    def __init__(self, config, device):
        self.device = device
        self.model = NeuralOperator(**config).to(device=device)

        print("Trainable params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        for p in self.model.parameters():
            nn.init.uniform_(p, -0.5, 0.5)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4))
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def relative_l2_loss(a, b):
        diff = a - b
        return torch.sum(diff ** 2) / (torch.sum(b ** 2) + 1e-12)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for a_batch, u_batch, x_coord in train_loader:
            # 确保在同一device
            a_batch = a_batch.to(self.device)
            u_batch = u_batch.to(self.device)
            x_coord = x_coord.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(a_batch, x_coord)
            loss = self.loss_fn(pred, u_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for a_batch, u_batch, x_coord in test_loader:
                a_batch = a_batch.to(self.device)
                u_batch = u_batch.to(self.device)
                x_coord = x_coord.to(self.device)

                pred = self.model(a_batch, x_coord)
                total_loss += self.loss_fn(pred, u_batch).item()

        return total_loss


class FunctionDataset(Dataset):
    """处理多维函数数据的Dataset类"""
    def __init__(self, u_in_samples, u_out_samples):
        """
        u_in_samples: 字典包含 {'x': 输入坐标, 'y': 输入函数值}
        u_out_samples: 字典包含 {'x': 输出坐标, 'y': 输出函数值}
        假设所有样本共享相同坐标
        """
        self.a_values = u_in_samples['y']      # [batch, num_points, input_dim]
        self.u_targets = u_out_samples['y']    # [batch, num_points, output_dim]
        self.x_coord = self._validate_coordinates(u_in_samples['x'], u_out_samples['x'])

    def _validate_coordinates(self, in_coord, out_coord):
        assert torch.allclose(in_coord[0], out_coord[0]), "输入输出坐标必须一致"
        return in_coord[0]  # 共享坐标 [num_points, coord_dim]

    def __len__(self):
        return self.a_values.size(0)

    def __getitem__(self, idx):
        return (
            self.a_values[idx],     # [num_points, input_dim]
            self.u_targets[idx],    # [num_points, output_dim]
            self.x_coord            # [num_points, coord_dim]
        )


def collate_fn(batch, shared_coord):
    """处理后的坐标形状应为 (batch, num_points, coord_dim)"""
    a_batch = torch.stack([item[0] for item in batch])  # (batch, num_points, input_dim)
    u_batch = torch.stack([item[1] for item in batch])  # (batch, num_points, output_dim)
    coord_batch = shared_coord.expand(a_batch.size(0), -1, -1)  # (batch, num_points, coord_dim)
    return a_batch, u_batch, coord_batch


def data_pipeline(u_in_samples, u_out_samples, batch_size=32):
    """构建完整数据管道"""
    full_dataset = FunctionDataset(u_in_samples, u_out_samples)

    total_size = len(full_dataset)
    train_size = total_size
    test_size = total_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, full_dataset.x_coord),
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, full_dataset.x_coord),
        num_workers=0,
        pin_memory=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # 加载原始数据
    from data import data_convertor1d, loader, data_convertor2d, data_convert_back2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[device] Using CPU")

    u_in_samples, u_out_samples = loader(
        "darcy_train_16.pt",
        data_convertor2d,
        dtype=torch.float32,
        device=device
    )

    # 数据维度验证
    print("输入样本维度:")
    print(f"u_in_x: {u_in_samples['x'].shape}")
    print(f"u_in_y: {u_in_samples['y'].shape}")
    print(f"u_out_y: {u_out_samples['y'].shape}")

    # 创建数据管道
    train_loader, test_loader = data_pipeline(u_in_samples, u_out_samples, batch_size=64)

    # 模型配置
    config = {
        'coord_dim': u_in_samples['x'].shape[-1],
        'input_dim': u_in_samples['y'].shape[-1],
        'output_dim': u_out_samples['y'].shape[-1],
        'J': 16,
        'L': 10,
        'hidden_dim': 128,
        'lr': 1e-3
    }

    # 初始化训练系统
    trainer = OperatorTrainer(config, device)

    # 训练循环
    for epoch in range(1000):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.evaluate(test_loader)
        print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4e} | Test Loss: {test_loss:.4e}")

    # =======================
    # 保存可视化结果（集群用）
    # =======================
    model = trainer.model.eval()

    test_in_samples, test_out_samples = loader(
        "darcy_test_16.pt",
        data_convertor2d,
        dtype=torch.float32,
        device=device
    )

    fig = plt.figure(figsize=(7, 7))

    # 平均相对误差
    total_relative_loss = 0.0
    for index in range(50):
        input_coord = test_in_samples['x'][index].unsqueeze(0).to(device)
        input_func = test_in_samples['y'][index].unsqueeze(0).to(device)
        out = model(a_batch=input_func, x_coord=input_coord)
        total_relative_loss += OperatorTrainer.relative_l2_loss(out, test_out_samples['y'][index].to(device))

    avg_rel = (total_relative_loss / 50).item() if torch.is_tensor(total_relative_loss) else float(total_relative_loss / 50)
    print("total relative loss: ", avg_rel)

    # 画 3 个样本
    for index in range(3):
        input_coord = test_in_samples['x'][index].unsqueeze(0).to(device)
        input_func = test_in_samples['y'][index].unsqueeze(0).to(device)

        out = model(a_batch=input_func, x_coord=input_coord)
        rel = OperatorTrainer.relative_l2_loss(out, test_out_samples['y'][index].to(device))
        print("relative loss: ", rel.item() if torch.is_tensor(rel) else rel)

        # Input x
        x = data_convert_back2d(test_in_samples['y'][index], 16)
        # Ground-truth
        y = data_convert_back2d(test_out_samples['y'][index], 16)
        # Model prediction
        out_img = data_convert_back2d(out.squeeze(0), 16).to(device)

        ax = fig.add_subplot(3, 3, index * 3 + 1)
        ax.imshow(x.cpu().squeeze().detach().numpy(), cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(3, 3, index * 3 + 2)
        ax.imshow(y.cpu().squeeze().detach().numpy(), cmap="gray")
        if index == 0:
            ax.set_title("Ground-truth y")
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(3, 3, index * 3 + 3)
        ax.imshow(out_img.cpu().squeeze().detach().numpy(), cmap="gray")
        if index == 0:
            ax.set_title("Model prediction")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Inputs, ground-truth output and prediction. avg rel l2={avg_rel:.4e}", y=0.98)
    plt.tight_layout()

    # 保存路径（你可以改成你想要的 out_dir）
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "darcy_pred_grid.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[saved] {save_path}")

    plt.close(fig)

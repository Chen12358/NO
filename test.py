import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NeuralOperator(nn.Module):
    def __init__(self, coord_dim=1, input_dim=1, output_dim=1, J=64, L=3, hidden_dim=64, lr=0.1):
        super().__init__()
        self.J = J
        self.L = L
        self.coord_dim = coord_dim
        
        # 初始化网络层和位移参数
        self.nets = nn.ModuleList([
            self._build_network(coord_dim, J+1, hidden_dim) 
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
        u_grid = u_grid.reshape(batch_size*J, -1, H, W)  # (batch*J, input_dim, H, W)
        
        # 调整坐标到5D格式 (batch, J, H, W, 2)
        xx_grid = xx.reshape(batch_size, J, H, W, 2)
        
        # 归一化坐标到[-1, 1]范围
        xx_normalized = (xx_grid - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        
        # 合并batch和J维度以适应grid_sample
        xx_normalized = xx_normalized.reshape(batch_size*J, H, W, 2)
        
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

    # 修改后的模型forward方法
    def forward(self, a_batch, x_coord):
        """
        a_batch: (batch, num_points, input_dim) 
        x_coord: (batch, num_points, coord_dim) - 已包含批次维度
        """
        batch_size = a_batch.size(0)
        u = a_batch
        # 不再需要扩展坐标，直接使用传入的坐标数据
        for net, cc in zip(self.nets, self.ccs):
            # 生成特征
            features = net(x_coord)  # (batch, num_points, J+1)
            features = features.permute(0, 2, 1)  # (batch, J+1, num_points)
            f, b = features[:, :self.J], features[:, self.J]
            
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
        self.model = NeuralOperator(**config).to(device=device)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        for p in self.model.parameters():
            nn.init.uniform_(p, -0.5, 0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4))
        self.loss_fn = nn.MSELoss()
        
    def relative_l2_loss(a, b):
        diff = a - b
        return torch.sum(diff ** 2) / torch.sum(b ** 2)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for a_batch, u_batch, x_coord in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(a_batch, x_coord)
            
            # 计算损失
            loss = self.loss_fn(pred, u_batch)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for a_batch, u_batch, x_coord in test_loader:
                pred = self.model(a_batch, x_coord)
                total_loss += self.loss_fn(pred, u_batch).item()
                
        return total_loss


import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class FunctionDataset(Dataset):
    """处理多维函数数据的Dataset类"""
    def __init__(self, u_in_samples, u_out_samples):
        """
        u_in_samples: 字典包含 {'x': 输入坐标, 'y': 输入函数值}
        u_out_samples: 字典包含 {'x': 输出坐标, 'y': 输出函数值}
        假设所有样本共享相同坐标
        """
        # 输入函数值 [batch, num_points, input_dim]
        self.a_values = u_in_samples['y']
        
        # 目标输出函数值 [batch, num_points, output_dim]
        self.u_targets = u_out_samples['y']
        
        # 统一坐标检查 [num_points, coord_dim]
        self.x_coord = self._validate_coordinates(u_in_samples['x'], u_out_samples['x'])

    def _validate_coordinates(self, in_coord, out_coord):
        """验证输入输出坐标一致性"""
        assert torch.allclose(in_coord[0], out_coord[0]), "输入输出坐标必须一致"
        return in_coord[0]  # 取第一个样本的坐标，假设所有样本共享

    def __len__(self):
        return self.a_values.size(0)

    def __getitem__(self, idx):
        return (
            self.a_values[idx],    # 输入函数值 [num_points, input_dim]
            self.u_targets[idx],  # 目标输出 [num_points, output_dim] 
            self.x_coord           # 共享坐标 [num_points, coord_dim]
        )

def data_pipeline(u_in_samples, u_out_samples, batch_size=32):
    """构建完整数据管道"""
    # 创建数据集
    full_dataset = FunctionDataset(u_in_samples, u_out_samples)
    
    # 数据集分割
    total_size = len(full_dataset)
    train_size = total_size
    test_size = total_size - train_size
    
    # 随机划分
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, full_dataset.x_coord)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, full_dataset.x_coord)
    )
    
    return train_loader, test_loader

# 更新collate_fn保证正确维度
def collate_fn(batch, shared_coord):
    """处理后的坐标形状应为 (batch, num_points, coord_dim)"""
    a_batch = torch.stack([item[0] for item in batch])  # (batch, 256, 1)
    u_batch = torch.stack([item[1] for item in batch])  # (batch, 256, 1)
    coord_batch = shared_coord.expand(a_batch.size(0), -1, -1)  # (batch, 256, 1)
    return a_batch, u_batch, coord_batch

# 使用示例 ----------------------------------------------------------
if __name__ == "__main__":
    # 加载原始数据
    from data import data_convertor1d, loader, data_convertor2d, data_convert_back2d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_in_samples, u_out_samples = loader(f"darcy_train_16.pt", data_convertor2d, dtype=torch.float32, device=device)
    
    # 数据维度验证
    print("输入样本维度:")
    print(f"u_in_x: {u_in_samples['x'].shape}")  # 应为 [batch_size, num_points, coord_dim]
    print(f"u_in_y: {u_in_samples['y'].shape}")  # 应为 [batch_size, num_points, input_dim]
    print(f"u_out_y: {u_out_samples['y'].shape}") # 应为 [batch_size, num_points, output_dim]

    # 创建数据管道
    train_loader, test_loader = data_pipeline(u_in_samples, u_out_samples, batch_size=64)
    
    # 模型配置 (根据实际数据维度调整)
    config = {
        'coord_dim': u_in_samples['x'].shape[-1],  # 自动获取坐标维度
        'input_dim': u_in_samples['y'].shape[-1],  # 输入函数维度
        'output_dim': u_out_samples['y'].shape[-1], # 输出函数维度
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

    # from draw import plot_multiple_functions
    # with torch.no_grad():
    #     model = trainer.model.eval()
    #     for index in range(100):
    #         input_coord = u_in_samples['x'][index].unsqueeze(0).to(device)  # [1, 256, 1]
    #         input_func = u_in_samples['y'][index].unsqueeze(0).to(device)   # [1, 256, 1]
    #         output_coord = u_out_samples['x'][index].unsqueeze(0).to(device) # [1, 256, 1]
    #         # 正确调用模型（基于最新模型接口）
    #         pred = model(a_batch=input_func, x_coord=input_coord)  # [1, 256, 1]
    #         # result = [{'x': u_out_samples['x'][index], 'y': pred, 'title': 'f1', 'color': 'green', 'label': '1', 'line_style': '-', 'marker': None},
    #         #           {'x': u_out_samples['x'][index], 'y': u_out_samples['y'][index], 'title': 'f3', 'color': 'gray', 'label': '3', 'line_style': '-', 'marker': None}]
    #         # plot_multiple_functions(result)
    model = trainer.model.eval()
    import matplotlib.pyplot as plt
    test_in_samples, test_out_samples = loader(f"darcy_test_16.pt", data_convertor2d, dtype=torch.float32, device=device)
    fig = plt.figure(figsize=(7, 7))
    total_relative_loss = 0
    for index in range(50):
        input_coord = test_in_samples['x'][index].unsqueeze(0).to(device)  # [1, 256, 1]
        input_func = test_in_samples['y'][index].unsqueeze(0).to(device)   # [1, 256, 1]
        output_coord = test_out_samples['x'][index].unsqueeze(0).to(device) # [1, 256, 1]
        # 正确调用模型（基于最新模型接口）
        out = model(a_batch=input_func, x_coord=input_coord)  # [1, 256, 1]
        _y = test_in_samples['y'][index]
        total_relative_loss += OperatorTrainer.relative_l2_loss(out, test_out_samples['y'][index])
    print("total relative loss: ", total_relative_loss / 50)
    for index in range(3):
        input_coord = test_in_samples['x'][index].unsqueeze(0).to(device)  # [1, 256, 1]
        input_func = test_in_samples['y'][index].unsqueeze(0).to(device)   # [1, 256, 1]
        output_coord = test_out_samples['x'][index].unsqueeze(0).to(device) # [1, 256, 1]
        # 正确调用模型（基于最新模型接口）
        out = model(a_batch=input_func, x_coord=input_coord)  # [1, 256, 1]
        _x = test_in_samples['x'][index]
        _y = test_in_samples['y'][index]
        print("relative loss: ", OperatorTrainer.relative_l2_loss(out, test_out_samples['y'][index]))
        # Input x
        x = data_convert_back2d(_y, 32)
        # Ground-truth
        y = data_convert_back2d(test_out_samples['y'][index], 32)
        # Model prediction
        out = out.squeeze(0)
        out = data_convert_back2d(out, 32).to(device)

        ax = fig.add_subplot(3, 3, index * 3 + 1)
        x = x.cpu().squeeze().detach().numpy()
        y = y.cpu().squeeze().detach().numpy()
        ax.imshow(x, cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index * 3 + 2)
        ax.imshow(y.squeeze())
        if index == 0:
            ax.set_title("Ground-truth y")
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index * 3 + 3)
        ax.imshow(out.cpu().squeeze().detach().numpy())
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
    plt.tight_layout()
    fig.show()
    input()
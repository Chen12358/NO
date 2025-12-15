import matplotlib.pyplot as plt
from data import data_convert_back2d, data_convertor2d, loader
from test import data_pipeline, OperatorTrainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_in_samples, test_out_samples = loader(f"darcy_train_32_1000.pt", data_convertor2d, dtype=torch.float32, device=device)
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    input_coord = test_in_samples['x'][index].unsqueeze(0).to(device)  # [1, 256, 1]
    input_func = test_in_samples['y'][index].unsqueeze(0).to(device)   # [1, 256, 1]
    output_coord = test_out_samples['x'][index].unsqueeze(0).to(device) # [1, 256, 1]
    # 正确调用模型（基于最新模型接口）
    out = input_func  # [1, 256, 1]
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
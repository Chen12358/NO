import torch
tensor_data = torch.load("darcy_train_32.pt")
print(tensor_data['x'].shape)
print(tensor_data['y'].shape)
tensor_data['x'] = tensor_data['x'][4000:4500,:,:]
tensor_data['y'] = tensor_data['y'][4000:4500,:,:]

print(tensor_data['x'].shape)
print(tensor_data['y'].shape)
torch.save(tensor_data, "darcy_train_32_1000.pt")

tensor_data = torch.load("darcy_train_32_1000.pt")
print(tensor_data['x'].shape)
print(tensor_data['y'].shape)
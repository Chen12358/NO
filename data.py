import torch
from torch import Tensor
from typing import Sequence

class Cell(Tensor):
    @staticmethod
    def __new__(cls, data: Tensor = None, size: Sequence = None, device = torch.device('cpu')):
        if data is not None:
            if data.dim() != 2:
                raise ValueError("Data must be 2D (got {})".format(data.dim()))
            tensor = data.to(device)
        elif size is not None:
            if len(size) != 2:
                raise ValueError("Size must be 2D (got {})".format(len(size)))
            tensor = torch.zeros(size, device=device)
        else:
            raise ValueError("Either 'data' or 'size' must be provided")

        return super().__new__(cls, tensor)

def x_convertor(row: int, col: int, n: int):
    n += 1
    row += 1
    col += 1
    x = torch.tensor([col/n, (n-row)/n])
    return x

def x_convertor1d(i, n):
    return torch.tensor((i+1)/(n+1))

def data_convertor2d(data: Tensor):
    size = data.size()
    n = size[0]
    total_points =  n * n
    x = torch.zeros((total_points, 2))
    y = torch.zeros((total_points, 1))
    counter = 0
    for row in range(size[0]):
        for col in range(size[1]):
            x[counter] = x_convertor(row, col, n)
            y[counter] = data[row][col]
            counter += 1
    return x, y

def data_convert_back2d(y, n):
    f = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            f[i][j] = y[i*n+j]
    return f

def data_convertor1d(data: Tensor):
    x = torch.zeros_like(data)
    n = data.size()[0]
    for i in range(n):
        x[i] = x_convertor1d(i, n)
    x = x.reshape(x.size()[0], 1)
    y = data.reshape(data.size()[0], 1)
    return x, y

'''
p -> point
Data struct:
p1: p(1,1) p(1,2) p(1,3) ...
p2: p(2,1) p(2,2) ...
...
p_n: ...
'''

def formatter(func, convertor, device = torch.device('cpu')):
    func_map = {'x': [], 'y': []}
    for func_i in func:
        x, y = convertor(func_i)
        func_map['x'].append(Cell(x))
        func_map['y'].append(Cell(y))
    func_map['x'] = torch.stack(func_map['x']).to(device)
    func_map['y'] = torch.stack(func_map['y']).to(device)
    return func_map

def loader(data_path, convertor, dtype, device = torch.device('cpu')) -> Tensor:
    tensor_data = torch.load(data_path)
    in_func = tensor_data['x'].to(dtype=dtype)
    out_func = tensor_data['y'].to(dtype=dtype)

    in_func_map = formatter(in_func, convertor, device)
    out_func_map = formatter(out_func, convertor, device)

    return in_func_map, out_func_map
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import time
import torch.nn.functional as F

# 定义 C++ 源代码，包括 d_sigmoid, lltm_forward 和 lltm_backward 函数
with open("lltm.cpp", "r") as file:
    cpp_source = file.read()

# 使用 PyTorch 的 load_inline 功能来编译并加载 C++ 扩展
lltm_cpp = load_inline(
    name="lltm_cpp",
    cpp_sources=cpp_source,
    # functions=["forward", "backward", "d_sigmoid"]
)

# lltm_cpp = load(
#     name = "lltm_cpp",
#     sources = ["lltm.cpp"],
# )

class LLTMFunction(torch.autograd.Function):
    # def __init__():
    #     super().__init__()
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        # print(f"input: {type(input)}, shape: {input.shape}")
        # print(f"weights: {type(weights)}, shape: {weights.shape}")
        # print(f"bias: {type(bias)}, shape: {bias.shape}")
        # print(f"old_h: {type(old_h)}, shape: {old_h.shape}")
        # print(f"old_cell: {type(old_cell)}, shape: {old_cell.shape}")
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)
        return new_h, new_cell
    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights.data, self.bias.data, *state)
    
import torch

import time

import torch

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features, requires_grad=True)
h = torch.randn(batch_size, state_size, requires_grad=True)
C = torch.randn(batch_size, state_size, requires_grad=True)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start
    if _ % 1000 == 0:
        print(_)

print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))
# Forward: 19.466 s | Backward 33.959 s
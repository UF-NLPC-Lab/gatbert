import torch

class _GRL_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GRL(torch.nn.Module):
    def forward(self, x):
        return _GRL_Func.apply(x)
import torch
import torch.nn as nn

class MemoryEfficientReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Store the sign of the input as a boolean tensor for backward
        ctx.save_for_backward(input)
        return input**2

    @staticmethod
    def backward(ctx, grad_output):
        (positive_mask,) = ctx.saved_tensors
        # Use the boolean mask to compute the gradient
        grad_input = grad_output.clone()
        grad_input[~positive_mask] = 0
        return grad_input

class CustomReLU(nn.Module):
    def forward(self, input):
        return MemoryEfficientReLU.apply(input)


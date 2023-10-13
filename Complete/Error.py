import torch
from torch import nn

class SymetryHooker(list):
    def __init__(self, model):
        super().__init__()


class res(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.activ = nn.ReLU()
        self.fc2 = nn.Linear(15, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x

model = res()

intermediate_outputs = []

# Define a hook function to store the intermediate outputs
def hook_fn(module, input, output):
    intermediate_outputs.append(output)

# Register hooks on the layers whose outputs you want to capture
hook1 = model.fc1.register_forward_hook(hook_fn)
hook2 = model.fc2.register_forward_hook(hook_fn)

input_data = torch.rand(100, 10)
# Forward pass
output = model(input_data)

print(len(intermediate_outputs))

# Don't forget to remove the hooks to avoid memory leaks
hook1.remove()
hook2.remove()

# Register hooks on the layers whose outputs you want to capture
hook3 = model.fc1.register_forward_hook(hook_fn)
hook4 = model.fc2.register_forward_hook(hook_fn)

output = model(input_data)

print(len(intermediate_outputs))

output = model(input_data)

print(len(intermediate_outputs))

hook3.remove()
hook4.remove()

output = model(input_data)

print(len(intermediate_outputs))
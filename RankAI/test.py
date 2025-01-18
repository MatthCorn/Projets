import torch
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(128, 3, 224, 224).to(device)
model = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).to(device)

start = time.time()
for _ in tqdm(range(100)):
    output = model(dummy_input)
    output.mean().backward()
end = time.time()

print(f"Time per iteration: {(end - start) / 100:.4f} seconds")
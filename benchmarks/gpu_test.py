import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU is not available.")

device = torch.device("cuda:0")
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Using:", torch.cuda.get_device_name(0))
properties = torch.cuda.get_device_properties(0)
print("VRAM:", round(properties.total_memory / 1024**3, 2), "GB")

size = 4096
print(f"\nCreating two {size} x {size} matrices...")
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)
print("A device:", A.device)
print("B device:", B.device)

C = A @ B
torch.cuda.synchronize()

runs = 10
start = time.perf_counter()
for _ in range(runs):
    C = A @ B
torch.cuda.synchronize()
end = time.perf_counter()

average_time = (end - start) / runs
print("\nAverage matrix multiplication time:", round(average_time * 1000, 2), "ms")
print("Output shape:", C.shape)
print("Output device:", C.device)

allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
print("\nGPU memory allocated:", round(allocated, 3), "GB")
print("GPU memory reserved:", round(reserved, 3), "GB")

# Simple script to check the availability of a GPU on the system.
import torch


def auto_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what's available."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def check_gpu_access():
    if torch.cuda.is_available():
        print("CUDA GPU is available.")
        print(f"Number of CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA GPU is not available.")

    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) GPU is available.")
        # Note: MPS typically refers to a single GPU on Apple Silicon devices
        print("MPS GPU: Apple Silicon GPU")
    else:
        print("MPS GPU is not available.")

if __name__ == "__main__":
    check_gpu_access()
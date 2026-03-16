# Simple script to check the availability of a GPU on the system.
import torch
from loguru import logger
from octron._logging import setup_logging, print_welcome



def auto_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what's available."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def check_gpu_access():
    setup_logging()
    print_welcome()
    if torch.cuda.is_available():
        logger.info("CUDA GPU is available.")
        logger.info(f"Number of CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA GPU is not available.")

    if torch.backends.mps.is_available():
        logger.info("MPS (Metal Performance Shaders) GPU is available.")
        # Note: MPS typically refers to a single GPU on Apple Silicon devices
        logger.info("MPS GPU: Apple Silicon GPU")
    else:
        logger.info("MPS GPU is not available.")

if __name__ == "__main__":
    check_gpu_access()
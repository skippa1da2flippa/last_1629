import torch
from torch import cuda


def GPU_check() -> cuda.device:
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("Number of GPUs detected:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    return torch.device("cpu")

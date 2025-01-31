import torch


def test_CUDA() -> None:
    assert torch.cuda.is_available()
    assert torch.version.cuda == "12.1"

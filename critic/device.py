from dataclasses import dataclass


@dataclass(frozen=True)
class TorchDeviceInfo:
    torch_available: bool
    torch_version: str | None
    cuda_version: str | None
    cuda_available: bool
    device_name: str | None
    capability: tuple[int, int] | None
    selected_device: str
    reason: str


def select_existing_device(torch_module=None) -> TorchDeviceInfo:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return TorchDeviceInfo(
                False,
                None,
                None,
                False,
                None,
                None,
                "cpu",
                "PyTorch is not installed in this interpreter",
            )

    cuda_available = bool(torch_module.cuda.is_available())
    cuda_version = getattr(torch_module.version, "cuda", None)
    torch_version = getattr(torch_module, "__version__", None)
    if not cuda_available:
        return TorchDeviceInfo(
            True,
            torch_version,
            cuda_version,
            False,
            None,
            None,
            "cpu",
            "existing PyTorch has no usable CUDA",
        )

    capability = tuple(torch_module.cuda.get_device_capability(0))
    device_name = torch_module.cuda.get_device_name(0)
    try:
        torch_module.zeros(1, device="cuda")
    except Exception as exc:
        return TorchDeviceInfo(
            True,
            torch_version,
            cuda_version,
            True,
            device_name,
            capability,
            "cpu",
            f"CUDA kernel probe failed: {exc}",
        )

    return TorchDeviceInfo(
        True,
        torch_version,
        cuda_version,
        True,
        device_name,
        capability,
        "cuda",
        "CUDA kernel probe succeeded",
    )

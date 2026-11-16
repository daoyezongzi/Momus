from critic.device import select_existing_device


class CpuTorch:
    __version__ = "2.13.0+cpu"

    class version:
        cuda = None

    class cuda:
        @staticmethod
        def is_available():
            return False


def test_cpu_torch_selects_cpu_without_installing_anything():
    info = select_existing_device(CpuTorch)
    assert info.selected_device == "cpu"
    assert info.cuda_available is False
    assert info.torch_available is True


def test_missing_torch_selects_cpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    info = select_existing_device(None)
    assert info.selected_device == "cpu"
    assert info.torch_available is False

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--cpu", action="store_true", help="Stub the native extension for CPU unit tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA kernels (no model checkpoint unless stated)")
    if config.getoption("--cpu"):
        import sys
        from types import ModuleType

        class Extension(ModuleType):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                def unavailable(*args, **kwargs):
                    raise NotImplementedError(f"CPU fixture must implement native operation {name}")
                setattr(self, name, unavailable)
                return unavailable

        module = ModuleType("exllamav3.ext")
        module.exllamav3_ext = Extension("exllamav3_ext")
        sys.modules["exllamav3.ext"] = module


@pytest.fixture(autouse = True)
def _restore_current_cuda_device(request):
    # Several test modules pin tensors to "cuda:0" but launch Triton kernels on the *current* device,
    # while others call torch.cuda.set_device() for their own device and never restore it. Reset the
    # current device after every test so the suite's result doesn't depend on collection order
    yield
    if not request.config.getoption("--cpu") and torch.cuda.is_available():
        torch.cuda.set_device(0)

from abc import ABC
import torch.nn

class Model(torch.nn.Module, ABC):
    def __init__(self):
        super(Model, self).__init__()
        self._device = None

    @property
    def device(self): return self._device

    def to(self, *args, **kwargs) -> "Model":
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self._device = device
        return self


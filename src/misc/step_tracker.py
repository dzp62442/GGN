import torch
from torch.multiprocessing import Manager


class StepTracker:
    def __init__(self):
        self.lock = None
        self.step = torch.tensor(0, dtype=torch.int64)
        try:
            manager = Manager()
            self.lock = manager.RLock()
            self.step = torch.tensor(0, dtype=torch.int64).share_memory_()
        except (OSError, PermissionError):
            # 在受限环境下退化为无锁实现，但尽量保持功能
            self.lock = None

    def set_step(self, step: int) -> None:
        if self.lock is None:
            self.step.fill_(step)
        else:
            with self.lock:
                self.step.fill_(step)

    def get_step(self) -> int:
        if self.lock is None:
            return self.step.item()
        with self.lock:
            return self.step.item()

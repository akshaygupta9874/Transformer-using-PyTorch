"""research/shared_epoch.py -- epoch number visible to PERSISTENT spawn DataLoader workers.

Problem: with persistent_workers=True the workers hold their own *copy* of the dataset, so
``dataset.epoch = e`` in the main process is never seen (e.g. choosing the .v1/.v2 source BPE-dropout shard
per epoch silently always uses epoch 0).  mp.Value cannot be pickled outside process creation (breaks
pickle-based tests and spawn args alike on some setups), so this uses an 8-byte memory-mapped file:
picklable by path, identical on Windows/Linux, shared by all processes.

    ep = SharedEpoch(os.path.join(cache_dir, "epoch.bin"))      # in main, passed into the dataset
    ...  # per epoch, in main:  ep.set(epoch); batch_sampler.set_epoch(epoch)
    ...  # inside Dataset.__getitem__ (worker):  e = ep.get(); shard = variants[e % len(variants)]
"""
from __future__ import annotations

import os

import numpy as np


class SharedEpoch:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            np.zeros(1, dtype=np.int64).tofile(path)
        self._mm = None

    def _open(self, mode: str):
        self._mm = np.memmap(self.path, dtype=np.int64, mode=mode, shape=(1,))
        return self._mm

    def set(self, epoch: int) -> None:                    # call from the main process only
        mm = self._mm if (self._mm is not None and self._mm.mode == "r+") else self._open("r+")
        mm[0] = int(epoch)
        mm.flush()

    def get(self) -> int:
        mm = self._mm if self._mm is not None else self._open("r")
        return int(mm[0])

    def __getstate__(self):
        return {"path": self.path, "_mm": None}            # workers re-open read-only lazily

    def close(self) -> None:
        m = getattr(self._mm, "_mmap", None)
        if m is not None:
            m.close()
        self._mm = None

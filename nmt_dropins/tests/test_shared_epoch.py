import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, Dataset
from research.shared_epoch import SharedEpoch


class EpochDS(Dataset):                         # module-level => picklable under spawn
    def __init__(self, ep): self.ep = ep
    def __len__(self): return 8
    def __getitem__(self, i): return torch.tensor([i, self.ep.get()])


def test_pickle_is_tiny_and_roundtrips(tmp_path):
    ep = SharedEpoch(str(tmp_path / "e.bin")); ep.set(3)
    assert len(pickle.dumps(ep)) < 200 and pickle.loads(pickle.dumps(ep)).get() == 3


def test_persistent_spawn_workers_see_epoch_updates(tmp_path):
    ep = SharedEpoch(str(tmp_path / "e.bin"))
    dl = DataLoader(EpochDS(ep), batch_size=4, num_workers=2, multiprocessing_context="spawn", persistent_workers=True)
    seen = []
    for e in range(3):
        ep.set(e)
        seen.append({int(x) for b in dl for x in b[:, 1]})
    assert seen == [{0}, {1}, {2}]
    # contrast: a plain attribute set in the parent is NOT seen by persistent workers
    class Plain(EpochDS):
        pass
    del dl

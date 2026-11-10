import os
import pickle

import numpy as np


class StartBank:
    def __init__(self, bucket_m=100.0, per_bucket=16, frontier=4, seed=0):
        self.bucket_m = float(bucket_m)
        self.per_bucket = int(per_bucket)
        self.frontier = int(frontier)
        self.rng = np.random.default_rng(seed)
        self.buckets = {}

    def add(self, entry):
        b = int(entry["progress_m"] // self.bucket_m)
        items = self.buckets.setdefault(b, [])
        if len(items) < self.per_bucket:
            items.append(entry)
        else:
            items[int(self.rng.integers(len(items)))] = entry

    def sample(self):
        keys = sorted(k for k, v in self.buckets.items() if v)
        if not keys:
            return None
        keys = keys[-self.frontier:]
        items = self.buckets[keys[int(self.rng.integers(len(keys)))]]
        return items[int(self.rng.integers(len(items)))]

    def remove(self, entry):
        for k, items in list(self.buckets.items()):
            for i, e in enumerate(items):
                if e is entry:
                    del items[i]
                    if not items:
                        del self.buckets[k]
                    return True
        return False

    def __len__(self):
        return sum(len(v) for v in self.buckets.values())

    def summary(self):
        return {int(k * self.bucket_m): len(v) for k, v in sorted(self.buckets.items()) if v}

    def save(self, path):
        path = str(path)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.buckets, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, **kw):
        bank = cls(**kw)
        if os.path.exists(str(path)):
            with open(str(path), "rb") as f:
                bank.buckets = pickle.load(f)
        return bank


def load(path, **kw):
    return StartBank.load(path, **kw)


def state_from_entry(entry, client=None):
    data = entry["data"]
    if client is not None and hasattr(client, "state_from_bytes"):
        return client.state_from_bytes(data)
    from tmnf.tmi_client import state_from_bytes
    return state_from_bytes(data)

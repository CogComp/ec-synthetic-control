import json
from typing import List


def read_copes_dataset_dict(path="data/COPES.json") -> List[dict]:
    all = []
    with open(path, "r") as f:
        for i, l in enumerate(f):
            d = json.loads(l)
            d["id"] = i
            all.append(d)
    return all

"""Explicit shared-item CSV input for the CLI and Python API."""

import csv

import numpy as np


def read_paired_csv(path):
    """Return (item_ids, binary outcomes[n, 2]) from one shared-item table.

    Required columns are exactly item_id, model_a, model_b, in any order. Each
    nonempty ID occurs once. Both scores must be literal 0 or 1; missing items
    are errors, not silently dropped or joined by row position across files.
    """
    ids, rows, seen = [], [], set()
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or sorted(reader.fieldnames) != [
            "item_id",
            "model_a",
            "model_b",
        ]:
            raise ValueError("CSV columns must be exactly item_id,model_a,model_b")
        for row in reader:
            item_id = row["item_id"]
            if None in row or not item_id or not item_id.strip() or item_id in seen:
                raise ValueError(
                    "CSV rows require unique nonempty item_id and exactly three fields"
                )
            if any(row[key] not in ("0", "1") for key in ("model_a", "model_b")):
                raise ValueError("CSV model_a and model_b must both be 0 or 1 on every item")
            seen.add(item_id)
            ids.append(item_id)
            rows.append([int(row["model_a"]), int(row["model_b"])])
    if len(rows) < 2:
        raise ValueError("CSV pilot requires at least two shared items")
    return np.array(ids), np.array(rows, dtype=np.uint8)

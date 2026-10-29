"""Write the shared-item CSV example from committed GSM8K pilot answers."""

import csv
import gzip
import io
import json
from pathlib import Path

PILOT = Path("results/prospective/primary/pilot")
EXAMPLE = Path("examples/gsm8k_qwen7_phi35_pilot.csv")


def build_example() -> str:
    """Greedy correctness of Qwen2.5-7B (A) and Phi-3.5-mini (B) on 64 shared pilot items."""
    items = {}
    for slug in ("qwen7", "phi35"):
        with gzip.open(PILOT / f"{slug}.json.gz", "rt") as stream:
            items[slug] = json.load(stream)["benchmarks"]["gsm8k"]
    if [item["item_id"] for item in items["qwen7"]] != [item["item_id"] for item in items["phi35"]]:
        raise ValueError("pilot item order differs between models")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["item_id", "model_a", "model_b"])
    for a, b in zip(items["qwen7"], items["phi35"], strict=True):
        writer.writerow([a["item_id"], int(a["greedy"]["correct"]), int(b["greedy"]["correct"])])
    return buffer.getvalue()


def main() -> None:
    EXAMPLE.parent.mkdir(exist_ok=True)
    EXAMPLE.write_text(build_example(), encoding="utf-8")
    print(f"Wrote {EXAMPLE}.")


if __name__ == "__main__":
    main()

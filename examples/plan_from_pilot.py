"""Plan a paired comparison from a shared-item pilot CSV, then check exact power."""

from pathlib import Path

from eval_power import (
    discordance_upper_bound,
    exact_mcnemar_power,
    paired_variance,
    read_paired_csv,
    required_items,
)

ids, pilot = read_paired_csv(Path(__file__).with_name("gsm8k_qwen7_phi35_pilot.csv"))
a, b = pilot[:, 0], pilot[:, 1]
discordant = int((a != b).sum())
effect = 0.05  # fixed before looking at any confirmation data

plug_in = required_items(effect, paired_variance(a, b))
bound = required_items(effect, discordance_upper_bound(discordant, len(ids)))
print(f"pilot: {len(ids)} shared items, {discordant} discordant, gap {a.mean() - b.mean():+.3f}")
print(f"items per model: plug-in variance {plug_in}, discordance bound {bound}")

# Exact McNemar power if the pilot's discordance rate were the truth.
q = discordant / len(ids)
for name, n in (("plug-in", plug_in), ("bound", bound)):
    power = exact_mcnemar_power(effect, q, n)
    print(f"exact power at q={q:.3f}, 5-point gap, {name} budget of {n}: {power:.3f}")

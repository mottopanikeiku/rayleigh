"""Pilot-based planning for paired binary evaluation outcomes."""

from .io import read_paired_csv
from .stats import (
    clustered_se,
    correlation,
    discordance_upper_bound,
    exact_mcnemar_power,
    gaussian_power,
    holm_adjust,
    mcnemar_pvalue,
    minimum_detectable_difference,
    paired_se,
    paired_variance,
    required_items,
    unpaired_se,
    unpaired_variance,
)

__all__ = [
    "clustered_se",
    "correlation",
    "discordance_upper_bound",
    "exact_mcnemar_power",
    "gaussian_power",
    "holm_adjust",
    "mcnemar_pvalue",
    "minimum_detectable_difference",
    "paired_se",
    "paired_variance",
    "read_paired_csv",
    "required_items",
    "unpaired_se",
    "unpaired_variance",
]

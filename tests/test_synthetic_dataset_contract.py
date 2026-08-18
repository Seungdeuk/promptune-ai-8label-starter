from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic-smoke"
LABELS = ("task","audience","context","format","tone","length","constraint","example")


class SyntheticDatasetContractTest(unittest.TestCase):
    def read(self, split: str):
        with (DATA / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def test_counts(self):
        self.assertEqual(len(self.read("train")), 144)
        self.assertEqual(len(self.read("valid")), 48)
        self.assertEqual(len(self.read("test")), 48)

    def test_binary_labels_and_both_classes(self):
        for split in ("train","valid","test"):
            rows = self.read(split)
            for label in LABELS:
                values = {r[f"missing_{label}"] for r in rows}
                self.assertEqual(values, {"0","1"})

    def test_no_scenario_leakage(self):
        scenarios = {
            split: {r["scenario_id"] for r in self.read(split)}
            for split in ("train","valid","test")
        }
        self.assertFalse(scenarios["train"] & scenarios["valid"])
        self.assertFalse(scenarios["train"] & scenarios["test"])
        self.assertFalse(scenarios["valid"] & scenarios["test"])


if __name__ == "__main__":
    unittest.main()

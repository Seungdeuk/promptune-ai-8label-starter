from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from app.core.labels import LABEL_COLUMNS


class PromptuneDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 256) -> None:
        self.df = pd.read_csv(csv_path)
        required = {"text", *LABEL_COLUMNS}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"missing dataset columns: {sorted(missing)}")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        encoded = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(
            [float(row[column]) for column in LABEL_COLUMNS], dtype=torch.float32
        )
        return item

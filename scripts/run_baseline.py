"""Generate a few thousand sequences and take one Adam step on a tiny transformer."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import (
    NUMERIC_DIM,
    STATIC_DIM,
    SequenceBatch,
    SyntheticSequenceDataset,
    collate_sequences,
)
from src.generator import generate_dataset
from src.schema import DataConfig, load_config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinySequenceTransformer(nn.Module):
    """Embedding plus numeric projection, masked mean pool, class and value heads."""

    def __init__(
        self,
        n_events: int,
        n_classes: int,
        d_model: int = 32,
        nhead: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.event_embedding = nn.Embedding(n_events, d_model)
        self.numeric_projection = nn.Linear(NUMERIC_DIM, d_model)
        self.static_projection = nn.Linear(STATIC_DIM, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.class_head = nn.Linear(d_model, n_classes)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, batch: SequenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.event_embedding(batch.event_ids) + self.numeric_projection(batch.numeric)
        encoded = self.encoder(hidden, src_key_padding_mask=~batch.padding_mask)
        weights = batch.padding_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        pooled = pooled + self.static_projection(batch.static)
        value = self.value_head(pooled).squeeze(-1)
        return self.class_head(pooled), value


def baseline_config(config: DataConfig, output_dir: Path) -> DataConfig:
    """Shrink the full corpus to a few thousand sequences on one worker."""
    raw = config.model_dump()
    raw["n_sequences"] = 512
    raw["chunk_size"] = 256
    raw["num_workers"] = 1
    raw["output_dir"] = str(output_dir)
    return DataConfig.model_validate(raw)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = baseline_config(
        load_config(ROOT / "configs" / "data_config.yaml"),
        ROOT / "data" / "baseline",
    )
    set_seed(config.seed)
    generate_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticSequenceDataset(Path(config.output_dir) / "train")
    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    batch = next(iter(loader)).to(device)
    model = TinySequenceTransformer(config.n_foods, n_classes=config.n_actions).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    class_logits, value_pred = model(batch)
    loss_class = nn.functional.cross_entropy(class_logits, batch.y_class)
    loss_value = nn.functional.mse_loss(value_pred, batch.y_value / 100.0)
    (loss_class + loss_value).backward()
    optimizer.step()

    print(f"device={device}", flush=True)
    print(f"event_ids={tuple(batch.event_ids.shape)}", flush=True)
    print(f"numeric={tuple(batch.numeric.shape)}", flush=True)
    print(f"padding_mask={tuple(batch.padding_mask.shape)}", flush=True)
    print(f"static={tuple(batch.static.shape)}", flush=True)
    print(f"y_class={tuple(batch.y_class.shape)}", flush=True)
    print(f"y_value={tuple(batch.y_value.shape)}", flush=True)
    print(f"loss_class={loss_class.item():.6f}", flush=True)
    print(f"loss_value={loss_value.item():.6f}", flush=True)


if __name__ == "__main__":
    main()

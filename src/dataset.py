"""Streaming reader for simulated meal trajectories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.foods import DIET_PATTERN_NAMES
from src.schema import HOURS_IN_DAY

_DIET_PATTERN_SCALE = np.float32(max(len(DIET_PATTERN_NAMES) - 1, 1))

NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    "carbs_g",
    "protein_g",
    "fat_g",
    "fiber_g",
    "added_sugar_g",
    "gi",
    "hour_sin",
    "hour_cos",
    "walk_minutes",
    "cuisine",
    "observed",
)
STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "age_years",
    "sex",
    "bmi",
    "family_history",
    "fasting_glucose",
    "diet_pattern",
)
NUMERIC_DIM = len(NUMERIC_FEATURE_NAMES)
STATIC_DIM = len(STATIC_FEATURE_NAMES)
_RECORD_BATCH_ROWS = 1024


@dataclass
class SequenceExample:
    """One person trajectory after it has been read from Parquet."""

    event_ids: np.ndarray
    carbs_g: np.ndarray
    protein_g: np.ndarray
    fat_g: np.ndarray
    fiber_g: np.ndarray
    added_sugar_g: np.ndarray
    gi: np.ndarray
    hours: np.ndarray
    walk_minutes: np.ndarray
    cuisine: np.ndarray
    observed: np.ndarray
    static: np.ndarray
    y_class: int
    y_value: float


@dataclass
class SequenceBatch:
    """Padded batch. ``padding_mask`` is true on real meals. ``y_value`` is mg/dL."""

    event_ids: torch.Tensor
    numeric: torch.Tensor
    padding_mask: torch.Tensor
    static: torch.Tensor
    y_class: torch.Tensor
    y_value: torch.Tensor

    def to(self, device: torch.device | str) -> SequenceBatch:
        return SequenceBatch(
            event_ids=self.event_ids.to(device),
            numeric=self.numeric.to(device),
            padding_mask=self.padding_mask.to(device),
            static=self.static.to(device),
            y_class=self.y_class.to(device),
            y_value=self.y_value.to(device),
        )


def _as_array(values: list[object], dtype: type) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _iter_parquet(path: Path) -> Iterator[SequenceExample]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=_RECORD_BATCH_ROWS):
        columns = batch.to_pydict()
        n_rows = len(columns["sequence_id"])
        static = np.stack(
            [
                np.asarray(columns["age_years"], dtype=np.float32) / np.float32(80.0),
                np.asarray(columns["sex"], dtype=np.float32),
                np.asarray(columns["bmi"], dtype=np.float32) / np.float32(40.0),
                np.asarray(columns["family_history"], dtype=np.float32),
                np.asarray(columns["fasting_glucose"], dtype=np.float32) / np.float32(200.0),
                np.asarray(columns["diet_pattern"], dtype=np.float32) / _DIET_PATTERN_SCALE,
            ],
            axis=1,
        )
        for index in range(n_rows):
            event_ids = _as_array(columns["food_ids"][index], np.int64)
            carbs_g = _as_array(columns["carbs_g"][index], np.float32)
            protein_g = _as_array(columns["protein_g"][index], np.float32)
            fat_g = _as_array(columns["fat_g"][index], np.float32)
            fiber_g = _as_array(columns["fiber_g"][index], np.float32)
            added_sugar_g = _as_array(columns["added_sugar_g"][index], np.float32)
            gi = _as_array(columns["gi"][index], np.float32)
            hours = _as_array(columns["hours"][index], np.float32)
            walk_minutes = _as_array(columns["walk_minutes"][index], np.float32)
            cuisine = _as_array(columns["cuisine"][index], np.float32)
            observed = _as_array(columns["observed"][index], np.float32)
            length = int(event_ids.shape[0])
            arrays = (carbs_g, protein_g, fat_g, fiber_g, added_sugar_g, gi, hours, walk_minutes, cuisine, observed)
            if any(array.shape[0] != length for array in arrays):
                raise ValueError(f"ragged lists on sequence_id {columns['sequence_id'][index]}")
            yield SequenceExample(
                event_ids=event_ids,
                carbs_g=carbs_g,
                protein_g=protein_g,
                fat_g=fat_g,
                fiber_g=fiber_g,
                added_sugar_g=added_sugar_g,
                gi=gi,
                hours=hours,
                walk_minutes=walk_minutes,
                cuisine=cuisine,
                observed=observed,
                static=static[index],
                y_class=int(columns["y_class"][index]),
                y_value=float(columns["y_value"][index]),
            )
        del columns


class SyntheticSequenceDataset(IterableDataset[SequenceExample]):
    """Stream one split directory. DataLoader workers each receive a shard of files."""

    def __init__(self, split_dir: str | Path) -> None:
        super().__init__()
        self.split_dir = Path(split_dir)
        self.files = tuple(sorted(self.split_dir.glob("*.parquet")))

    def __iter__(self) -> Iterator[SequenceExample]:
        files = self.files
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        for path in files:
            yield from _iter_parquet(path)


def collate_sequences(examples: list[SequenceExample]) -> SequenceBatch:
    """Pad to the batch max length. Numeric channels are scaled into a small range."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    width = max(int(example.event_ids.shape[0]) for example in examples)
    event_ids = torch.zeros(batch_size, width, dtype=torch.long)
    numeric = torch.zeros(batch_size, width, NUMERIC_DIM, dtype=torch.float32)
    padding_mask = torch.zeros(batch_size, width, dtype=torch.bool)
    static = torch.zeros(batch_size, STATIC_DIM, dtype=torch.float32)
    y_class = torch.zeros(batch_size, dtype=torch.long)
    y_value = torch.zeros(batch_size, dtype=torch.float32)
    for index, example in enumerate(examples):
        length = int(example.event_ids.shape[0])
        event_ids[index, :length] = torch.from_numpy(example.event_ids.astype(np.int64, copy=False))
        hours = example.hours.astype(np.float32, copy=False)
        angle = 2.0 * np.pi * hours / np.float32(HOURS_IN_DAY)
        channels = (
            example.carbs_g / np.float32(80.0),
            example.protein_g / np.float32(40.0),
            example.fat_g / np.float32(30.0),
            example.fiber_g / np.float32(15.0),
            example.added_sugar_g / np.float32(40.0),
            example.gi / np.float32(100.0),
            np.sin(angle).astype(np.float32, copy=False),
            np.cos(angle).astype(np.float32, copy=False),
            example.walk_minutes / np.float32(45.0),
            example.cuisine,
            example.observed,
        )
        for channel, values in enumerate(channels):
            numeric[index, :length, channel] = torch.from_numpy(np.asarray(values, dtype=np.float32))
        padding_mask[index, :length] = True
        static[index] = torch.from_numpy(np.asarray(example.static, dtype=np.float32))
        y_class[index] = example.y_class
        y_value[index] = example.y_value
    return SequenceBatch(
        event_ids=event_ids,
        numeric=numeric,
        padding_mask=padding_mask,
        static=static,
        y_class=y_class,
        y_value=y_value,
    )

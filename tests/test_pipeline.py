"""Pipeline tests against a tiny override of the packaged generation config."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from src.dataset import NUMERIC_DIM, STATIC_DIM, SyntheticSequenceDataset, collate_sequences
from src.foods import DIET_PATTERN_NAMES, ESTIMATION_METHOD, FOOD_CATALOG, MAX_ITEMS, MIN_ITEMS, N_FOODS, QUALITY_FLAG
from src.generator import (
    MANIFEST_NAME,
    generate_chunk,
    generate_dataset,
    make_chunk_generator,
    write_validated_chunk,
)
from src.schema import SPLIT_NAMES, DataConfig, load_config
from src.validate import ChunkValidationError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_config.yaml"


def tiny_config(output_dir: Path, **overrides: object) -> DataConfig:
    """Load the real YAML and shrink it onto ``output_dir``."""
    raw = load_config(CONFIG_PATH).model_dump()
    raw["lengths"]["min_len"] = 8
    raw["lengths"]["max_len"] = 24
    raw["lengths"]["lognormal_mu"] = 2.6
    raw["lengths"]["lognormal_sigma"] = 0.35
    raw.update(
        {
            "n_sequences": 2000,
            "chunk_size": 500,
            "num_workers": 1,
            "output_dir": str(output_dir),
            "seed": 42,
        }
    )
    raw.update(overrides)
    return DataConfig.model_validate(raw)


@dataclass
class GeneratedData:
    root: Path
    config: DataConfig
    counts: dict[str, int]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> GeneratedData:
    root = tmp_path_factory.mktemp("synthetic")
    config = tiny_config(root)
    counts = generate_dataset(config)
    return GeneratedData(root=root, config=config, counts=counts)


def _concat(root: Path, columns: list[str]) -> pa.Table:
    tables = [
        pq.read_table(path, columns=columns)
        for split in SPLIT_NAMES
        for path in sorted((root / split).glob("*.parquet"))
    ]
    if not tables:
        raise AssertionError("no parquet shards were written")
    return pa.concat_tables(tables)


def test_config_rejects_impossible_splits_and_length_bounds() -> None:
    raw = load_config(CONFIG_PATH).model_dump()
    bad_splits = dict(raw)
    bad_splits["splits"] = {"train": 0.9, "validation": 0.2, "test": 0.2}
    with pytest.raises(ValidationError):
        DataConfig.model_validate(bad_splits)

    bad_lengths = dict(raw)
    bad_lengths["lengths"] = dict(raw["lengths"])
    bad_lengths["lengths"]["min_len"] = 64
    bad_lengths["lengths"]["max_len"] = 8
    with pytest.raises(ValidationError):
        DataConfig.model_validate(bad_lengths)


def test_generation_writes_split_directories_with_disjoint_ids(generated: GeneratedData) -> None:
    root = generated.root
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    files = {path.name for path in root.iterdir() if path.is_file()}
    assert directories == set(SPLIT_NAMES)
    assert files == {MANIFEST_NAME}

    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["seed"] == generated.config.seed
    assert manifest["config"]["n_sequences"] == generated.config.n_sequences
    counts = manifest["counts"]
    assert counts == generated.counts
    total = sum(counts.values())
    assert total == generated.config.n_sequences

    id_sets: dict[str, set[int]] = {}
    expected_columns = {
        "sequence_id",
        "diet_pattern",
        "region_id",
        "age_years",
        "sex",
        "bmi",
        "family_history",
        "fasting_glucose",
        "food_ids",
        "cuisine",
        "carbs_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "added_sugar_g",
        "gi",
        "local_name",
        "country",
        "food_region",
        "preparation_state",
        "portion_g",
        "energy_kcal",
        "carbohydrate_g_100g",
        "protein_g_100g",
        "fat_g_100g",
        "fibre_g_100g",
        "sodium_mg",
        "calcium_mg",
        "iron_mg",
        "vitc_mg",
        "estimation_method",
        "quality_flag",
        "hours",
        "walk_minutes",
        "observed",
        "day_index",
        "glucose_2h",
        "glucose_delta",
        "y_class",
        "y_value",
    }
    for split, ratio in (
        ("train", generated.config.splits.train),
        ("validation", generated.config.splits.validation),
        ("test", generated.config.splits.test),
    ):
        assert abs(counts[split] / total - ratio) < 0.05
        paths = list((root / split).glob("*.parquet"))
        assert paths, f"{split} has no shard"
        table = pa.concat_tables([pq.read_table(path) for path in paths])
        assert table.num_rows == counts[split]
        assert expected_columns <= set(table.column_names)
        assert set(table.column("estimation_method").to_pylist()) == {ESTIMATION_METHOD}
        assert set(table.column("quality_flag").to_pylist()) == {QUALITY_FLAG}
        sequence_ids = table.column("sequence_id").to_numpy()
        id_sets[split] = set(sequence_ids.tolist())
        assert len(id_sets[split]) == table.num_rows
        diet = table.column("diet_pattern").to_numpy()
        assert int(diet.min()) >= 0
        assert int(diet.max()) < generated.config.n_diet_patterns
        glucose = table.column("y_value").to_numpy()
        assert float(glucose.min()) >= generated.config.checks.glucose_min
        assert float(glucose.max()) <= generated.config.checks.glucose_max

    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])
    assert len(id_sets["train"] | id_sets["validation"] | id_sets["test"]) == total


def _first_chunk_labels(root: Path, chunk_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = _concat(root, ["sequence_id", "y_class", "y_value"])
    sequence_ids = table.column("sequence_id").to_numpy()
    y_class = table.column("y_class").to_numpy()
    y_value = table.column("y_value").to_numpy()
    selected = sequence_ids < chunk_size
    order = np.argsort(sequence_ids[selected])
    return sequence_ids[selected][order], y_class[selected][order], y_value[selected][order]


def test_same_seed_reproduces_first_chunk_labels(tmp_path: Path) -> None:
    chunk_size = 500
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left = tiny_config(left_dir, n_sequences=1000, chunk_size=chunk_size, seed=123)
    right = tiny_config(right_dir, n_sequences=1000, chunk_size=chunk_size, seed=123)
    generate_dataset(left)
    generate_dataset(right)
    left_ids, left_class, left_value = _first_chunk_labels(left_dir, chunk_size)
    right_ids, right_class, right_value = _first_chunk_labels(right_dir, chunk_size)
    assert left_ids.shape[0] == chunk_size
    assert np.array_equal(left_ids, right_ids)
    assert np.array_equal(left_class, right_class)
    assert np.array_equal(left_value, right_value)


def test_validator_rejects_nan_chunk_without_writing(tmp_path: Path) -> None:
    for split in SPLIT_NAMES:
        (tmp_path / split).mkdir()
    config = tiny_config(tmp_path, n_sequences=400, chunk_size=400)
    rng = make_chunk_generator(config.seed, n_chunks=1, chunk_index=0)
    chunk = generate_chunk(config, start_id=0, n=400, rng=rng)
    chunk.carbs_g[0, 0] = np.nan
    with pytest.raises(ChunkValidationError):
        write_validated_chunk(chunk, config, tmp_path, chunk_index=0)
    assert list(tmp_path.rglob("*.parquet")) == []
    assert list(tmp_path.rglob("*part*")) == []


def test_dataset_collate_finite_tensors_match_lengths(generated: GeneratedData) -> None:
    assert NUMERIC_DIM == 11
    assert STATIC_DIM == 6
    dataset = SyntheticSequenceDataset(generated.root / "train")
    examples = []
    stream: Iterator[object] = iter(dataset)
    for _ in range(12):
        examples.append(next(stream))
    batch = collate_sequences(examples)
    lengths = torch.tensor([example.event_ids.shape[0] for example in examples], dtype=torch.long)
    assert torch.equal(batch.padding_mask.sum(dim=1), lengths)
    width = batch.padding_mask.shape[1]
    for index, length in enumerate(lengths.tolist()):
        assert bool(batch.padding_mask[index, :length].all())
        if length < width:
            assert not bool(batch.padding_mask[index, length:].any())
    assert batch.event_ids.dtype == torch.long
    assert batch.y_class.dtype == torch.long
    assert batch.padding_mask.dtype == torch.bool
    assert batch.numeric.shape[-1] == NUMERIC_DIM
    assert batch.static.shape[-1] == STATIC_DIM
    assert torch.isfinite(batch.numeric).all()
    assert torch.isfinite(batch.static).all()
    assert torch.isfinite(batch.y_value).all()
    assert torch.isfinite(batch.event_ids.to(dtype=torch.float32)).all()


def test_labels_are_not_constant(generated: GeneratedData) -> None:
    dataset = SyntheticSequenceDataset(generated.root / "train")
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_sequences, num_workers=0)
    labels = [batch.y_class for batch in loader]
    found = torch.cat(labels)
    assert found.numel() == generated.counts["train"]
    assert found.unique().numel() >= 2


def _mean_list_column(table: pa.Table, name: str) -> np.ndarray:
    values = table.column(name).to_pylist()
    return np.array([float(np.mean(row)) if row else 0.0 for row in values], dtype=np.float64)


def test_diet_patterns_separate_south_asian_and_western_meals(generated: GeneratedData) -> None:
    table = _concat(generated.root, ["diet_pattern", "cuisine"])
    diet = table.column("diet_pattern").to_numpy()
    cuisine = _mean_list_column(table, "cuisine")
    rice_forward = cuisine[diet == 0]
    western = cuisine[diet == len(DIET_PATTERN_NAMES) - 1]
    assert rice_forward.size > 20
    assert western.size > 20
    assert float(rice_forward.mean()) < float(western.mean()) - 0.15


def test_each_diet_pattern_has_100_to_1000_individual_items() -> None:
    config = load_config(CONFIG_PATH)
    assert 5000 <= config.n_sequences <= 15000
    assert config.lengths.min_len == MIN_ITEMS
    assert config.lengths.max_len == MAX_ITEMS
    assert MIN_ITEMS <= N_FOODS <= MAX_ITEMS
    assert FOOD_CATALOG.energy_kcal.shape == (N_FOODS,)
    assert float(np.min(FOOD_CATALOG.portion_g)) > 0.0
    kcal = 4.0 * FOOD_CATALOG.carbs + 4.0 * FOOD_CATALOG.protein + 9.0 * FOOD_CATALOG.fat
    assert float(np.max(np.abs(FOOD_CATALOG.energy_kcal - kcal))) < 0.05
    density = FOOD_CATALOG.carbs / FOOD_CATALOG.portion_g * 100.0
    assert float(np.max(np.abs(FOOD_CATALOG.carbs_100g - density))) < 0.2
    curd = FOOD_CATALOG.names.index("plain_curd")
    rice = FOOD_CATALOG.names.index("steamed_white_rice")
    guava = FOOD_CATALOG.names.index("guava")
    assert float(FOOD_CATALOG.calcium_mg[curd]) > float(FOOD_CATALOG.calcium_mg[rice])
    assert float(FOOD_CATALOG.vitc_mg[guava]) > float(FOOD_CATALOG.vitc_mg[rice])
    assert len(FOOD_CATALOG.pattern_item_ids) == len(DIET_PATTERN_NAMES)
    for item_ids in FOOD_CATALOG.pattern_item_ids:
        assert MIN_ITEMS <= len(item_ids) <= MAX_ITEMS
        assert len(set(item_ids)) == len(item_ids)


def test_each_food_changes_glucose_along_the_patient_timeline(generated: GeneratedData) -> None:
    table = _concat(generated.root, ["added_sugar_g", "glucose_delta", "day_index", "fasting_glucose", "glucose_2h"])
    sugars = table.column("added_sugar_g").to_pylist()
    deltas = table.column("glucose_delta").to_pylist()
    days = table.column("day_index").to_pylist()
    fasting = table.column("fasting_glucose").to_numpy()
    glucose = table.column("glucose_2h").to_pylist()
    low: list[float] = []
    high: list[float] = []
    for index, (sugar, delta, day, series) in enumerate(zip(sugars, deltas, days, glucose, strict=True)):
        assert day[0] == 0
        assert day == sorted(day)
        assert len(day) == len(sugar) == len(delta) == len(series)
        assert abs(float(series[0]) - float(fasting[index]) - float(delta[0])) < 0.05
        order = np.argsort(np.asarray(sugar, dtype=np.float64))
        quartile = max(len(order) // 4, 1)
        low.extend(float(delta[item]) for item in order[:quartile])
        high.extend(float(delta[item]) for item in order[-quartile:])
    assert float(np.mean(high)) > float(np.mean(low)) + 2.0


def test_added_sugar_raises_simulated_glucose(generated: GeneratedData) -> None:
    table = _concat(generated.root, ["added_sugar_g", "y_value"])
    sugar = _mean_list_column(table, "added_sugar_g")
    glucose = table.column("y_value").to_numpy()
    low = glucose[sugar <= np.quantile(sugar, 0.25)]
    high = glucose[sugar >= np.quantile(sugar, 0.75)]
    assert float(high.mean()) > float(low.mean()) + 5.0

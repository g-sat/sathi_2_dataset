"""Per-chunk checks for meal trajectories. A failure means the chunk is not written."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa
except ModuleNotFoundError:  # pragma: no cover - older pandera layouts
    import pandera as pa

from pandera.errors import SchemaError

from src.foods import FOOD_CATALOG, N_ACTIONS
from src.schema import SPLIT_NAMES, DataConfig, GeneratedChunk, valid_timestep_mask

logger = logging.getLogger(__name__)


class ChunkValidationError(ValueError):
    """A chunk failed checks and must not be written."""


def static_schema(config: DataConfig) -> pa.DataFrameSchema:
    """Pandera schema for the person-level columns."""
    population = config.population
    return pa.DataFrameSchema(
        {
            "sequence_id": pa.Column(np.int64, checks=pa.Check.greater_than_or_equal_to(0), nullable=False, unique=True),
            "diet_pattern": pa.Column(np.int64, checks=pa.Check.isin(list(range(config.n_diet_patterns))), nullable=False),
            "region_id": pa.Column(np.int64, checks=pa.Check.isin(list(range(config.n_regions))), nullable=False),
            "age_years": pa.Column(
                np.float64,
                checks=pa.Check.in_range(population.age_min, population.age_max, include_min=True, include_max=True),
                nullable=False,
            ),
            "sex": pa.Column(np.int64, checks=pa.Check.isin([0, 1]), nullable=False),
            "bmi": pa.Column(
                np.float64,
                checks=pa.Check.in_range(population.bmi_min, population.bmi_max, include_min=True, include_max=True),
                nullable=False,
            ),
            "family_history": pa.Column(np.int64, checks=pa.Check.isin([0, 1]), nullable=False),
            "fasting_glucose": pa.Column(
                np.float64,
                checks=pa.Check.in_range(70.0, 200.0, include_min=True, include_max=True),
                nullable=False,
            ),
        },
        strict=True,
        coerce=False,
    )


def _static_frame(chunk: GeneratedChunk) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": np.asarray(chunk.sequence_id, dtype=np.int64),
            "diet_pattern": np.asarray(chunk.diet_pattern, dtype=np.int64),
            "region_id": np.asarray(chunk.region_id, dtype=np.int64),
            "age_years": np.asarray(chunk.age_years, dtype=np.float64),
            "sex": np.asarray(chunk.sex, dtype=np.int64),
            "bmi": np.asarray(chunk.bmi, dtype=np.float64),
            "family_history": np.asarray(chunk.family_history, dtype=np.int64),
            "fasting_glucose": np.asarray(chunk.fasting_glucose, dtype=np.float64),
        }
    )


def _require_finite(name: str, values: np.ndarray) -> None:
    if values.size == 0 or np.isfinite(values).all():
        return
    raise ChunkValidationError(f"non-finite values in {name}")


def validate_chunk(chunk: GeneratedChunk, config: DataConfig) -> None:
    """Raise ChunkValidationError when a chunk is outside the configured contract."""
    frame = _static_frame(chunk)
    try:
        static_schema(config).validate(frame, lazy=False)
    except SchemaError as exc:
        logger.error("static schema rejected a chunk: %s", exc)
        raise ChunkValidationError(f"static schema failed: {exc}") from exc

    lengths = np.asarray(chunk.length)
    if lengths.size == 0:
        raise ChunkValidationError("chunk is empty")
    if int(lengths.min()) < config.lengths.min_len or int(lengths.max()) > config.lengths.max_len:
        raise ChunkValidationError(
            f"lengths {int(lengths.min())}..{int(lengths.max())} outside "
            f"[{config.lengths.min_len}, {config.lengths.max_len}]"
        )
    width = int(chunk.food_ids.shape[1])
    if width != config.lengths.max_len:
        raise ChunkValidationError(f"meal width {width} does not equal max_len {config.lengths.max_len}")
    for name in (
        "cuisine",
        "carbs_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "added_sugar_g",
        "gi",
        "local_name",
        "country",
        "region",
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
        "hours",
        "walk_minutes",
        "observed",
        "day_index",
        "glucose_2h",
        "glucose_delta",
    ):
        column = getattr(chunk, name)
        if column.shape != chunk.food_ids.shape:
            raise ChunkValidationError(f"{name} shape {column.shape} does not match food_ids")

    valid = valid_timestep_mask(lengths, width)
    if not bool(valid.any()):
        raise ChunkValidationError("chunk has no real meals")

    foods = chunk.food_ids[valid]
    if foods.min() < 0 or foods.max() >= config.n_foods:
        raise ChunkValidationError("food_ids outside the food catalog")
    expected = FOOD_CATALOG.cuisine[foods.astype(np.int64)]
    if not np.array_equal(chunk.cuisine[valid].astype(np.int64), expected.astype(np.int64)):
        raise ChunkValidationError("cuisine flag does not match the food catalog")

    for name in (
        "carbs_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "added_sugar_g",
        "walk_minutes",
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
    ):
        values = np.asarray(getattr(chunk, name), dtype=np.float64)
        _require_finite(name, values)
        if np.any(values[valid] < 0.0):
            raise ChunkValidationError(f"{name} has negative values")
    expected_kcal = 4.0 * np.asarray(chunk.carbs_g) + 4.0 * np.asarray(chunk.protein_g) + 9.0 * np.asarray(chunk.fat_g)
    if float(np.max(np.abs(np.asarray(chunk.energy_kcal) - expected_kcal))) > 0.05:
        raise ChunkValidationError("energy_kcal does not match the serving macros")
    for serving_name, per_100_name in (
        ("carbs_g", "carbohydrate_g_100g"),
        ("protein_g", "protein_g_100g"),
        ("fat_g", "fat_g_100g"),
        ("fiber_g", "fibre_g_100g"),
    ):
        density = np.asarray(getattr(chunk, serving_name), dtype=np.float64) / np.maximum(chunk.portion_g, 1e-6) * 100.0
        if float(np.max(np.abs(density - np.asarray(getattr(chunk, per_100_name), dtype=np.float64)))) > 0.2:
            raise ChunkValidationError(f"{per_100_name} does not match the serving and portion weight")
    gi = np.asarray(chunk.gi, dtype=np.float64)
    _require_finite("gi", gi)
    if np.any((gi[valid] < 0.0) | (gi[valid] > 100.0)):
        raise ChunkValidationError("gi outside 0..100")
    _require_finite("hours", np.asarray(chunk.hours, dtype=np.float64))
    glucose_2h = np.asarray(chunk.glucose_2h, dtype=np.float64)
    glucose_delta = np.asarray(chunk.glucose_delta, dtype=np.float64)
    _require_finite("glucose_2h", glucose_2h)
    _require_finite("glucose_delta", glucose_delta)
    logged = np.asarray(chunk.observed, dtype=bool) & valid
    if np.any((glucose_2h[logged] < 60.0) | (glucose_2h[logged] > 350.0)):
        raise ChunkValidationError("glucose_2h outside 60..350 on logged meals")
    expected_delta = glucose_2h - np.asarray(chunk.fasting_glucose, dtype=np.float64)[:, None]
    if float(np.max(np.abs(glucose_delta - expected_delta))) > 1e-4:
        raise ChunkValidationError("glucose_delta is not the meal effect above fasting")
    expected_day = np.broadcast_to(np.arange(width) // 4, chunk.food_ids.shape)
    if not np.array_equal(np.asarray(chunk.day_index), expected_day):
        raise ChunkValidationError("day_index is not the meal order grouped into days")
    _require_finite("y_value", np.asarray(chunk.y_value, dtype=np.float64))
    if np.any((chunk.y_value < config.checks.glucose_min) | (chunk.y_value > config.checks.glucose_max)):
        raise ChunkValidationError("y_value outside the glucose band")
    if not np.isin(chunk.y_class, list(range(N_ACTIONS))).all():
        raise ChunkValidationError("y_class outside the nutrition actions")
    if lengths.size >= 80 and np.unique(chunk.y_class).size < 2:
        raise ChunkValidationError("nutrition actions collapsed to a single class")

    n_valid = int(valid.sum())
    missing = (~np.asarray(chunk.observed, dtype=bool) & valid).sum() / n_valid
    if not config.checks.missing_rate_min <= float(missing) <= config.checks.missing_rate_max:
        message = (
            f"missing rate {float(missing):.4f} outside "
            f"[{config.checks.missing_rate_min}, {config.checks.missing_rate_max}]"
        )
        logger.error(message)
        raise ChunkValidationError(message)

    if set(SPLIT_NAMES) != {"train", "validation", "test"}:
        raise ChunkValidationError("split names drifted from train/validation/test")

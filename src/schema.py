"""Configuration and the simulated glucose response for South Asian meal sequences.

Each row is one person. The sequence is that person's foods in time order.
``glucose_2h`` is the simulated glucose after each food, and ``glucose_delta``
is how far that meal moved glucose above the person's fasting value.
``y_value`` is the mean of the logged meals. ``y_class`` is a nutrition
adjustment chosen from that same simulation. See ``src.foods.DISCLAIMER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.foods import DIET_PATTERN_NAMES, N_ACTIONS, N_FOODS, REGION_NAMES

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
HOURS_IN_DAY = 24.0
MEAL_SLOT_HOURS: tuple[float, ...] = (8.0, 13.0, 17.0, 20.5)


class SplitRatios(BaseModel):
    """Train, validation, and test mass. The three ratios sum to one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train: float = Field(gt=0.0, lt=1.0)
    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> Self:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {total}")
        return self


class LengthConfig(BaseModel):
    """Number of individual food items in one diet log, drawn from a truncated lognormal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_len: int = Field(ge=1)
    max_len: int = Field(ge=1)
    lognormal_mu: float
    lognormal_sigma: float = Field(gt=0.0)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> Self:
        if self.min_len > self.max_len:
            raise ValueError(f"min_len ({self.min_len}) must be <= max_len ({self.max_len})")
        return self


class PopulationConfig(BaseModel):
    """Person-level covariates for a South Asian glucose simulator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diet_pattern_probs: tuple[float, ...] = Field(min_length=1)
    age_min: float = Field(ge=18.0)
    age_max: float = Field(le=90.0)
    bmi_mean: float = Field(gt=0.0)
    bmi_sigma: float = Field(gt=0.0)
    bmi_min: float = Field(gt=0.0)
    bmi_max: float = Field(gt=0.0)
    family_history_rate: float = Field(ge=0.0, le=1.0)
    portion_sigma: float = Field(gt=0.0)
    walk_rate: float = Field(ge=0.0, le=1.0)
    missing_rate: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> Self:
        if self.age_min > self.age_max:
            raise ValueError("age_min must be <= age_max")
        if self.bmi_min > self.bmi_max:
            raise ValueError("bmi_min must be <= bmi_max")
        if len(self.diet_pattern_probs) != len(DIET_PATTERN_NAMES):
            raise ValueError("diet_pattern_probs must have one weight per diet pattern")
        if any(prob <= 0.0 for prob in self.diet_pattern_probs):
            raise ValueError("diet_pattern_probs must be positive")
        if abs(sum(self.diet_pattern_probs) - 1.0) > 1e-6:
            raise ValueError("diet_pattern_probs must sum to 1")
        return self


class GlucoseConfig(BaseModel):
    """Directional coefficients for the simulated 2-hour glucose.

    Reference body size is BMI 23, the Asian risk range used in WHO guidance,
    not a personal diagnosis threshold. Coefficients are simulator settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    carb_coefficient: float = Field(gt=0.0)
    sugar_coefficient: float = Field(ge=0.0)
    protein_coefficient: float = Field(ge=0.0)
    fat_coefficient: float = Field(ge=0.0)
    walk_coefficient: float = Field(ge=0.0)
    excursion_noise: float = Field(ge=0.0)
    two_hour_fraction: float = Field(gt=0.0, le=1.0)
    target_mgdl: float = Field(gt=70.0, lt=200.0)


class LabelConfig(BaseModel):
    """Thresholds that turn a simulated meal pattern into one nutrition action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    flip_rate: float = Field(ge=0.0, lt=1.0)
    sugar_cut_grams: float = Field(ge=0.0)
    refined_gi: float = Field(ge=0.0, le=100.0)
    refined_carb_grams: float = Field(ge=0.0)
    fiber_gap_grams: float = Field(ge=0.0)
    low_protein_grams: float = Field(ge=0.0)
    high_carb_grams: float = Field(ge=0.0)


class CheckConfig(BaseModel):
    """Acceptance band applied to every chunk before it is written."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_rate_min: float = Field(ge=0.0, le=1.0)
    missing_rate_max: float = Field(ge=0.0, le=1.0)
    glucose_min: float = Field(gt=40.0)
    glucose_max: float = Field(lt=400.0)

    @model_validator(mode="after")
    def bands_are_ordered(self) -> Self:
        if self.missing_rate_min > self.missing_rate_max:
            raise ValueError("missing_rate_min must be <= missing_rate_max")
        if self.glucose_min >= self.glucose_max:
            raise ValueError("glucose_min must be < glucose_max")
        return self


class DataConfig(BaseModel):
    """Resolved generation config. Invalid YAML is rejected before any sampling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0)
    n_sequences: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    num_workers: int = Field(ge=1)
    output_dir: str = Field(min_length=1)
    splits: SplitRatios
    lengths: LengthConfig
    population: PopulationConfig
    glucose: GlucoseConfig
    labels: LabelConfig
    checks: CheckConfig

    @field_validator("output_dir", mode="before")
    @classmethod
    def output_dir_to_str(cls, value: str | Path) -> str:
        return str(value)

    @property
    def n_foods(self) -> int:
        return N_FOODS

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def n_regions(self) -> int:
        return len(REGION_NAMES)

    @property
    def n_diet_patterns(self) -> int:
        return len(DIET_PATTERN_NAMES)


@dataclass
class GeneratedChunk:
    """One in-memory chunk of people, still unsplit and unwritten."""

    sequence_id: np.ndarray
    diet_pattern: np.ndarray
    region_id: np.ndarray
    age_years: np.ndarray
    sex: np.ndarray
    bmi: np.ndarray
    family_history: np.ndarray
    fasting_glucose: np.ndarray
    length: np.ndarray
    food_ids: np.ndarray
    cuisine: np.ndarray
    carbs_g: np.ndarray
    protein_g: np.ndarray
    fat_g: np.ndarray
    fiber_g: np.ndarray
    added_sugar_g: np.ndarray
    gi: np.ndarray
    local_name: np.ndarray
    country: np.ndarray
    region: np.ndarray
    preparation_state: np.ndarray
    portion_g: np.ndarray
    energy_kcal: np.ndarray
    carbohydrate_g_100g: np.ndarray
    protein_g_100g: np.ndarray
    fat_g_100g: np.ndarray
    fibre_g_100g: np.ndarray
    sodium_mg: np.ndarray
    calcium_mg: np.ndarray
    iron_mg: np.ndarray
    vitc_mg: np.ndarray
    hours: np.ndarray
    walk_minutes: np.ndarray
    observed: np.ndarray
    day_index: np.ndarray
    glucose_2h: np.ndarray
    glucose_delta: np.ndarray
    y_class: np.ndarray
    y_value: np.ndarray


def load_config(path: str | Path) -> DataConfig:
    """Load YAML and reject it unless it matches the generation schema."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return DataConfig.model_validate(raw)


def insulin_resistance(
    config: DataConfig,
    *,
    age_years: np.ndarray,
    bmi: np.ndarray,
    family_history: np.ndarray,
) -> np.ndarray:
    """Scale meal excursions. BMI 23 and age 40 are the simulator's reference point."""
    resistance = (
        1.0
        + 0.02 * (bmi - 23.0)
        + 0.004 * (age_years - 40.0)
        + 0.15 * family_history.astype(np.float64)
    )
    return np.clip(resistance, 0.75, 1.9)


def meal_excursion(
    config: DataConfig,
    *,
    resistance: np.ndarray,
    carbs_g: np.ndarray,
    protein_g: np.ndarray,
    fat_g: np.ndarray,
    fiber_g: np.ndarray,
    added_sugar_g: np.ndarray,
    gi: np.ndarray,
    walk_minutes: np.ndarray,
) -> np.ndarray:
    """Positive glucose rise from one meal, before the 2-hour fraction is applied."""
    available = np.maximum(carbs_g - 0.5 * fiber_g, 0.0)
    glucose = config.glucose
    delta = resistance[:, None] * (
        glucose.carb_coefficient * available * (gi / 100.0)
        + glucose.sugar_coefficient * added_sugar_g
        - glucose.protein_coefficient * protein_g
        - glucose.fat_coefficient * fat_g
        - glucose.walk_coefficient * walk_minutes
    )
    return np.clip(delta, 0.0, 180.0)


def two_hour_glucose(config: DataConfig, *, fasting_glucose: np.ndarray, excursion: np.ndarray) -> np.ndarray:
    """Map a meal excursion onto a simulated 2-hour glucose in mg/dL."""
    values = fasting_glucose[:, None] + config.glucose.two_hour_fraction * excursion
    return np.clip(values, 60.0, 350.0)


def nutrition_action(
    config: DataConfig,
    *,
    glucose_mgdl: np.ndarray,
    carbs_g: np.ndarray,
    protein_g: np.ndarray,
    fiber_g: np.ndarray,
    added_sugar_g: np.ndarray,
    gi: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Pick one adjustment from the simulated meals. This is a simulator policy."""
    valid = observed.astype(bool)
    counts = np.maximum(valid.sum(axis=1), 1).astype(np.float64)
    mean_glucose = (np.where(valid, glucose_mgdl, 0.0).sum(axis=1)) / counts
    mean_sugar = (np.where(valid, added_sugar_g, 0.0).sum(axis=1)) / counts
    mean_fiber = (np.where(valid, fiber_g, 0.0).sum(axis=1)) / counts
    mean_protein = (np.where(valid, protein_g, 0.0).sum(axis=1)) / counts
    mean_carbs = (np.where(valid, carbs_g, 0.0).sum(axis=1)) / counts
    refined = np.where(valid & (gi >= config.labels.refined_gi), carbs_g, 0.0)
    mean_refined = refined.sum(axis=1) / counts
    labels = config.labels
    action = np.full(mean_glucose.shape[0], 4, dtype=np.int64)
    hold = mean_glucose < config.glucose.target_mgdl
    cut_sugar = mean_sugar >= labels.sugar_cut_grams
    swap_staple = (mean_refined >= labels.refined_carb_grams) & (mean_fiber < labels.fiber_gap_grams)
    add_protein = (mean_protein < labels.low_protein_grams) & (mean_carbs >= labels.high_carb_grams)
    action[add_protein] = 2
    action[swap_staple] = 1
    action[cut_sugar] = 3
    action[hold] = 0
    return action


def valid_timestep_mask(lengths: np.ndarray, width: int) -> np.ndarray:
    """Return a boolean mask of shape (n, width) for meals inside each trajectory."""
    return np.arange(width)[None, :] < lengths[:, None]

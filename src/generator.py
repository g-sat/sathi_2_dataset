"""Chunked writer of South Asian meal trajectories to split Parquet shards."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.foods import (
    ACTION_NAMES,
    DIET_PATTERNS,
    DISCLAIMER,
    ESTIMATION_METHOD,
    FOOD_CATALOG,
    QUALITY_FLAG,
    REGION_NAMES,
    food_record,
)
from src.schema import (
    SPLIT_NAMES,
    DataConfig,
    GeneratedChunk,
    MEAL_SLOT_HOURS,
    SplitRatios,
    insulin_resistance,
    load_config,
    meal_excursion,
    nutrition_action,
    two_hour_glucose,
    valid_timestep_mask,
)
from src.validate import ChunkValidationError, validate_chunk

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


class ChunkTask(TypedDict):
    """Picklable work item. The seed child is a function of (seed, n_chunks, index) only."""

    chunk_index: int
    start_id: int
    n: int
    n_chunks: int
    config: dict[str, Any]
    output_dir: str


def make_chunk_generator(seed: int, n_chunks: int, chunk_index: int) -> np.random.Generator:
    """Return the SeedSequence child for one chunk."""
    parent = np.random.SeedSequence(seed)
    child = parent.spawn(n_chunks)[chunk_index]
    return np.random.default_rng(child)


def _sample_meals(rng: np.random.Generator, diet_pattern: np.ndarray, width: int) -> np.ndarray:
    n_rows = diet_pattern.shape[0]
    food_ids = np.zeros((n_rows, width), dtype=np.int32)
    probs = FOOD_CATALOG.diet_slot_probs
    for step in range(width):
        slot = step % probs.shape[1]
        for pattern in range(probs.shape[0]):
            rows = np.flatnonzero(diet_pattern == pattern)
            if rows.size == 0:
                continue
            food_ids[rows, step] = rng.choice(probs.shape[2], size=rows.size, p=probs[pattern, slot])
    return food_ids


def generate_chunk(config: DataConfig, *, start_id: int, n: int, rng: np.random.Generator) -> GeneratedChunk:
    """Sample ``n`` people whose sequence ids start at ``start_id``."""
    width = config.lengths.max_len
    population = config.population
    raw_lengths = rng.lognormal(config.lengths.lognormal_mu, config.lengths.lognormal_sigma, size=n)
    raw_lengths = np.nan_to_num(raw_lengths, posinf=float(width), neginf=float(config.lengths.min_len))
    lengths = np.clip(np.rint(raw_lengths), config.lengths.min_len, width).astype(np.int32, copy=False)

    diet_probs = np.asarray(population.diet_pattern_probs, dtype=np.float64)
    diet_probs = diet_probs / diet_probs.sum()
    diet_pattern = rng.choice(config.n_diet_patterns, size=n, p=diet_probs).astype(np.int64, copy=False)
    region_id = np.empty(n, dtype=np.int64)
    for pattern in range(config.n_diet_patterns):
        rows = np.flatnonzero(diet_pattern == pattern)
        if rows.size == 0:
            continue
        region_id[rows] = rng.choice(config.n_regions, size=rows.size, p=FOOD_CATALOG.region_probs[pattern])

    age_years = rng.uniform(population.age_min, population.age_max, size=n)
    sex = rng.integers(0, 2, size=n).astype(np.int64, copy=False)
    bmi = np.clip(
        rng.normal(population.bmi_mean, population.bmi_sigma, size=n),
        population.bmi_min,
        population.bmi_max,
    )
    family_history = (rng.random(n) < population.family_history_rate).astype(np.int64, copy=False)
    fasting = (
        94.0
        + 1.3 * (bmi - 23.0)
        + 0.22 * (age_years - 40.0)
        + 10.0 * family_history.astype(np.float64)
        + rng.normal(0.0, 7.0, size=n)
    )
    fasting_glucose = np.clip(fasting, 75.0, 190.0)

    food_ids = _sample_meals(rng, diet_pattern, width)
    portion = np.clip(rng.lognormal(0.0, population.portion_sigma, size=(n, width)), 0.65, 1.55)
    carbs_g = FOOD_CATALOG.carbs[food_ids] * portion
    protein_g = FOOD_CATALOG.protein[food_ids] * portion
    fat_g = FOOD_CATALOG.fat[food_ids] * portion
    fiber_g = FOOD_CATALOG.fiber[food_ids] * portion
    added_sugar_g = FOOD_CATALOG.added_sugar[food_ids] * portion
    gi = np.array(FOOD_CATALOG.gi[food_ids], dtype=np.float64, copy=True)
    cuisine = np.array(FOOD_CATALOG.cuisine[food_ids], dtype=np.int16, copy=True)
    portion_g = FOOD_CATALOG.portion_g[food_ids] * portion
    energy_kcal = 4.0 * carbs_g + 4.0 * protein_g + 9.0 * fat_g
    local_name = np.asarray(FOOD_CATALOG.local_names, dtype=object)[food_ids]
    country = np.asarray(FOOD_CATALOG.countries, dtype=object)[food_ids]
    food_region = np.asarray(FOOD_CATALOG.regions, dtype=object)[food_ids]
    preparation_state = np.asarray(FOOD_CATALOG.preparation_states, dtype=object)[food_ids]
    sodium_mg = FOOD_CATALOG.sodium_mg[food_ids] * portion
    calcium_mg = FOOD_CATALOG.calcium_mg[food_ids] * portion
    iron_mg = FOOD_CATALOG.iron_mg[food_ids] * portion
    vitc_mg = FOOD_CATALOG.vitc_mg[food_ids] * portion
    carbohydrate_g_100g = np.array(FOOD_CATALOG.carbs_100g[food_ids], dtype=np.float64, copy=True)
    protein_g_100g = np.array(FOOD_CATALOG.protein_100g[food_ids], dtype=np.float64, copy=True)
    fat_g_100g = np.array(FOOD_CATALOG.fat_100g[food_ids], dtype=np.float64, copy=True)
    fibre_g_100g = np.array(FOOD_CATALOG.fiber_100g[food_ids], dtype=np.float64, copy=True)

    slot = np.arange(width) % len(MEAL_SLOT_HOURS)
    hours = np.asarray(MEAL_SLOT_HOURS, dtype=np.float64)[slot][None, :] + rng.normal(0.0, 0.35, size=(n, width))
    hours = np.clip(hours, 5.0, 23.5)
    dinner = slot == 3
    walk_threshold = np.where(dinner[None, :], min(population.walk_rate * 1.4, 0.85), population.walk_rate)
    walk_minutes = np.where(
        rng.random((n, width)) < walk_threshold,
        rng.uniform(8.0, 40.0, size=(n, width)),
        0.0,
    )
    snack = slot == 2
    miss_prob = np.where(snack[None, :], min(population.missing_rate * 1.5, 0.5), population.missing_rate)
    observed = rng.random((n, width)) >= miss_prob
    valid = valid_timestep_mask(lengths, width)
    observed = observed & valid
    none_logged = ~observed.any(axis=1)
    observed[none_logged, 0] = True

    resistance = insulin_resistance(config, age_years=age_years, bmi=bmi, family_history=family_history)
    excursion = meal_excursion(
        config,
        resistance=resistance,
        carbs_g=carbs_g,
        protein_g=protein_g,
        fat_g=fat_g,
        fiber_g=fiber_g,
        added_sugar_g=added_sugar_g,
        gi=gi,
        walk_minutes=walk_minutes,
    )
    excursion = np.clip(excursion + rng.normal(0.0, config.glucose.excursion_noise, size=(n, width)), 0.0, 180.0)
    glucose = two_hour_glucose(config, fasting_glucose=fasting_glucose, excursion=excursion)
    glucose_delta = glucose - fasting_glucose[:, None]
    day_index = np.broadcast_to(np.arange(width, dtype=np.int16) // len(MEAL_SLOT_HOURS), (n, width)).copy()
    logged = observed.astype(np.float64)
    y_value = np.clip(
        (np.where(observed, glucose, 0.0).sum(axis=1) / np.maximum(logged.sum(axis=1), 1.0)),
        config.checks.glucose_min,
        config.checks.glucose_max,
    )
    y_class = nutrition_action(
        config,
        glucose_mgdl=glucose,
        carbs_g=carbs_g,
        protein_g=protein_g,
        fiber_g=fiber_g,
        added_sugar_g=added_sugar_g,
        gi=gi,
        observed=observed,
    )
    flip = rng.random(n) < config.labels.flip_rate
    if np.any(flip):
        replacement = rng.integers(0, config.n_actions, size=n)
        same = replacement == y_class
        replacement[same] = (replacement[same] + 1) % config.n_actions
        y_class = np.where(flip, replacement, y_class).astype(np.int64, copy=False)

    return GeneratedChunk(
        sequence_id=np.arange(start_id, start_id + n, dtype=np.int64),
        diet_pattern=diet_pattern,
        region_id=region_id,
        age_years=age_years,
        sex=sex,
        bmi=bmi,
        family_history=family_history,
        fasting_glucose=fasting_glucose,
        length=lengths,
        food_ids=food_ids,
        cuisine=cuisine,
        carbs_g=carbs_g,
        protein_g=protein_g,
        fat_g=fat_g,
        fiber_g=fiber_g,
        added_sugar_g=added_sugar_g,
        gi=gi,
        local_name=local_name,
        country=country,
        region=food_region,
        preparation_state=preparation_state,
        portion_g=portion_g,
        energy_kcal=energy_kcal,
        carbohydrate_g_100g=carbohydrate_g_100g,
        protein_g_100g=protein_g_100g,
        fat_g_100g=fat_g_100g,
        fibre_g_100g=fibre_g_100g,
        sodium_mg=sodium_mg,
        calcium_mg=calcium_mg,
        iron_mg=iron_mg,
        vitc_mg=vitc_mg,
        hours=hours,
        walk_minutes=walk_minutes,
        observed=observed,
        day_index=day_index,
        glucose_2h=glucose,
        glucose_delta=glucose_delta,
        y_class=y_class,
        y_value=y_value,
    )


def _bucket_unit_interval(sequence_id: np.ndarray) -> np.ndarray:
    """SplitMix64 mix of sequence_id, mapped onto [0, 1)."""
    mixed = sequence_id.astype(np.uint64, copy=False)
    mixed = np.bitwise_xor(mixed, np.right_shift(mixed, 30))
    mixed = np.multiply(mixed, np.uint64(0xBF58476D1CE4E5B9))
    mixed = np.bitwise_xor(mixed, np.right_shift(mixed, 27))
    mixed = np.multiply(mixed, np.uint64(0x94D049BB133111EB))
    mixed = np.bitwise_xor(mixed, np.right_shift(mixed, 31))
    top = np.right_shift(mixed, 11).astype(np.float64)
    return top * (1.0 / 9007199254740992.0)


def assign_splits(sequence_id: np.ndarray, splits: SplitRatios) -> np.ndarray:
    """Bucket sequence ids into train, validation, and test."""
    unit = _bucket_unit_interval(sequence_id)
    assignment = np.empty(sequence_id.shape[0], dtype="U10")
    train_cut = splits.train
    validation_cut = splits.train + splits.validation
    train = unit < train_cut
    validation = (unit >= train_cut) & (unit < validation_cut)
    assignment[train] = "train"
    assignment[validation] = "validation"
    assignment[~(train | validation)] = "test"
    return assignment


def _string_list_array(values: np.ndarray, lengths: np.ndarray) -> pa.Array:
    width = values.shape[1]
    valid = np.arange(width)[None, :] < lengths[:, None]
    flat = values[valid].tolist()
    offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int64()), pa.array(flat, type=pa.string()))


def _list_array(values: np.ndarray, lengths: np.ndarray, value_type: pa.DataType) -> pa.Array:
    width = values.shape[1]
    valid = np.arange(width)[None, :] < lengths[:, None]
    flat = np.ascontiguousarray(values[valid])
    offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int64()), pa.array(flat, type=value_type))


def _table_for_rows(chunk: GeneratedChunk, row_index: np.ndarray) -> pa.Table:
    lengths = np.asarray(chunk.length[row_index], dtype=np.int32)
    return pa.table(
        {
            "sequence_id": pa.array(chunk.sequence_id[row_index], type=pa.int64()),
            "diet_pattern": pa.array(chunk.diet_pattern[row_index], type=pa.int16()),
            "region_id": pa.array(chunk.region_id[row_index], type=pa.int16()),
            "age_years": pa.array(np.asarray(chunk.age_years[row_index], dtype=np.float32), type=pa.float32()),
            "sex": pa.array(chunk.sex[row_index], type=pa.int8()),
            "bmi": pa.array(np.asarray(chunk.bmi[row_index], dtype=np.float32), type=pa.float32()),
            "family_history": pa.array(chunk.family_history[row_index], type=pa.int8()),
            "fasting_glucose": pa.array(np.asarray(chunk.fasting_glucose[row_index], dtype=np.float32), type=pa.float32()),
            "length": pa.array(lengths, type=pa.int32()),
            "food_ids": _list_array(np.asarray(chunk.food_ids[row_index], dtype=np.int32), lengths, pa.int32()),
            "cuisine": _list_array(np.asarray(chunk.cuisine[row_index], dtype=np.int16), lengths, pa.int16()),
            "carbs_g": _list_array(np.asarray(chunk.carbs_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "protein_g": _list_array(np.asarray(chunk.protein_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "fat_g": _list_array(np.asarray(chunk.fat_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "fiber_g": _list_array(np.asarray(chunk.fiber_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "added_sugar_g": _list_array(np.asarray(chunk.added_sugar_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "gi": _list_array(np.asarray(chunk.gi[row_index], dtype=np.float32), lengths, pa.float32()),
            "local_name": _string_list_array(chunk.local_name[row_index], lengths),
            "country": _string_list_array(chunk.country[row_index], lengths),
            "food_region": _string_list_array(chunk.region[row_index], lengths),
            "preparation_state": _string_list_array(chunk.preparation_state[row_index], lengths),
            "portion_g": _list_array(np.asarray(chunk.portion_g[row_index], dtype=np.float32), lengths, pa.float32()),
            "energy_kcal": _list_array(np.asarray(chunk.energy_kcal[row_index], dtype=np.float32), lengths, pa.float32()),
            "carbohydrate_g_100g": _list_array(np.asarray(chunk.carbohydrate_g_100g[row_index], dtype=np.float32), lengths, pa.float32()),
            "protein_g_100g": _list_array(np.asarray(chunk.protein_g_100g[row_index], dtype=np.float32), lengths, pa.float32()),
            "fat_g_100g": _list_array(np.asarray(chunk.fat_g_100g[row_index], dtype=np.float32), lengths, pa.float32()),
            "fibre_g_100g": _list_array(np.asarray(chunk.fibre_g_100g[row_index], dtype=np.float32), lengths, pa.float32()),
            "sodium_mg": _list_array(np.asarray(chunk.sodium_mg[row_index], dtype=np.float32), lengths, pa.float32()),
            "calcium_mg": _list_array(np.asarray(chunk.calcium_mg[row_index], dtype=np.float32), lengths, pa.float32()),
            "iron_mg": _list_array(np.asarray(chunk.iron_mg[row_index], dtype=np.float32), lengths, pa.float32()),
            "vitc_mg": _list_array(np.asarray(chunk.vitc_mg[row_index], dtype=np.float32), lengths, pa.float32()),
            "hours": _list_array(np.asarray(chunk.hours[row_index], dtype=np.float32), lengths, pa.float32()),
            "walk_minutes": _list_array(np.asarray(chunk.walk_minutes[row_index], dtype=np.float32), lengths, pa.float32()),
            "observed": _list_array(np.asarray(chunk.observed[row_index], dtype=bool), lengths, pa.bool_()),
            "day_index": _list_array(np.asarray(chunk.day_index[row_index], dtype=np.int16), lengths, pa.int16()),
            "glucose_2h": _list_array(np.asarray(chunk.glucose_2h[row_index], dtype=np.float32), lengths, pa.float32()),
            "glucose_delta": _list_array(np.asarray(chunk.glucose_delta[row_index], dtype=np.float32), lengths, pa.float32()),
            "estimation_method": pa.array([ESTIMATION_METHOD] * row_index.size, type=pa.string()),
            "quality_flag": pa.array([QUALITY_FLAG] * row_index.size, type=pa.string()),
            "y_class": pa.array(chunk.y_class[row_index], type=pa.int8()),
            "y_value": pa.array(np.asarray(chunk.y_value[row_index], dtype=np.float32), type=pa.float32()),
        }
    )


def _write_split_files(
    chunk: GeneratedChunk,
    output_dir: Path,
    chunk_index: int,
    splits: SplitRatios,
) -> dict[str, int]:
    """Write this chunk straight into each split directory. No global concatenation."""
    assignment = assign_splits(chunk.sequence_id, splits)
    counts = {split: 0 for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        row_index = np.flatnonzero(assignment == split)
        counts[split] = int(row_index.size)
        if row_index.size == 0:
            continue
        destination = output_dir / split / f"part-{chunk_index:05d}.parquet"
        table = _table_for_rows(chunk, row_index)
        pq.write_table(table, destination, compression="zstd")
        del table
    return counts


def write_validated_chunk(
    chunk: GeneratedChunk,
    config: DataConfig,
    output_dir: Path,
    chunk_index: int,
) -> dict[str, int]:
    """Validate the whole chunk, then write its split shards. A failure writes nothing."""
    validate_chunk(chunk, config)
    return _write_split_files(chunk, output_dir, chunk_index, config.splits)


def _generate_and_write_chunk(task: ChunkTask) -> dict[str, int]:
    """Top-level worker. Safe to spawn on Windows because it lives on this module."""
    config = DataConfig.model_validate(task["config"])
    rng = make_chunk_generator(config.seed, task["n_chunks"], task["chunk_index"])
    chunk = generate_chunk(config, start_id=task["start_id"], n=task["n"], rng=rng)
    try:
        counts = write_validated_chunk(chunk, config, Path(task["output_dir"]), task["chunk_index"])
    except ChunkValidationError as exc:
        raise ChunkValidationError(
            f"chunk {task['chunk_index']} failed validation and was not written: {exc}"
        ) from exc
    finally:
        del chunk
    logger.info("wrote chunk %s counts %s", task["chunk_index"], counts)
    return counts


def _chunk_plan(n_sequences: int, chunk_size: int) -> list[tuple[int, int, int]]:
    plan: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < n_sequences:
        size = min(chunk_size, n_sequences - start)
        plan.append((index, start, size))
        start += size
        index += 1
    return plan


def _prepare_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or (resolved / "pyproject.toml").exists():
        raise ValueError(f"refusing to wipe output directory {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    for split in SPLIT_NAMES:
        (resolved / split).mkdir(parents=True, exist_ok=True)


def _write_manifest(output_dir: Path, config: DataConfig, counts: dict[str, int]) -> None:
    payload = {
        "disclaimer": DISCLAIMER,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "counts": {split: int(counts[split]) for split in SPLIT_NAMES},
        "diet_patterns": [
            {
                "id": index,
                "name": pattern.name,
                "plate": pattern.plate,
                "n_items": len(FOOD_CATALOG.pattern_item_ids[index]),
                "items": [food_record(item_id) for item_id in FOOD_CATALOG.pattern_item_ids[index]],
            }
            for index, pattern in enumerate(DIET_PATTERNS)
        ],
        "regions": list(REGION_NAMES),
        "actions": list(ACTION_NAMES),
        "foods": [food_record(index) for index in range(len(FOOD_CATALOG.names))],
        "targets": {
            "y_class": "nutrition adjustment index, see actions",
            "y_value": "simulated mean 2-hour glucose in mg/dL over logged meals",
            "day_index": "day in the patient log, four meals per day, in the same order as the foods",
            "glucose_2h": "simulated 2-hour glucose in mg/dL after each food, in time order",
            "glucose_delta": "glucose_2h minus that patient's fasting glucose, the simulated effect of that food",
            "energy_kcal": "kcal for the eaten serving, 4 kcal/g carbohydrate and protein and 9 kcal/g fat",
            "carbohydrate_g_100g": "carbohydrate per 100 g, from the serving and portion weight",
            "sodium_mg": "simulator estimate of sodium in the eaten serving",
            "calcium_mg": "simulator estimate of calcium in the eaten serving",
            "iron_mg": "simulator estimate of iron in the eaten serving",
            "vitc_mg": "simulator estimate of vitamin C in the eaten serving",
            "estimation_method": ESTIMATION_METHOD,
            "quality_flag": QUALITY_FLAG,
        },
    }
    destination = output_dir / MANIFEST_NAME
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote manifest %s", destination)


def generate_dataset(config: DataConfig) -> dict[str, int]:
    """Generate every chunk into train/validation/test and write the manifest."""
    output_dir = Path(config.output_dir).resolve()
    resolved = DataConfig.model_validate({**config.model_dump(), "output_dir": str(output_dir)})
    _prepare_output(output_dir)
    plan = _chunk_plan(resolved.n_sequences, resolved.chunk_size)
    config_payload = resolved.model_dump(mode="json")
    tasks: list[ChunkTask] = [
        {
            "chunk_index": index,
            "start_id": start_id,
            "n": size,
            "n_chunks": len(plan),
            "config": config_payload,
            "output_dir": str(output_dir),
        }
        for index, start_id, size in plan
    ]
    logger.info(
        "generating %s meal trajectories in %s chunks with %s workers into %s",
        resolved.n_sequences,
        len(tasks),
        resolved.num_workers,
        output_dir,
    )
    if resolved.num_workers == 1:
        results = [_generate_and_write_chunk(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=resolved.num_workers, mp_context=context) as pool:
            results = list(pool.map(_generate_and_write_chunk, tasks))

    counts = {split: 0 for split in SPLIT_NAMES}
    for result in results:
        for split in SPLIT_NAMES:
            counts[split] += int(result[split])
    written = sum(counts.values())
    if written != resolved.n_sequences:
        raise RuntimeError(f"wrote {written} sequences, expected {resolved.n_sequences}")
    _write_manifest(output_dir, resolved, counts)
    logger.info("finished counts %s", counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry: python -m src.generator --config configs/data_config.yaml."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Write simulated South Asian meal-glucose Parquet shards.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    generate_dataset(load_config(args.config))


if __name__ == "__main__":
    main()

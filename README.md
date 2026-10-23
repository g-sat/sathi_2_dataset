# Saathi synthetic meal dataset

This dataset is a simulator for pretraining a model that looks at what someone eats and estimates how that eating pattern sits relative to glucose control. It is built for South Asian meals, with Western meals included only as a contrast. One row is one person. The foods in that row are in time order, and each food has a simulated glucose change.

These are not measured patients, not laboratory food assays, and not medical advice. There are no medicines and no insulin doses. A model trained only on this data cannot give a person a correct glucose plan. Real food logs paired with measured glucose, checked on people the model has not seen, are required before any person-level use.

## How it is tailored

Western diet datasets are a poor match for this problem because the everyday plate is different. This corpus is shaped around that difference in four ways.

**The foods are named dishes, not generic nutrients.** The catalog has 663 individual servings. Each base dish exists as a small, regular, and large serving. Examples include idli, dosa, sambar, steamed rice, millet roti, dal, fish curry, chicken biryani, mithai, sweet chai, and sweet lassi, plus a Western set such as bread, pasta, fries, pizza, burgers, cereal, and soda. Carb, protein, fat, fiber, added sugar, and glycemic index are rounded home-serving estimates for the simulator.

**Each person is assigned one eating pattern, and that pattern only draws from its own food list.** Every list has between 100 and 1000 items. A South Indian rice-and-fish pattern rarely samples pasta. A Western pattern rarely samples idli. The diaspora pattern eats Western foods earlier in the day and rice or roti with dal, curry, or sabzi at dinner.


| python -c "import pandas as pd; from src.foods import FOOD_CATALOG; df = [pd.read](http://pd.read)_parquet(r'data\synthetic\train\part-00000.parquet'); row = df.iloc[0]; out = pd.DataFrame({'day': [row.day](http://row.day)_index, 'food': [FOOD_CATALOG.names[i] for i in [row.food](http://row.food)_ids], 'glucose_2h': row.glucose_2h, 'glucose_delta': row.glucose_delta}); print(out.head(20).to_string()); print('patient', int(row.sequence_id), 'meals', int(row.length))" |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |


Region is sampled with the pattern. The regions are north India, south India, east India, west India, Pakistan, Bangladesh, Sri Lanka, Nepal, and diaspora.

**The glucose response follows the food, the person, and the clock.** Each meal is breakfast, lunch, snack, or dinner, then the next day. A log is 100 to 1000 foods, so a person covers many days. Four meals make one day.

The simulated 2-hour glucose starts from a fasting value. Fasting is higher when BMI is above 23, when age is higher, and when there is a family history. BMI 23 is the simulator's reference point in the World Health Organization Asian risk range. It is not a diagnosis cutoff for a real person.

From that fasting value, the meal moves glucose like this:

- More available carbohydrate and a higher glycemic index raise it.
- Added sugar raises it.
- Protein, fat, fiber, and a walk after the meal pull it down.
- The same meal moves glucose more when the simulated insulin resistance is higher.

`glucose_2h` is the simulated glucose after that food, in mg/dL. `glucose_delta` is that value minus the person's fasting glucose, so it is the effect of that food on that person. `y_value` is the average of the logged meals.

**The label is a nutrition adjustment, not a prescription.** `y_class` is one of five simulator actions for the whole log:


| Id  | Action                                                                             |
| --- | ---------------------------------------------------------------------------------- |
| 0   | Hold the current pattern, when mean simulated 2-hour glucose stays under 140 mg/dL |
| 1   | Move toward a higher-fiber staple                                                  |
| 2   | Add protein and vegetables                                                         |
| 3   | Cut sugary drinks and sweets                                                       |
| 4   | Use a smaller starch portion                                                       |


About 4% of labels are flipped on purpose so the label is not a perfect copy of the rule.

## Who is in the file

The generated corpus has 10,000 people, which sits in the requested band of 5,000 to 15,000. Ages are 28 to 70. BMI is centered near 25.5 and kept between 17 and 38. About 35% have a family-history flag. Sex is coded 0 or 1. Some meals are marked unobserved, more often at the snack slot, so the log is not a perfect diary.

The split is 80% train, 10% validation, and 10% test, by person. The current run wrote 7,961 train, 999 validation, and 1,040 test. People are not shared across splits.

## Where the files are

```text
data/synthetic/
  manifest.json          food catalog, diet patterns, and the settings used
  train/part-*.parquet
  validation/part-*.parquet
  test/part-*.parquet
```

`manifest.json` describes the foods and the diet patterns. The people are in the Parquet files. Those files are binary. Open them with pandas, which is already installed, rather than in the text editor.

Person-level columns:


| Column                                      | Meaning                                                |
| ------------------------------------------- | ------------------------------------------------------ |
| `sequence_id`                               | Person id                                              |
| `diet_pattern`                              | Eating pattern id, 0 through 10                        |
| `region_id`                                 | Region id                                              |
| `age_years`, `sex`, `bmi`, `family_history` | Person description                                     |
| `fasting_glucose`                           | Simulated fasting glucose, mg/dL                       |
| `length`                                    | Number of foods in the log                             |
| `y_class`                                   | Nutrition-adjustment label                             |
| `y_value`                                   | Mean simulated 2-hour glucose over logged meals, mg/dL |
| `estimation_method`                         | `simulator_estimate` for the added nutrition facts      |
| `quality_flag`                              | `not_a_laboratory_assay`                                |


Each of the next columns is a list with one entry per food, in the same order:


| Column                                                      | Meaning                                  |
| ----------------------------------------------------------- | ---------------------------------------- |
| `food_ids`                                                  | Item id in the catalog                   |
| `cuisine`                                                   | 0 South Asian, 1 Western                 |
| `carbs_g`, `protein_g`, `fat_g`, `fiber_g`, `added_sugar_g` | Serving macros                           |
| `local_name`, `country`, `food_region`, `preparation_state` | Dish name, place, and how it is prepared |
| `portion_g`                                                 | Eaten serving weight, grams              |
| `energy_kcal`                                               | Energy for that serving, from the macros |
| `carbohydrate_g_100g`, `protein_g_100g`, `fat_g_100g`, `fibre_g_100g` | The same macros per 100 g        |
| `sodium_mg`, `calcium_mg`, `iron_mg`, `vitc_mg`             | Simulator estimates for that serving     |
| `gi`                                                        | Glycemic index used by the simulator     |
| `hours`                                                     | Clock hour of the meal                   |
| `day_index`                                                 | Day in the log                           |
| `walk_minutes`                                              | Walk after that meal, or 0               |
| `observed`                                                  | Whether this meal is treated as logged   |
| `glucose_2h`                                                | Simulated glucose after that food, mg/dL |
| `glucose_delta`                                             | Rise above that person's fasting glucose |




## Look at one person

From the project folder, with the virtual environment active:

```powershell
python scripts/view_patient.py --id 0
```

That finds patient 0 in `data/synthetic`, prints a short summary, and writes the full timeline to `data/synthetic/views/patient_0.csv`. Open that CSV in Excel. Each row is one food in order: day, meal, macros, and the simulated glucose after that food.

Use another person with `--id 12`. Add `--split train` if you already know which folder they are in.

## Generate it again

Settings live in `configs/data_config.yaml`. The seed, the 10,000 people, the diet-pattern weights, and the glucose coefficients are there.

```powershell
python -m src.generator --config configs/data_config.yaml
```

That rewrites `data/synthetic`. The same seed produces the same people.
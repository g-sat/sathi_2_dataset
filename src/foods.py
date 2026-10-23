"""Individual South Asian and Western food items for a glucose simulator.

Every diet pattern keeps its own list of individual items, and that list is
between 100 and 1000 items. Macro and glycemic-index values are rounded
serving estimates for simulation. They are not laboratory assays and they
are not a prescription.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISCLAIMER = (
    "Simulated South Asian meal and glucose trajectories for model pretraining. "
    "Not a medical device, not individualized advice, and not a substitute for clinical care. "
    "Measured food logs and glucose are required before any person-level use."
)

@dataclass(frozen=True)
class DietPattern:
    """One named plate. ``plate`` is the description written into the manifest."""

    name: str
    plate: str
    foods: dict[str, float]
    south_asian_scale: float
    western_scale: float
    region_probs: tuple[float, ...]


DIET_PATTERNS: tuple[DietPattern, ...] = (
    DietPattern(
        name="south_indian_rice_idli_dosa_fish",
        plate=(
            "Breakfast is plain idli or dosa with sambar. Lunch and dinner are a large "
            "serving of steamed white rice, sambar, and fish curry. Chai is usually taken "
            "with sugar. Bread, pasta, fries, and soda are rare."
        ),
        foods={
            "plain_idli": 6.0, "plain_dosa": 5.0, "sambar": 5.0, "steamed_white_rice": 6.0,
            "fish_curry": 5.0, "cooked_dal": 2.0, "vegetable_sabzi": 2.0, "chai_with_sugar": 3.0,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.05, 0.72, 0.04, 0.03, 0.02, 0.02, 0.06, 0.01, 0.05),
    ),
    DietPattern(
        name="south_indian_vegetarian_rice_sambar_curd",
        plate=(
            "Breakfast is idli or dosa. Lunch and dinner are steamed white rice, sambar, "
            "cooked dal, vegetable sabzi, and plain curd. Fish and chicken are rare. "
            "Chai with sugar is common. Western fast food is rare."
        ),
        foods={
            "plain_idli": 6.0, "plain_dosa": 4.5, "sambar": 5.0, "steamed_white_rice": 6.0,
            "cooked_dal": 4.0, "vegetable_sabzi": 3.5, "plain_curd": 4.0, "chai_with_sugar": 2.5,
            "fish_curry": 0.15, "chicken_curry": 0.15, "chicken_biryani": 0.3,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.04, 0.78, 0.03, 0.04, 0.01, 0.01, 0.03, 0.01, 0.05),
    ),
    DietPattern(
        name="bengali_bangladeshi_rice_fish_and_mishti",
        plate=(
            "The plate is steamed white rice with fish curry. Mithai and chai with sugar "
            "are eaten on ordinary days, not only at festivals. Dal and vegetables are "
            "side dishes. Roti and Western meals are uncommon."
        ),
        foods={
            "steamed_white_rice": 6.5, "fish_curry": 5.5, "mithai": 4.5, "chai_with_sugar": 4.0,
            "cooked_dal": 2.5, "vegetable_sabzi": 2.5, "sweet_lassi": 1.5,
            "whole_wheat_roti": 0.4, "paratha": 0.4,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.03, 0.04, 0.42, 0.02, 0.02, 0.38, 0.02, 0.01, 0.06),
    ),
    DietPattern(
        name="sri_lankan_rice_fish_and_vegetables",
        plate=(
            "Lunch and dinner are steamed rice, fish curry, and cooked vegetables. "
            "Breakfast is idli, dosa, or a smaller rice meal. Sweets and sugary drinks "
            "are less frequent than in the mishti pattern."
        ),
        foods={
            "steamed_white_rice": 6.0, "fish_curry": 5.0, "vegetable_sabzi": 4.0,
            "cooked_dal": 2.5, "plain_idli": 2.5, "plain_dosa": 2.0, "chai_without_sugar": 2.5,
            "mithai": 0.4, "sweet_lassi": 0.3, "chai_with_sugar": 0.8,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.02, 0.10, 0.03, 0.02, 0.01, 0.02, 0.72, 0.01, 0.07),
    ),
    DietPattern(
        name="north_indian_vegetarian_roti_dal_sabzi_curd",
        plate=(
            "Breakfast is roti or paratha. Lunch and dinner are whole-wheat roti, cooked dal, "
            "vegetable sabzi, and plain curd. Paneer appears often. Meat and fish are rare. "
            "Chai with sugar is daily. White rice is a side, not the center of the meal."
        ),
        foods={
            "whole_wheat_roti": 6.5, "millet_roti": 1.5, "paratha": 2.5, "cooked_dal": 5.0,
            "vegetable_sabzi": 5.0, "plain_curd": 4.0, "paneer_curry": 3.0, "chai_with_sugar": 3.5,
            "steamed_white_rice": 0.8, "fish_curry": 0.15, "chicken_curry": 0.2,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.55, 0.03, 0.04, 0.20, 0.04, 0.02, 0.01, 0.04, 0.07),
    ),
    DietPattern(
        name="punjabi_pakistani_paratha_chicken_biryani",
        plate=(
            "Breakfast is paratha. Main meals are chicken curry or chicken biryani, with roti. "
            "Snacks are samosa and sweet lassi. Plain steamed rice is less common than biryani. "
            "Chai with sugar is daily."
        ),
        foods={
            "paratha": 5.5, "whole_wheat_roti": 3.5, "chicken_curry": 5.0, "chicken_biryani": 4.5,
            "samosa": 3.5, "sweet_lassi": 3.0, "chai_with_sugar": 3.5, "cooked_dal": 2.0,
            "steamed_white_rice": 0.7, "plain_idli": 0.2, "plain_dosa": 0.2,
        },
        south_asian_scale=1.0,
        western_scale=0.10,
        region_probs=(0.38, 0.02, 0.03, 0.08, 0.38, 0.02, 0.01, 0.02, 0.06),
    ),
    DietPattern(
        name="west_indian_millet_roti_chana_and_vegetables",
        plate=(
            "The staple is millet roti, not white rice. Meals pair it with chana masala, "
            "cooked dal, and vegetable sabzi. Sweets, biryani, and Western fast food are uncommon."
        ),
        foods={
            "millet_roti": 7.0, "chana_masala": 5.0, "cooked_dal": 4.0, "vegetable_sabzi": 4.5,
            "plain_curd": 2.5, "whole_wheat_roti": 1.5, "steamed_white_rice": 0.45,
            "mithai": 0.35, "chicken_biryani": 0.3, "chai_without_sugar": 2.0,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.12, 0.04, 0.04, 0.62, 0.04, 0.02, 0.01, 0.03, 0.08),
    ),
    DietPattern(
        name="nepali_dal_bhat_rice_lentils_vegetables",
        plate=(
            "The plate is dal bhat: steamed rice, cooked dal, and vegetables, with chicken only "
            "sometimes. Breakfast is poha, banana, or curd. Mithai and soda are not daily foods."
        ),
        foods={
            "steamed_white_rice": 6.0, "cooked_dal": 6.0, "vegetable_sabzi": 4.5, "chicken_curry": 1.6,
            "poha": 2.5, "banana": 2.5, "plain_curd": 2.5, "khichdi": 2.0,
            "mithai": 0.3, "soda": 0.2, "paratha": 0.5,
        },
        south_asian_scale=1.0,
        western_scale=0.08,
        region_probs=(0.12, 0.02, 0.04, 0.03, 0.02, 0.02, 0.01, 0.66, 0.08),
    ),
    DietPattern(
        name="daily_mithai_sweet_chai_and_sweet_lassi",
        plate=(
            "Rice or roti is still the staple, but mithai, chai with sugar, and sweet lassi "
            "are eaten every day. This is the high added-sugar South Asian pattern."
        ),
        foods={
            "mithai": 8.0, "chai_with_sugar": 7.0, "sweet_lassi": 5.5, "steamed_white_rice": 3.0,
            "whole_wheat_roti": 3.0, "samosa": 2.5, "mango": 2.0,
            "chai_without_sugar": 0.15,
        },
        south_asian_scale=1.0,
        western_scale=0.15,
        region_probs=(0.16, 0.16, 0.14, 0.12, 0.12, 0.12, 0.06, 0.06, 0.06),
    ),
    DietPattern(
        name="diaspora_western_day_south_asian_dinner",
        plate=(
            "Breakfast is sweetened cereal, white bread, or oatmeal. Lunch is a burger, pasta, "
            "pizza, or fries. Dinner switches back to rice or roti with dal, curry, or sabzi. "
            "Chai may still be sweet."
        ),
        foods={
            "sweetened_cereal": 4.0, "white_bread": 3.5, "oatmeal": 2.5, "burger": 3.5,
            "white_pasta": 3.0, "pizza": 2.5, "french_fries": 2.5, "steamed_white_rice": 3.5,
            "whole_wheat_roti": 3.0, "cooked_dal": 3.0, "chicken_curry": 2.5, "vegetable_sabzi": 2.5,
            "chai_with_sugar": 2.0,
        },
        south_asian_scale=0.7,
        western_scale=1.6,
        region_probs=(0.04, 0.04, 0.03, 0.03, 0.04, 0.03, 0.02, 0.02, 0.75),
    ),
    DietPattern(
        name="western_bread_pasta_fries_soda",
        plate=(
            "Meals are white bread, pasta, french fries, pizza, burgers, sweetened cereal, "
            "and soda. Rice, roti, dal, idli, dosa, and chai are uncommon."
        ),
        foods={
            "white_bread": 5.0, "white_pasta": 4.5, "french_fries": 4.5, "pizza": 4.0,
            "burger": 4.0, "sweetened_cereal": 3.5, "soda": 5.0, "boiled_potato": 2.0,
            "chocolate_bar": 2.5, "grilled_chicken": 2.0, "green_salad": 1.5,
        },
        south_asian_scale=0.12,
        western_scale=4.0,
        region_probs=(0.06, 0.06, 0.05, 0.05, 0.06, 0.05, 0.04, 0.03, 0.60),
    ),
)
DIET_PATTERN_NAMES: tuple[str, ...] = tuple(pattern.name for pattern in DIET_PATTERNS)
REGION_NAMES: tuple[str, ...] = (
    "north_india",
    "south_india",
    "east_india",
    "west_india",
    "pakistan",
    "bangladesh",
    "sri_lanka",
    "nepal",
    "diaspora",
)
ACTION_NAMES: tuple[str, ...] = (
    "hold_current_pattern",
    "higher_fiber_staple",
    "add_protein_and_vegetable",
    "cut_sugary_drinks_and_sweets",
    "smaller_starch_portion",
)
N_ACTIONS = len(ACTION_NAMES)

MIN_ITEMS = 100
MAX_ITEMS = 1000

_PATTERN_ALLOW: dict[str, tuple[str, ...]] = {
    "south_indian_rice_idli_dosa_fish": ("shared", "south"),
    "south_indian_vegetarian_rice_sambar_curd": ("shared", "south"),
    "bengali_bangladeshi_rice_fish_and_mishti": ("shared", "east", "sweet"),
    "sri_lankan_rice_fish_and_vegetables": ("shared", "sri", "south"),
    "north_indian_vegetarian_roti_dal_sabzi_curd": ("shared", "north"),
    "punjabi_pakistani_paratha_chicken_biryani": ("shared", "north", "pak"),
    "west_indian_millet_roti_chana_and_vegetables": ("shared", "west"),
    "nepali_dal_bhat_rice_lentils_vegetables": ("shared", "nepal"),
    "daily_mithai_sweet_chai_and_sweet_lassi": ("shared", "sweet", "north", "south"),
    "diaspora_western_day_south_asian_dinner": ("shared", "western", "south", "north"),
    "western_bread_pasta_fries_soda": ("western",),
}

_PORTIONS: tuple[tuple[str, float], ...] = (
    ("small", 0.70),
    ("regular", 1.00),
    ("large", 1.40),
)

# name|cuisine|carbs|protein|fat|fiber|added_sugar|gi|tags
# cuisine 0 is South Asian and 1 is Western. Tags select which diet pattern can use the item.
_BASE_TABLE = """
steamed_white_rice|0|45|4|0.4|0.6|0|73|shared,main
cooked_brown_rice|0|45|5|1.8|3.5|0|50|shared,main
millet_roti|0|28|5|2|4.5|0|52|shared,west,main
whole_wheat_roti|0|30|6|4|4|1|62|shared,main
chapati|0|28|5|3|3.5|0|62|shared,main
plain_curd|0|6|6|4|0|0|28|shared,breakfast,snack
banana|0|27|1|0.3|3|0|51|shared,breakfast,snack
mango|0|25|1|0.5|3|0|51|shared,snack
papaya|0|12|0.6|0.2|2|0|60|shared,breakfast
apple|0|19|0.4|0.2|3|0|36|shared,snack
orange|0|15|1|0.2|3|0|42|shared,snack
dates|0|18|0.4|0.1|2|0|70|shared,snack
pomegranate|0|16|1|0.5|3|0|35|shared,snack
guava|0|14|1|0.4|5|0|29|shared,snack
boiled_egg|0|1|6|5|0|0|0|shared,breakfast
chai_with_sugar|0|14|1|1.5|0|12|70|shared,breakfast,snack,sweet
chai_without_sugar|0|3|1|1.5|0|0|30|shared,breakfast,snack
milk|0|12|8|5|0|0|30|shared,breakfast
cooked_dal|0|22|10|5|7|1|32|shared,main
vegetable_sabzi|0|12|3|6|4|1|35|shared,main
cucumber_raita|0|6|3|3|0.5|1|25|shared,snack
buttermilk|0|5|3|1|0|1|30|shared,snack
roasted_chana|0|18|7|3|6|0|28|shared,snack
peanuts|0|6|7|14|2|0|14|shared,snack
poha|0|36|4|8|2|1|65|shared,breakfast
upma|0|38|5|8|2|1|68|shared,breakfast
khichdi|0|36|8|6|3.5|1|52|shared,main
aloo|0|30|3|0.2|3|0|78|shared,main,snack
salted_lassi|0|12|6|3|0|2|35|shared,breakfast
mithai|0|42|3|12|0.5|28|78|shared,snack,sweet
plain_idli|0|28|4|1|1.2|0|60|south,breakfast
rava_idli|0|32|4|3|1|0|66|south,breakfast
plain_dosa|0|30|4|6|1.2|0|66|south,breakfast
masala_dosa|0|42|6|12|2.5|1|70|south,breakfast
rava_dosa|0|36|4|10|1|0|72|south,breakfast
uttapam|0|34|5|8|2|0|64|south,breakfast
medu_vada|0|22|4|12|2|0|70|south,snack
sambar|0|16|6|4|4|1|42|south,main
rasam|0|8|2|2|1|0|35|south,main
coconut_chutney|0|6|2|8|2|0|40|south,breakfast
lemon_rice|0|48|4|8|1|0|70|south,main
curd_rice|0|42|6|6|0.8|1|62|south,main
tamarind_rice|0|50|4|8|1.5|2|72|south,main
bisibelebath|0|48|8|8|4|1|65|south,main
avial|0|14|3|8|4|0|40|south,main
beans_thoran|0|12|3|7|4|0|35|south,main
veg_kurma|0|16|4|10|3|1|42|south,main
appam|0|28|3|4|1|1|68|south,breakfast
puttu|0|36|4|4|3|0|65|south,breakfast
idiyappam|0|34|3|2|1|0|70|south,breakfast
ven_pongal|0|40|8|8|2|0|62|south,breakfast
tomato_rice|0|48|4|7|1.5|1|70|south,main
coconut_rice|0|46|4|10|1|0|68|south,main
fish_curry|0|6|20|10|1|1|28|south,fish,main
meen_moilee|0|6|18|12|1|0|30|south,fish,main
prawn_curry|0|5|18|8|0.5|1|28|south,fish,main
chicken_chettinad|0|8|22|14|1|1|32|south,meat,main
filter_coffee_sweet|0|16|2|2|0|14|70|south,sweet,breakfast
payasam|0|40|4|8|0.5|22|75|south,sweet,snack
mysore_pak|0|36|3|16|0|24|72|south,sweet,snack
macher_jhol|0|6|18|8|1|1|30|east,fish,main
shorshe_ilish|0|4|20|14|0.5|0|28|east,fish,main
chingri_malai|0|8|16|12|0.5|1|32|east,fish,main
chicken_kosha|0|8|22|16|1|1|34|east,meat,main
luchi|0|32|4|14|1|1|75|east,breakfast
aloo_posto|0|28|4|10|3|0|70|east,main
cholar_dal|0|24|9|8|6|2|32|east,main
shukto|0|12|3|6|3|1|38|east,main
begun_bhaja|0|14|2|12|3|0|55|east,snack
dim_er_dalna|0|8|12|10|1|1|35|east,main
bhat_dal|0|46|8|4|2|0|68|east,main
muri_mixture|0|28|4|10|2|2|72|east,snack
mishti_doi|0|24|5|6|0|16|60|east,sweet,snack
rosogolla|0|32|4|4|0|22|72|east,sweet,snack
sandesh|0|28|6|8|0|18|65|east,sweet,snack
nolen_gur_payesh|0|38|5|8|0|20|70|east,sweet,snack
pantua|0|34|4|14|0|22|74|east,sweet,snack
doi_bhat|0|40|6|5|0.5|4|64|east,main
rice_and_curry|0|48|8|8|3|0|68|sri,main
hopper|0|30|4|5|1|1|70|sri,breakfast
string_hoppers|0|34|3|2|1|0|72|sri,breakfast
kottu|0|46|12|14|2|1|70|sri,main
fish_ambul_thiyal|0|6|20|8|1|0|28|sri,fish,main
pol_sambol|0|8|2|10|3|0|40|sri,main
dhal_curry|0|20|9|6|5|0|34|sri,main
coconut_sambol|0|8|2|12|3|0|38|sri,main
sri_chicken_curry|0|8|22|12|1|1|32|sri,meat,main
gotu_kola_salad|0|6|2|4|3|0|20|sri,main
wambatu_moju|0|16|2|8|3|4|55|sri,snack
kiribath|0|42|4|6|0.5|2|72|sri,breakfast
woodapple_cream|0|28|2|4|2|16|58|sri,sweet,snack
fish_cutlets|0|18|10|12|1|1|55|sri,fish,snack
parippu|0|20|9|5|5|0|32|sri,main
mallung|0|8|2|5|3|0|25|sri,main
coconut_roti|0|32|4|8|3|2|66|sri,breakfast
paratha|0|32|5|12|3|1|70|north,breakfast,main
missi_roti|0|26|6|4|4|0|55|north,main
bajra_roti|0|28|5|3|4|0|52|north,main
jowar_roti|0|28|4|2|4|0|50|north,main
makki_roti|0|30|4|3|3|0|58|north,main
aloo_paratha|0|40|6|14|3|1|72|north,breakfast
gobi_paratha|0|36|6|13|4|1|68|north,breakfast
puri|0|32|4|14|1|1|75|north,breakfast
chole|0|30|10|8|8|2|32|north,main
rajma|0|28|10|6|8|1|30|north,main
dal_tadka|0|22|9|6|5|1|34|north,main
dal_makhani|0|24|10|12|6|1|36|north,main
palak_paneer|0|10|14|16|3|1|28|north,main
paneer_butter_masala|0|12|14|18|1|2|32|north,main
paneer_curry|0|8|14|16|2|1|30|north,main
chana_masala|0|32|11|8|9|2|28|north,west,main
aloo_gobi|0|22|4|8|4|1|55|north,main
bhindi_masala|0|12|3|7|4|1|35|north,main
baingan_bharta|0|14|3|8|4|1|40|north,main
matar_paneer|0|16|12|14|4|1|35|north,main
jeera_rice|0|46|4|6|0.8|0|70|north,main
veg_pulao|0|48|6|8|2|1|68|north,main
samosa|0|28|4|16|2|1|75|north,snack
pakora|0|18|4|14|2|0|68|north,snack
gulab_jamun|0|40|3|12|0|26|78|north,sweet,snack
jalebi|0|42|2|12|0|28|80|north,sweet,snack
gajar_halwa|0|32|4|10|3|18|65|north,sweet,snack
kheer|0|36|5|8|0|20|70|north,sweet,snack
besan_laddu|0|34|5|12|2|22|68|north,sweet,snack
boondi_raita|0|14|5|6|1|2|40|north,snack
thepla|0|28|5|8|3|1|58|west,breakfast
dhokla|0|18|5|4|2|1|45|west,snack
khaman|0|16|4|3|1|2|48|west,snack
handvo|0|26|6|8|3|1|55|west,breakfast
pav_bhaji|0|48|8|14|5|3|68|west,main
vada_pav|0|42|6|16|3|2|75|west,snack
misal_pav|0|46|12|14|6|2|66|west,main
puran_poli|0|44|6|10|3|18|72|west,sweet,snack
undhiyu|0|24|6|10|6|1|42|west,main
bajri_no_rotlo|0|26|5|3|4|0|50|west,main
sev_tameta|0|28|5|10|3|2|60|west,main
patra|0|22|4|8|3|2|58|west,snack
shrikhand|0|28|6|8|0|20|62|west,sweet,snack
fafda|0|24|4|14|2|1|70|west,snack
dal_dhokli|0|30|8|6|4|1|48|west,main
khichu|0|32|4|4|1|0|68|west,breakfast
ringan_bharta|0|14|3|8|4|1|40|west,main
naan|0|38|7|6|2|2|72|pak,main
roghni_naan|0|42|7|10|2|3|74|pak,main
chicken_karahi|0|6|24|14|1|1|28|pak,meat,main
mutton_karahi|0|4|24|16|0.5|0|26|pak,meat,main
nihari|0|6|22|14|1|1|30|pak,meat,main
haleem|0|24|16|10|3|1|42|pak,meat,main
seekh_kebab|0|4|20|14|0|0|20|pak,meat,main
chicken_biryani|0|55|16|14|1.5|2|68|pak,meat,main
beef_biryani|0|54|18|16|1.5|2|66|pak,meat,main
chicken_handi|0|8|22|14|1|1|32|pak,meat,main
aloo_keema|0|18|16|14|2|1|45|pak,meat,main
sweet_lassi|0|28|6|4|0|18|65|pak,sweet,snack
chana_chaat|0|26|8|6|6|3|40|pak,snack
fruit_chaat|0|22|1|1|3|2|52|pak,snack
rabri|0|24|6|8|0|18|60|pak,sweet,snack
chicken_curry|0|8|22|12|1|1|30|north,pak,meat,main
dal_bhat|0|48|10|4|3|0|64|nepal,main
tarkari|0|14|3|6|4|0|38|nepal,main
nepali_chicken|0|8|20|10|1|0|30|nepal,meat,main
veg_momo|0|28|6|6|2|1|60|nepal,snack
chicken_momo|0|26|12|8|1|1|55|nepal,meat,snack
sel_roti|0|36|3|8|1|12|75|nepal,sweet,breakfast
gundruk_soup|0|8|3|2|3|0|30|nepal,main
chiura_curry|0|44|6|6|2|0|72|nepal,main
aloo_tama|0|26|4|6|4|0|58|nepal,main
black_tea|0|2|0|0|0|0|15|nepal,breakfast
saag|0|8|3|4|3|0|25|nepal,main
kwati|0|24|10|4|6|0|34|nepal,main
bara|0|24|6|8|2|0|58|nepal,snack
chatamari|0|30|5|6|1|1|66|nepal,snack
yomari|0|36|4|6|1|14|72|nepal,sweet,snack
dhido|0|40|4|2|3|0|70|nepal,main
white_bread|1|28|4|2|1|2|75|western,breakfast
whole_wheat_bread|1|24|6|2|4|2|50|western,breakfast
white_pasta|1|43|8|1.5|2|1|49|western,main
french_fries|1|40|4|17|4|0|75|western,snack
burger|1|40|20|22|2|6|66|western,main
pizza|1|52|16|18|3|4|60|western,main
sweetened_cereal|1|40|4|2|2|16|80|western,sweet,breakfast
soda|1|39|0|0|0|39|63|western,sweet,snack
oatmeal|1|27|5|3|4|0|55|western,breakfast
green_salad|1|10|3|8|3|0|15|western,main
grilled_chicken|1|0|31|8|0|0|0|western,main
chocolate_bar|1|32|3|15|1|26|42|western,sweet,snack
mashed_potato|1|34|4|6|3|1|78|western,main
hash_brown|1|28|3|12|2|0|76|western,breakfast
bagel|1|48|9|2|2|6|72|western,breakfast
croissant|1|32|5|16|1|4|67|western,breakfast
pancake_with_syrup|1|48|6|10|1|22|76|western,sweet,breakfast
bacon_and_eggs|1|2|18|20|0|0|0|western,breakfast
ham_sandwich|1|36|16|12|2|3|62|western,main
cheese_sandwich|1|34|14|16|2|3|60|western,main
caesar_salad|1|14|12|16|2|2|40|western,main
fried_chicken|1|12|24|18|0|1|45|western,main
fish_and_chips|1|48|18|20|3|1|70|western,main
mac_and_cheese|1|46|14|18|2|3|64|western,main
hot_dog|1|32|12|18|1|4|62|western,main
donut|1|36|4|16|1|18|76|western,sweet,snack
cookie|1|28|3|12|1|16|62|western,sweet,snack
ice_cream|1|26|4|12|0|20|60|western,sweet,snack
potato_chips|1|20|2|12|1|1|70|western,snack
energy_drink|1|28|0|0|0|27|70|western,sweet,snack
orange_juice|1|26|1|0|0|10|50|western,sweet,breakfast
latte_with_sugar|1|18|8|6|0|14|45|western,sweet,breakfast
black_coffee|1|1|0|0|0|0|0|western,breakfast
greek_yogurt|1|8|15|5|0|4|20|western,breakfast
protein_bar|1|20|15|8|3|8|40|western,snack
quinoa_bowl|1|32|8|6|5|1|53|western,main
brown_rice_bowl|1|44|6|4|3|0|50|western,main
sushi_roll|1|38|10|4|1|2|55|western,main
taco|1|30|14|12|4|2|58|western,main
burrito|1|55|16|16|6|2|60|western,main
garlic_bread|1|28|5|10|1|2|72|western,snack
scrambled_eggs|1|2|12|10|0|0|0|western,breakfast
granola|1|40|6|8|4|12|65|western,breakfast
muffin|1|42|5|14|1|18|72|western,sweet,snack
chicken_wrap|1|36|22|12|3|2|55|western,main
tomato_soup|1|16|3|4|2|4|38|western,main
boiled_potato|1|37|4|0.2|4|0|78|western,main,snack
"""


@dataclass(frozen=True)
class _Item:
    name: str
    base: str
    cuisine: int
    carbs: float
    protein: float
    fat: float
    fiber: float
    sugar: float
    gi: float
    tags: tuple[str, ...]


def _parse_bases(
    table: str,
) -> tuple[tuple[str, int, float, float, float, float, float, float, tuple[str, ...]], ...]:
    parsed: list[tuple[str, int, float, float, float, float, float, float, tuple[str, ...]]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(table.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 9:
            raise ValueError(f"food table line {line_number} does not have 9 fields")
        name, cuisine, carbs, protein, fat, fiber, sugar, gi, tags = parts
        if name in seen:
            raise ValueError(f"duplicate food {name}")
        seen.add(name)
        parsed.append(
            (
                name,
                int(cuisine),
                float(carbs),
                float(protein),
                float(fat),
                float(fiber),
                float(sugar),
                float(gi),
                tuple(tag for tag in tags.split(",") if tag),
            )
        )
    return tuple(parsed)


def _expand_items(
    bases: tuple[tuple[str, int, float, float, float, float, float, float, tuple[str, ...]], ...],
) -> tuple[_Item, ...]:
    items: list[_Item] = []
    for name, cuisine, carbs, protein, fat, fiber, sugar, gi, tags in bases:
        for suffix, scale in _PORTIONS:
            item_name = name if suffix == "regular" else f"{name}_{suffix}"
            items.append(
                _Item(
                    name=item_name,
                    base=name,
                    cuisine=cuisine,
                    carbs=round(carbs * scale, 1),
                    protein=round(protein * scale, 1),
                    fat=round(fat * scale, 1),
                    fiber=round(fiber * scale, 1),
                    sugar=round(sugar * scale, 1),
                    gi=gi,
                    tags=tags,
                )
            )
    return tuple(items)


def _item_masks(items: tuple[_Item, ...]) -> list[np.ndarray]:
    missing = [name for name in DIET_PATTERN_NAMES if name not in _PATTERN_ALLOW]
    extra = [name for name in _PATTERN_ALLOW if name not in DIET_PATTERN_NAMES]
    if missing or extra:
        raise ValueError(f"pattern item rules do not match diet patterns: missing={missing} extra={extra}")
    masks: list[np.ndarray] = []
    problems: list[str] = []
    for pattern in DIET_PATTERNS:
        allow = set(_PATTERN_ALLOW[pattern.name])
        highlighted = set(pattern.foods)
        mask = np.array(
            [bool(set(item.tags) & allow or item.base in highlighted) for item in items],
            dtype=bool,
        )
        count = int(mask.sum())
        if count < MIN_ITEMS or count > MAX_ITEMS:
            problems.append(f"{pattern.name}={count}")
        masks.append(mask)
    total = len(items)
    if problems or not MIN_ITEMS <= total <= MAX_ITEMS:
        raise ValueError(
            f"individual item counts must be {MIN_ITEMS} to {MAX_ITEMS}; "
            f"catalog={total}; off pattern lists: {problems or 'none'}"
        )
    return masks


@dataclass(frozen=True)
class FoodCatalog:
    names: tuple[str, ...]
    cuisine: np.ndarray
    carbs: np.ndarray
    protein: np.ndarray
    fat: np.ndarray
    fiber: np.ndarray
    added_sugar: np.ndarray
    gi: np.ndarray
    diet_slot_probs: np.ndarray
    region_probs: np.ndarray
    pattern_item_ids: tuple[tuple[int, ...], ...]


def _weight_table(items: tuple[_Item, ...], masks: list[np.ndarray]) -> np.ndarray:
    by_base: dict[str, list[int]] = {}
    for position, item in enumerate(items):
        by_base.setdefault(item.base, []).append(position)
    cuisine = np.array([item.cuisine for item in items], dtype=np.int16)
    n_foods = len(items)
    weights = np.ones((len(DIET_PATTERNS), 4, n_foods), dtype=np.float64)
    south_asian = cuisine == 0
    western = cuisine == 1
    for pattern_id, pattern in enumerate(DIET_PATTERNS):
        weights[pattern_id, :, ~masks[pattern_id]] = 0.0
        weights[pattern_id, :, south_asian] *= pattern.south_asian_scale
        weights[pattern_id, :, western] *= pattern.western_scale
        for name, value in pattern.foods.items():
            positions = by_base.get(name)
            if positions is None:
                raise KeyError(f"{pattern.name} lists unknown food {name}")
            for position in positions:
                weights[pattern_id, :, position] *= value
        for position, item in enumerate(items):
            if "breakfast" in item.tags:
                weights[pattern_id, 0, position] *= 2.2
            if "main" in item.tags:
                weights[pattern_id, 1, position] *= 2.0
                weights[pattern_id, 3, position] *= 2.0
            if "snack" in item.tags:
                weights[pattern_id, 2, position] *= 2.4
    diaspora_id = DIET_PATTERN_NAMES.index("diaspora_western_day_south_asian_dinner")
    weights[diaspora_id, 0, western] *= 3.5
    weights[diaspora_id, 1, western] *= 3.5
    weights[diaspora_id, 0, south_asian] *= 0.35
    weights[diaspora_id, 1, south_asian] *= 0.45
    weights[diaspora_id, 3, south_asian] *= 3.0
    weights[diaspora_id, 3, western] *= 0.4
    slot_sums = weights.sum(axis=2)
    if np.any(slot_sums <= 0.0):
        raise ValueError("a diet pattern has a meal slot with no individual items")
    weights /= slot_sums[:, :, None]
    return weights


def load_food_catalog() -> FoodCatalog:
    items = _expand_items(_parse_bases(_BASE_TABLE))
    masks = _item_masks(items)
    region_probs = np.array([pattern.region_probs for pattern in DIET_PATTERNS], dtype=np.float64)
    if region_probs.shape != (len(DIET_PATTERNS), len(REGION_NAMES)):
        raise ValueError("each diet pattern needs one region probability per region")
    region_probs /= region_probs.sum(axis=1, keepdims=True)
    return FoodCatalog(
        names=tuple(item.name for item in items),
        cuisine=np.array([item.cuisine for item in items], dtype=np.int16),
        carbs=np.array([item.carbs for item in items], dtype=np.float64),
        protein=np.array([item.protein for item in items], dtype=np.float64),
        fat=np.array([item.fat for item in items], dtype=np.float64),
        fiber=np.array([item.fiber for item in items], dtype=np.float64),
        added_sugar=np.array([item.sugar for item in items], dtype=np.float64),
        gi=np.array([item.gi for item in items], dtype=np.float64),
        diet_slot_probs=_weight_table(items, masks),
        region_probs=region_probs,
        pattern_item_ids=tuple(tuple(int(index) for index in np.flatnonzero(mask)) for mask in masks),
    )


def _item_text(name: str, cuisine_id: int) -> str:
    portion = "regular"
    base = name
    if name.endswith("_small"):
        portion = "small"
        base = name[: -len("_small")]
    elif name.endswith("_large"):
        portion = "large"
        base = name[: -len("_large")]
    place = "South Asian" if cuisine_id == 0 else "Western"
    return f"{portion} serving of {base.replace('_', ' ')}, {place}"


def food_record(index: int) -> dict[str, object]:
    """One individual food item, with the serving used by the simulator."""
    name = FOOD_CATALOG.names[index]
    cuisine_id = int(FOOD_CATALOG.cuisine[index])
    return {
        "id": index,
        "name": name,
        "item": _item_text(name, cuisine_id),
        "cuisine": "south_asian" if cuisine_id == 0 else "western",
        "carbs_g": float(FOOD_CATALOG.carbs[index]),
        "protein_g": float(FOOD_CATALOG.protein[index]),
        "fat_g": float(FOOD_CATALOG.fat[index]),
        "fiber_g": float(FOOD_CATALOG.fiber[index]),
        "added_sugar_g": float(FOOD_CATALOG.added_sugar[index]),
        "gi": float(FOOD_CATALOG.gi[index]),
    }


FOOD_CATALOG = load_food_catalog()
N_FOODS = len(FOOD_CATALOG.names)

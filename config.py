import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_FILES = [
    os.path.join(BASE_DIR, "food-nutrition-dataset.zip"),
    os.path.join(BASE_DIR, "FoodData_Central_foundation_food_csv_2025-12-18.zip"),
    os.path.join(BASE_DIR, "FoodData_Central_sr_legacy_food_csv_2018-04.zip"),
]

NUTRITION_MATRIX_XLSX = os.path.join(BASE_DIR, "Nutrition_Matrix.xlsx")
PROJECT_DATA_XLSX     = os.path.join(BASE_DIR, "Project_Data.xlsx")
DESIGN_POINTS_XLSX    = os.path.join(BASE_DIR, "Design_Points.xlsx")

INGREDIENTS = ["SPI", "Gluten", "Pea", "Oil", "Binder"]
D = len(INGREDIENTS)

PROTEIN_IDX = [0, 1, 2]
OIL_IDX     = 3
BINDER_IDX  = 4

SEARCH_GROUPS = {
    "SPI":    r"soy.*protein.*isolate",
    "Gluten": r"wheat.*gluten|vital.*gluten",
    "Pea":    r"Peas,\s+green,\s+split,\s+mature\s+seeds,\s+raw",
    "Oil":    r"(?i)oil\s*,\s*coconut|coconut\s+oil",
    "Binder": r"Seaweed,\s+agar,\s+dried",
}

NUTRIENT_MAP = {1003: "Protein (g)", 1008: "Energy (kcal)", 1004: "Fat (g)", 1005: "Carb (g)"}

THB_PER_USD = 33.0
USD_PER_EUR = 1.08

COST_USD_PER_KG = {
    "SPI":    4.63,
    "Gluten": 2.02,
    "Pea":    3.31,
    "Oil":    1.52,
    "Binder": 10.80,
}

def cost_baht_per_100g():

    return {k: v * 0.1 * THB_PER_USD for k, v in COST_USD_PER_KG.items()}

PROTEIN_SUM_MIN, PROTEIN_SUM_MAX = 0.05, 0.88

INGREDIENT_BOUNDS = {
    "SPI":    (0.00, 0.88),
    "Gluten": (0.05, 0.88),
    "Pea":    (0.00, 0.88),
    "Oil":    (0.05, 0.35),
    "Binder": (0.02, 0.15),
}

OIL_MIN,    OIL_MAX    = INGREDIENT_BOUNDS["Oil"]
BINDER_MIN, BINDER_MAX = INGREDIENT_BOUNDS["Binder"]

PROCESS_LOSS = {
    "Protein (g)":   0.10,
    "Fat (g)":       0.15,
    "Carb (g)":      0.05,
    "Energy (kcal)": 0.00,
}
KJ_PER_KCAL = 4.184

WATER_FRACTION = 0.0

PRODUCT_GROUPS = {
    "Burger (เบอร์เกอร์)":       {"Energy": (355, 1160), "Protein": (2.9, 20.9), "Fat": (0.5, 17.7), "Carb": (3.9, 33.0)},
    "Ground meat (เนื้อบดเทียม)": {"Energy": (312,  950), "Protein": (4.6, 23.4), "Fat": (0.5, 15.0), "Carb": (1.6, 27.0)},
    "Sausage (ไส้กรอก)":         {"Energy": (458, 1103), "Protein": (2.8, 23.0), "Fat": (1.0, 19.0), "Carb": (0.5, 24.4)},
    "Chicken (ไก่เทียม)":        {"Energy": (274, 1130), "Protein": (1.2, 36.1), "Fat": (2.0, 14.3), "Carb": (2.7, 21.5)},
    "Seafood (อาหารทะเลเทียม)":  {"Energy": (231, 1178), "Protein": (0.3, 14.0), "Fat": (0.5, 12.6), "Carb": (4.4, 30.0)},
    "Other (อื่น ๆ)":           {"Energy": (405, 1180), "Protein": (5.7, 26.4), "Fat": (1.0, 23.2), "Carb": (1.3, 26.7)},
}

COST_T = 0.0
COST_U = 30.0

DF_WEIGHTS = {"Energy": 1.0, "Protein": 1.0, "Fat": 1.0, "Carb": 1.0, "Cost": 1.0}

GP_XI     = 10.0
GP_NUGGET = 1e-8

DEFAULT_N_POINTS = 30
RANDOM_SEED      = 42
MAXENT_CANDIDATE_POOL = 2000
MORRIS_MITCHELL_LAMBDA = 40.0
MORRIS_MITCHELL_P      = 1.0

MAX_POINTS            = 500
GPR_MAX_N             = 800

MAXENT_MAX_N          = 500

MAXIMIN_REFINE_MAX_N  = 400

DESIGN_METHODS = [
    "Latin Hypercube (Rejection)",
    "Latin Hypercube (Transformation)",
    "Simplex Lattice",
    "Simplex Centroid",
    "Maximin (Morris-Mitchell)",
    "Maximum Entropy",
    "Dirichlet",
]

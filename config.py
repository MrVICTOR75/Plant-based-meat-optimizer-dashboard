"""
================================================================================
 config.py  --  central configuration (ค่าคงที่กลางของทั้งโปรเจกต์)
--------------------------------------------------------------------------------
 Project: Optimization of Plant-Based Meat Formulations
          (การหาค่าเหมาะที่สุดของสูตรผลิตภัณฑ์เนื้อจากพืช)
 ทุกโมดูล (m1..m4 และ main_inpho) import ค่าจากไฟล์นี้ เพื่อให้แก้ที่เดียวจบ
================================================================================
"""

import os

################################################################################
# 1) PATHS
#    ยึด "โฟลเดอร์ที่วางโค้ดนี้" (BASE_DIR) เป็นหลัก จึงย้ายเครื่อง/โฟลเดอร์ได้
#    โดยไม่ต้องแก้ path ; ไฟล์ .xlsx จะถูกเซฟข้าง ๆ โค้ดเสมอ
################################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_FILES = [
    os.path.join(BASE_DIR, "food-nutrition-dataset.zip"),
    os.path.join(BASE_DIR, "FoodData_Central_foundation_food_csv_2025-12-18.zip"),
    os.path.join(BASE_DIR, "FoodData_Central_sr_legacy_food_csv_2018-04.zip"),
]

NUTRITION_MATRIX_XLSX = os.path.join(BASE_DIR, "Nutrition_Matrix.xlsx")
PROJECT_DATA_XLSX     = os.path.join(BASE_DIR, "Project_Data.xlsx")
DESIGN_POINTS_XLSX    = os.path.join(BASE_DIR, "Design_Points.xlsx")

################################################################################
# 2) INGREDIENTS  --  ลำดับวัตถุดิบ 5 ชนิด (ใช้ลำดับนี้ทั้งโปรเจกต์ ห้ามสลับ)
################################################################################
INGREDIENTS = ["SPI", "Gluten", "Pea", "Oil", "Binder"]
D = len(INGREDIENTS)                       # d = 5 (simplex S^4)

PROTEIN_IDX = [0, 1, 2]                     # SPI, Gluten, Pea
OIL_IDX     = 3
BINDER_IDX  = 4

# regex ค้นหาชื่ออาหาร (Module 1) : Oil = coconut oil, Binder = agar (โภชนาการ)
SEARCH_GROUPS = {
    "SPI":    r"soy.*protein.*isolate",
    "Gluten": r"wheat.*gluten|vital.*gluten",
    "Pea":    r"Peas,\s+green,\s+split,\s+mature\s+seeds,\s+raw",
    "Oil":    r"(?i)oil\s*,\s*coconut|coconut\s+oil",
    "Binder": r"Seaweed,\s+agar,\s+dried",
}

NUTRIENT_MAP = {1003: "Protein (g)", 1008: "Energy (kcal)", 1004: "Fat (g)", 1005: "Carb (g)"}

################################################################################
# 3) COST  --  ต้นทุน (จาก Ingrediant_cost.png : ถูกที่สุดต่อชนิด) เป็น USD/kg
#    แปลงเป็น "บาท/100 g" ในโค้ด (โปร่งใส)
#    * Oil : ตารางมีเฉพาะ soybean oil ($1.45/kg) จึงใช้เป็น proxy ของ coconut
################################################################################
THB_PER_USD = 33.0          # อัตราแลกเปลี่ยน (ปรับได้)
USD_PER_EUR = 1.08

COST_USD_PER_KG = {
    "SPI":    4.63,   # China $2.10/lb -> 4.63 USD/kg (ถูกสุดใน 3 แหล่ง)
    "Gluten": 1.403,  # Europe proxy $1,403.33/MT = 1.403 USD/kg
    "Pea":    1.36,   # Global cowpea low $1.36/kg
    "Oil":    1.45,   # Soybean oil refined $1.45/kg (proxy สำหรับ coconut)
    "Binder": 8.80,   # Methylcellulose food/pharma grade low $8.80/kg
}

def cost_baht_per_100g():
    """บาท/100g = USD/kg x 0.1 x THB_PER_USD"""
    return {k: v * 0.1 * THB_PER_USD for k, v in COST_USD_PER_KG.items()}

################################################################################
# 4) CONSTRAINTS  --  bounded polytope บน simplex
#    Sum(x)=1, x>=0  บวก:
#      group : 0.05 <= xSPI+xGluten+xPea <= 0.88   (ผลรวมโปรตีน)
#      per-ingredient bounds (min, max) ต่อชนิด (ดู INGREDIENT_BOUNDS)
#
#    * เหตุที่ตั้ง 'ขอบล่าง' ให้ Gluten และ Binder:
#      DF ให้รางวัลเฉพาะ 'โภชนาการ + ต้นทุน' ไม่รู้ว่ากลูเตน (โครงสร้างเส้นใย) และ
#      binder (ยึดเกาะ) 'จำเป็นเชิงหน้าที่' -> optimizer จึงดันเป็น 0 (เพราะแพง/ถ่วง)
#      แก้ด้วยการบังคับขอบล่าง (Cornell L-pseudocomponents) ให้ต้องใส่เสมอ
#      ค่าเหล่านี้ปรับได้สด ๆ ผ่าน slider ใน dashboard
################################################################################
PROTEIN_SUM_MIN, PROTEIN_SUM_MAX = 0.05, 0.88

# (min, max) ต่อวัตถุดิบ ; ต้อง sum(min) <= 1 และ sum(max) >= 1
# อ้างอิงค่า default ขอบล่าง:
#  - Gluten >= 5% : wheat gluten เป็น texturizer หลัก งานวิจัยใช้ 15-40% (บางสูตร
#    ถึง 40% ให้เส้นใยดีสุด) ; 5% จึงเป็น "พื้นขั้นต่ำ" ที่ต่ำกว่าระดับใช้งานทั่วไป
#    (Zhang et al. 2023, LWT; Chiang et al. 2021, J. Food Eng.)
#  - Binder(Methylcellulose) >= 2% : เชิงพาณิชย์ (Beyond/Impossible) ใช้ <2% (~2 g/
#    ชิ้น) ทั่วไป 0.2-2% ; งานวิจัยใช้ 2-3% (Bakhsh et al. 2021, Food Sci. Anim.
#    Resour. 41:983) ; GFI formulation guide
INGREDIENT_BOUNDS = {
    "SPI":    (0.00, 0.88),
    "Gluten": (0.05, 0.88),   # บังคับกลูเตน >= 5% (โครงสร้างเส้นใย)
    "Pea":    (0.00, 0.88),
    "Oil":    (0.05, 0.35),
    "Binder": (0.02, 0.15),   # บังคับ binder >= 2% (สารยึดเกาะ)
}

# (เก็บไว้เผื่อ backward-compat) -- อ่านจาก INGREDIENT_BOUNDS
OIL_MIN,    OIL_MAX    = INGREDIENT_BOUNDS["Oil"]
BINDER_MIN, BINDER_MAX = INGREDIENT_BOUNDS["Binder"]

################################################################################
# 5) PROCESSING LOSS  --  % หักลบหลังแปรรูป (Camire et al., 1990) ; ตัวคูณ=(1-loss)
################################################################################
PROCESS_LOSS = {
    "Protein (g)":   0.10,   # ไลซีนไวความร้อน ~10% (high moisture extrusion)
    "Fat (g)":       0.15,   # เซลล์ไขมันแตก + tocopherol ~15%
    "Carb (g)":      0.05,   # Maillard/เจลของแป้ง ~5%
    "Energy (kcal)": 0.00,   # พลังงานถือว่าคงตัว (ปรับได้)
}
KJ_PER_KCAL = 4.184          # kcal -> kJ (ขอบเขต DF ในรายงานเป็น kJ)

# --- HYDRATION / WATER --------------------------------------------------------
# วัตถุดิบ 5 ชนิดรวม = 100% "ไม่มีน้ำ" แต่ขอบเขต DF เป็นค่าผลิตภัณฑ์ที่เติมน้ำแล้ว
# -> dry blend มีพลังงาน/โปรตีนสูงเกิน -> d=0 หมด
# value_product = value_dryblend x (1 - WATER_FRACTION)
#   0.0  -> ตรงตามรายงานเป๊ะ (ฐานแห้ง) แต่ D มัก = 0
#   ~0.55 -> สมจริง (เติมน้ำ) และทำให้ D > 0
#
# อ้างอิงค่า ~0.55 (ความชื้นของผลิตภัณฑ์เนื้อจากพืชจริง):
#  - De Marchi et al. (2021), Sci. Rep. 11:2049 : plant-based burger ดิบ มีความชื้น
#    มัธยฐาน 60.9% (พิสัย 50.5-77.9%)  https://doi.org/10.1038/s41598-021-81684-9
#  - Kyriakopoulou et al. (2021) [อยู่ใน bib ของรายงานแล้ว] และงาน high-moisture
#    extrusion : ผลิตด้วยความชื้น feed 40-80%  -> 0.55 อยู่กลางพิสัยจริง
WATER_FRACTION = 0.0

################################################################################
# 6) DESIRABILITY (DF) BOUNDS  --  ต่อ 100 g แยกตามกลุ่มผลิตภัณฑ์
#    โภชนาการ (Energy/Protein/Fat/Carb) = two-sided [L, T, U] ; T = กึ่งกลาง (แก้ได้)
#    Energy หน่วย kJ ; Protein/Fat/Carb หน่วย g
################################################################################
PRODUCT_GROUPS = {
    "Burger (เบอร์เกอร์)":       {"Energy": (355, 1160), "Protein": (2.9, 20.9), "Fat": (0.5, 17.7), "Carb": (3.9, 33.0)},
    "Ground meat (เนื้อบดเทียม)": {"Energy": (312,  950), "Protein": (4.6, 23.4), "Fat": (0.5, 15.0), "Carb": (1.6, 27.0)},
    "Sausage (ไส้กรอก)":         {"Energy": (458, 1103), "Protein": (2.8, 23.0), "Fat": (1.0, 19.0), "Carb": (0.5, 24.4)},
    "Chicken (ไก่เทียม)":        {"Energy": (274, 1130), "Protein": (1.2, 36.1), "Fat": (2.0, 14.3), "Carb": (2.7, 21.5)},
    "Seafood (อาหารทะเลเทียม)":  {"Energy": (231, 1178), "Protein": (0.3, 14.0), "Fat": (0.5, 12.6), "Carb": (4.4, 30.0)},
    "Other (อื่น ๆ)":           {"Energy": (405, 1180), "Protein": (5.7, 26.4), "Fat": (1.0, 23.2), "Carb": (1.3, 26.7)},
}

# Cost = one-sided minimize : d=1 เมื่อ y<T, ลดเหลือ 0 ที่ y=U
COST_T = 0.0
COST_U = 30.0

DF_WEIGHTS = {"Energy": 1.0, "Protein": 1.0, "Fat": 1.0, "Carb": 1.0, "Cost": 1.0}

################################################################################
# 7) GAUSSIAN PROCESS (GPR / Max-Entropy) kernel : R(h)=prod_j exp(-xi*h_j^2)
################################################################################
GP_XI     = 10.0
GP_NUGGET = 1e-8

################################################################################
# 8) DESIGN DEFAULTS + PERFORMANCE LIMITS
################################################################################
DEFAULT_N_POINTS = 30
RANDOM_SEED      = 42
MAXENT_CANDIDATE_POOL = 2000
MORRIS_MITCHELL_LAMBDA = 40.0
MORRIS_MITCHELL_P      = 1.0

# --- ขีดจำกัดด้านประสิทธิภาพ (กันค้างตอน n ใหญ่) ---
MAX_POINTS            = 10000   # เพดานจำนวนจุดที่ dashboard ยอมรับ
GPR_MAX_N             = 800     # ถ้า design > ค่านี้ GPR/LOOCV/DF จะสุ่มตัวอย่างย่อย
                                # (kriging เป็น O(n^3) ทำ 10,000 จุดตรง ๆ ไม่ไหว)
MAXENT_MAX_N          = 500     # Max-Entropy จำกัดจำนวนจุด (ดูหมายเหตุด้านล่าง)
# ** ทำไม Max-Entropy ถึงจำกัดจำนวนจุด (ต่างจาก LHD/Dirichlet ที่ทำหมื่นจุดได้) **
#  1) ต้นทุนคำนวณ: เป็น greedy เลือกทีละจุดจาก pool -> O(n^2 x pool) ต่อการรัน
#  2) เชิงตัวเลข: kernel R(h)=exp(-xi h^2) ทำให้จุดที่ 'ใกล้กัน' สัมพันธ์กันสูง
#     พอจำนวนจุดมากในพื้นที่จำกัด จุดเริ่มอัดกัน -> R เกือบ singular, det(R)->0
#     เกณฑ์ det(R) จึงหมดความหมาย (saturation)  รายงานก็ระบุว่า MED เหมาะกับ
#     การทดลองจำนวนน้อย  => n ใหญ่ควรใช้ Dirichlet/LHD/Maximin แทน
MAXIMIN_REFINE_MAX_N  = 400     # เกินนี้ Maximin ใช้ greedy อย่างเดียว (ไม่ polish SLSQP)

DESIGN_METHODS = [
    "Latin Hypercube (Rejection)",
    "Latin Hypercube (Transformation)",
    "Simplex Lattice",
    "Simplex Centroid",
    "Maximin (Morris-Mitchell)",
    "Maximum Entropy",
    "Dirichlet",
]

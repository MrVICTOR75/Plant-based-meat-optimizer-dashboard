# Plant-Based Meat Formulation Optimizer

ระบบหาค่าเหมาะที่สุดของสูตรผลิตภัณฑ์เนื้อจากพืช ด้วย **Space-Filling Mixture Design

+ Gaussian Process Regression (GPR) + Multi-Objective Desirability Function (DF)**

## โครงสร้างไฟล์

| ไฟล์            | หน้าที่                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`       | ค่าคงที่กลาง: path, วัตถุดิบ, ข้อจำกัด, ต้นทุน, processing loss, ขอบเขต DF, พารามิเตอร์ GPR |
| `constraints.py`  | เรขาคณิตพื้นที่ออกแบบ (bounded polytope), ตรวจ feasibility, สุ่มจุด, เมทริกซ์ constraint               |
| `m1_nutrition.py` | โมดูล 1: สร้าง`Nutrition_Matrix.xlsx` จากไฟล์ zip (USDA/Kaggle) + ต้นทุน                                              |
| `m2_design.py`    | โมดูล 2: 5 วิธี space-filling + ภาพ simplex 3D                                                                                     |
| `m3_evaluate.py`  | โมดูล 3: แทนค่าสัดส่วน × โภชนาการ × (1−loss) × เจือจางน้ำ →`Project_Data.xlsx`                      |
| `m4_gpr_df.py`    | โมดูล 4: GPR (kriging), LOOCV R², desirability, SLSQP optimization                                                                       |
| `main_inpho.py`   | Dashboard (Streamlit) รวมทุกขั้นตอน                                                                                               |

## การติดตั้ง

```bash
pip install -r requirements.txt
```

วางไฟล์ข้อมูลดิบ 3 ไฟล์ **ในโฟลเดอร์เดียวกับโค้ด** (ไม่บังคับ — ถ้าไม่มีจะใช้ข้อมูลจำลอง):

- `food-nutrition-dataset.zip`
- `FoodData_Central_foundation_food_csv_2025-12-18.zip`
- `FoodData_Central_sr_legacy_food_csv_2018-04.zip`

## การใช้งาน

**เปิด Dashboard:**

```bash
streamlit run main_inpho.py
```

**รันแยกโมดูล (ทดสอบ):**

```bash
python m1_nutrition.py     # สร้าง Nutrition_Matrix.xlsx
python m2_design.py        # ทดสอบ 5 วิธีออกแบบ
python m4_gpr_df.py        # ทดสอบ GPR + LOOCV + DF ครบวงจร
```

## หมายเหตุสำคัญ: Water fraction

วัตถุดิบ 5 ชนิดรวมกัน = 100% **ไม่มีน้ำ** แต่ขอบเขต DF เป็นค่าของผลิตภัณฑ์ที่เติมน้ำแล้ว
(~50–65% น้ำ) จึงต้องตั้ง **Water fraction** ใน dashboard:

- `0.0` = ตรงตามรายงาน (ฐานแห้ง) — แต่ค่าโภชนาการมักเกินขอบเขต ทำให้ **D = 0**
- `~0.55` = สมจริง (ผลิตภัณฑ์เติมน้ำ) — ทำให้ **D > 0** และหาสูตรที่ดีที่สุดได้

## วิธี Space-Filling ที่รองรับ

1. Latin Hypercube — **Rejection** (ตัดจุดนอก simplex/ข้อจำกัด)
2. Latin Hypercube — **Transformation** (inverse-CDF Beta; Fang, Li & Sudjianto 2006)
3. **Simplex Lattice** (Cornell 2002)
4. **Simplex Centroid** (Cornell 2002)
5. **Maximin** — Morris–Mitchell φ_p + row-by-row (Stinstra et al. 2003)
6. **Maximum Entropy** — greedy maximize det(R) ของ Gaussian kernel
7. **Dirichlet(1,…,1)** = uniform บน simplex

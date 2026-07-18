"""
================================================================================
 m3_evaluate.py  --  โมดูล 3 : แทนค่าสัดส่วนลงแบบจำลอง -> คุณค่าโภชนาการ + ต้นทุน
--------------------------------------------------------------------------------
 สำหรับจุดทดลองแต่ละจุด (สัดส่วน x ที่ผลรวม = 1) :
   nutrient_dryblend = ( sum_i  x_i * nutrient_i_per100g ) * (1 - loss_nutrient)
   nutrient_product  = nutrient_dryblend * (1 - WATER_FRACTION)   # เจือจางด้วยน้ำ
   cost_product      = ( sum_i  x_i * cost_i_per100g ) * (1 - WATER_FRACTION)
 เหตุผล: 100 g ของ 'dry blend' มีวัตถุดิบ i อยู่ (x_i*100) g ให้สารอาหาร
         = x_i*(ค่าต่อ 100 g ของ i) ; คูณตัวหักลบหลังแปรรูป ; แล้วเจือจางสู่ผลิตภัณฑ์
         ที่ผ่านการเติมน้ำ (เทียบกับขอบเขต DF ที่เป็นค่าผลิตภัณฑ์)
 บันทึกผลเป็น Project_Data.xlsx
================================================================================
"""

import numpy as np
import pandas as pd

import config as C


################################################################################
# 1) เตรียมเวกเตอร์คุณค่าต่อ 100 g ของวัตถุดิบ (เรียงตาม INGREDIENTS)
################################################################################
def _nutrient_vectors(matrix_df):
    """ดึงเวกเตอร์ (ยาว 5 เรียงตาม INGREDIENTS) ของแต่ละสารอาหาร + ต้นทุน"""
    m = matrix_df.set_index("Category")
    order = C.INGREDIENTS
    return {
        "Protein (g)":   m.loc[order, "Protein (g)"].to_numpy(float),
        "Energy (kcal)": m.loc[order, "Energy (kcal)"].to_numpy(float),
        "Fat (g)":       m.loc[order, "Fat (g)"].to_numpy(float),
        "Carb (g)":      m.loc[order, "Carb (g)"].to_numpy(float),
        "Cost (Baht)":   m.loc[order, "Cost (Baht)"].to_numpy(float),
    }


################################################################################
# 2) คำนวณผลลัพธ์ของ 'ทั้งชุด' จุดทดลอง
################################################################################
def evaluate_design(design_df, matrix_df, save=True, water_fraction=None):
    """คืน DataFrame ต่อท้ายคอลัมน์ผลลัพธ์ (ต่อ 100 g ผลิตภัณฑ์ หลังแปรรูป+เติมน้ำ):
       Protein_out (g), Energy_out (kcal), Energy_out (kJ), Fat_out (g),
       Carb_out (g), Cost (Baht)
       water_fraction : None = ใช้ config.WATER_FRACTION"""
    wf = C.WATER_FRACTION if water_fraction is None else water_fraction
    dil = (1.0 - wf)
    vec = _nutrient_vectors(matrix_df)
    X = design_df[C.INGREDIENTS].to_numpy(float)          # (n x 5)

    def weighted(nut):                                    # sum_i x_i * value_i
        return X @ vec[nut]

    prot    = weighted("Protein (g)")   * (1 - C.PROCESS_LOSS["Protein (g)"])   * dil
    fat     = weighted("Fat (g)")       * (1 - C.PROCESS_LOSS["Fat (g)"])       * dil
    carb    = weighted("Carb (g)")      * (1 - C.PROCESS_LOSS["Carb (g)"])      * dil
    en_kcal = weighted("Energy (kcal)") * (1 - C.PROCESS_LOSS["Energy (kcal)"]) * dil
    cost    = weighted("Cost (Baht)")   * dil

    out = design_df.copy()
    out["Protein_out (g)"]   = prot
    out["Energy_out (kcal)"] = en_kcal
    out["Energy_out (kJ)"]   = en_kcal * C.KJ_PER_KCAL    # DF ใช้หน่วย kJ
    out["Fat_out (g)"]       = fat
    out["Carb_out (g)"]      = carb
    out["Cost (Baht)"]       = cost

    if save:
        out.to_excel(C.PROJECT_DATA_XLSX, index=False, engine="openpyxl")
    return out


################################################################################
# 3) เวกเตอร์ผลลัพธ์สำหรับ 'จุดเดียว' (ใช้ตอน optimize DF)
################################################################################
def responses_from_x(x, matrix_df, water_fraction=None):
    """รับสัดส่วน x (ยาว 5) -> dict response (หน่วยตรงกับ DF):
       Energy(kJ), Protein(g), Fat(g), Carb(g), Cost(บาท/100g)"""
    wf = C.WATER_FRACTION if water_fraction is None else water_fraction
    dil = (1.0 - wf)
    vec = _nutrient_vectors(matrix_df)
    x = np.asarray(x, float)
    prot = float(x @ vec["Protein (g)"])   * (1 - C.PROCESS_LOSS["Protein (g)"])   * dil
    fat  = float(x @ vec["Fat (g)"])       * (1 - C.PROCESS_LOSS["Fat (g)"])       * dil
    carb = float(x @ vec["Carb (g)"])      * (1 - C.PROCESS_LOSS["Carb (g)"])      * dil
    en   = float(x @ vec["Energy (kcal)"]) * (1 - C.PROCESS_LOSS["Energy (kcal)"]) * dil
    cost = float(x @ vec["Cost (Baht)"])   * dil
    return {"Energy": en * C.KJ_PER_KCAL, "Protein": prot,
            "Fat": fat, "Carb": carb, "Cost": cost}


if __name__ == "__main__":
    import m1_nutrition as m1, m2_design as m2
    matrix, _ = m1.load_matrix()
    design, _ = m2.generate_design("Maximin (Morris-Mitchell)", 20)
    out = evaluate_design(design, matrix, save=True)
    print(out.round(2).to_string(index=False))

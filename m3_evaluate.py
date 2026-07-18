import numpy as np
import pandas as pd

import config as C

def _nutrient_vectors(matrix_df):

    m = matrix_df.set_index("Category")
    order = C.INGREDIENTS
    return {
        "Protein (g)":   m.loc[order, "Protein (g)"].to_numpy(float),
        "Energy (kcal)": m.loc[order, "Energy (kcal)"].to_numpy(float),
        "Fat (g)":       m.loc[order, "Fat (g)"].to_numpy(float),
        "Carb (g)":      m.loc[order, "Carb (g)"].to_numpy(float),
        "Cost (Baht)":   m.loc[order, "Cost (Baht)"].to_numpy(float),
    }

def evaluate_design(design_df, matrix_df, save=True, water_fraction=None):

    wf = C.WATER_FRACTION if water_fraction is None else water_fraction
    dil = (1.0 - wf)
    vec = _nutrient_vectors(matrix_df)
    X = design_df[C.INGREDIENTS].to_numpy(float)

    def weighted(nut):
        return X @ vec[nut]

    prot    = weighted("Protein (g)")   * (1 - C.PROCESS_LOSS["Protein (g)"])   * dil
    fat     = weighted("Fat (g)")       * (1 - C.PROCESS_LOSS["Fat (g)"])       * dil
    carb    = weighted("Carb (g)")      * (1 - C.PROCESS_LOSS["Carb (g)"])      * dil
    en_kcal = weighted("Energy (kcal)") * (1 - C.PROCESS_LOSS["Energy (kcal)"]) * dil
    cost    = weighted("Cost (Baht)")   * dil

    out = design_df.copy()
    out["Protein_out (g)"]   = prot
    out["Energy_out (kcal)"] = en_kcal
    out["Energy_out (kJ)"]   = en_kcal * C.KJ_PER_KCAL
    out["Fat_out (g)"]       = fat
    out["Carb_out (g)"]      = carb
    out["Cost (Baht)"]       = cost

    if save:
        out.to_excel(C.PROJECT_DATA_XLSX, index=False, engine="openpyxl")
    return out

def responses_from_x(x, matrix_df, water_fraction=None):

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

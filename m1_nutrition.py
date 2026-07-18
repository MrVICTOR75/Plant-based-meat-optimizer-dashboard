import os
import zipfile
import numpy as np
import pandas as pd

import config as C

def _load_global_nutrients():

    g_nut = pd.DataFrame()
    for path in C.ZIP_FILES:
        if not os.path.exists(path):
            continue
        if "food-nutrition-dataset" in os.path.basename(path).lower():
            continue
        try:
            with zipfile.ZipFile(path, "r") as z:
                nut_files = [f for f in z.namelist() if "food_nutrient.csv" in f.lower()]
                if nut_files:
                    with z.open(nut_files[0]) as f:
                        df = pd.read_csv(f, usecols=["fdc_id", "nutrient_id", "amount"],
                                         low_memory=False)
                        g_nut = pd.concat([g_nut, df], ignore_index=True)
        except Exception as e:
            print(f"[!] โหลด nutrient จาก {os.path.basename(path)} ไม่สำเร็จ: {e}")
    if not g_nut.empty:
        g_nut.drop_duplicates(inplace=True)
    return g_nut

def _scan_foods():

    found = []
    for path in C.ZIP_FILES:
        if not os.path.exists(path):
            print(f"[!] ไม่พบไฟล์: {path}")
            continue
        try:
            with zipfile.ZipFile(path, "r") as z:
                food_files = [f for f in z.namelist()
                              if f.endswith(".csv") and
                              ("FOOD-DATA-GROUP" in f.upper() or f.lower().endswith("food.csv"))]
                for target in food_files:
                    with z.open(target) as f:
                        df = pd.read_csv(f, low_memory=False)
                    col = next((c for c in ["long_description", "description", "food"]
                                if c in df.columns), None)
                    if col is None:
                        continue

                    internal = ("Protein" in df.columns) or ("Caloric Value" in df.columns)
                    for cat, pattern in C.SEARCH_GROUPS.items():
                        hits = df[df[col].fillna("").str.contains(pattern, case=False, regex=True)]
                        for _, row in hits.iterrows():
                            pkt = {
                                "Category": cat,
                                "Food": row[col],
                                "SourceZip": os.path.basename(path),
                                "fdc_id": row["fdc_id"] if "fdc_id" in row.index else np.nan,
                                "internal": internal,
                            }
                            if internal:
                                pkt["Protein (g)"]   = row.get("Protein", 0.0)
                                pkt["Energy (kcal)"] = row.get("Caloric Value", 0.0)
                                pkt["Fat (g)"]       = row.get("Fat", 0.0)
                                pkt["Carb (g)"]      = row.get("Carbohydrates", 0.0)
                            found.append(pkt)
        except Exception as e:
            print(f"[!] ประมวลผล {os.path.basename(path)} ไม่สำเร็จ: {e}")
    return found

def _source_label(src):
    s = src.lower()
    if "sr_legacy" in s:              return "SR_Legacy"
    if "foundation" in s:             return "Foundation"
    if "food-nutrition-dataset" in s: return "Kaggle"
    return src

def build_nutrition_matrix(save=True, verbose=True):

    g_nut = _load_global_nutrients()
    found = _scan_foods()
    if not found:
        raise FileNotFoundError(
            "ไม่พบวัตถุดิบใด ๆ -- ตรวจว่าไฟล์ zip อยู่ในโฟลเดอร์เดียวกับโค้ด (ดู config.ZIP_FILES)")

    df_all = pd.DataFrame(found)
    cost_baht = C.cost_baht_per_100g()
    rows = []

    for cat in C.SEARCH_GROUPS.keys():
        sub = df_all[df_all["Category"] == cat]
        best = {"Category": cat, "Source": "None", "Food": "None",
                "Protein (g)": 0.0, "Energy (kcal)": 0.0, "Fat (g)": 0.0, "Carb (g)": 0.0}
        best_total, saved = -1.0, False

        sub = sub.sort_values("internal")
        for _, m in sub.iterrows():
            nuts, total = {}, 0.0
            if m["internal"]:
                for name in C.NUTRIENT_MAP.values():
                    v = float(m.get(name, 0.0) or 0.0)
                    nuts[name] = v; total += v
            else:
                for nid, name in C.NUTRIENT_MAP.items():
                    ids = [nid] if nid != 1008 else [1008, 2047, 2048]
                    v = 0.0
                    if not g_nut.empty and pd.notna(m["fdc_id"]):
                        hit = g_nut[(g_nut["fdc_id"] == m["fdc_id"]) &
                                    (g_nut["nutrient_id"].isin(ids))]
                        if not hit.empty:
                            v = float(hit.sort_values("amount", ascending=False)["amount"].iloc[0])
                    nuts[name] = v; total += v

            if (total > best_total) or (not saved):
                best_total, saved = total, True
                best["Source"] = _source_label(m["SourceZip"])
                best["Food"]   = str(m["Food"])
                for name, v in nuts.items():
                    best[name] = v if pd.notna(v) else 0.0

        best["Cost (Baht)"] = round(cost_baht.get(cat, 0.0), 4)
        rows.append(best)

    cols = ["Category", "Source", "Food", "Protein (g)", "Energy (kcal)",
            "Fat (g)", "Carb (g)", "Cost (Baht)"]
    matrix = pd.DataFrame(rows)[cols]

    if save:
        matrix.to_excel(C.NUTRITION_MATRIX_XLSX, index=False, engine="openpyxl")
        if verbose:
            print(f"[OK] บันทึก {C.NUTRITION_MATRIX_XLSX}")
    if verbose:
        print(matrix.to_string(index=False))
    return matrix

def get_mock_matrix():

    cost = C.cost_baht_per_100g()
    data = [

        ["SPI",    "SR_Legacy",  "Soy protein isolate",        88.3, 335, 0.5,  0.0],
        ["Gluten", "SR_Legacy",  "Wheat gluten, vital",        75.2, 370, 1.9,  13.8],
        ["Pea",    "SR_Legacy",  "Peas, split, mature, raw",   24.6, 341, 1.2,  60.4],
        ["Oil",    "SR_Legacy",  "Oil, coconut",               0.0,  892, 99.1, 0.0],
        ["Binder", "SR_Legacy",  "Seaweed, agar, dried",       6.2,  306, 0.3,  80.9],
    ]
    df = pd.DataFrame(data, columns=["Category", "Source", "Food",
                                     "Protein (g)", "Energy (kcal)", "Fat (g)", "Carb (g)"])
    df["Cost (Baht)"] = [round(cost[c], 4) for c in df["Category"]]
    return df

def load_matrix(prefer_real=True, verbose=False):

    have_zip = all(os.path.exists(p) for p in C.ZIP_FILES)
    if prefer_real and have_zip:
        try:
            return build_nutrition_matrix(save=True, verbose=verbose), False
        except Exception as e:
            print(f"[!] สร้างของจริงไม่สำเร็จ ({e}) -> ใช้ mock")

    if os.path.exists(C.NUTRITION_MATRIX_XLSX):
        try:
            df = pd.read_excel(C.NUTRITION_MATRIX_XLSX)
            df["Cost (Baht)"] = df["Category"].map(C.cost_baht_per_100g()).fillna(df["Cost (Baht)"])
            return df, False
        except Exception:
            pass
    return get_mock_matrix(), True

if __name__ == "__main__":
    build_nutrition_matrix(save=True, verbose=True)

"""
================================================================================
 m4_gpr_df.py  --  โมดูล 4 : GPR + LOOCV + Desirability Function + SLSQP
--------------------------------------------------------------------------------
 (A) GPR  : ŷ(x*) = mu + r*(x*)^T R^{-1}(y - mu)  (ordinary kriging)
 (B) LOOCV: สูตรปิด Dubrule (1983) -> R^2 = 1 - PRESS/SSTO (คำนวณ R^{-1} ครั้งเดียว)
 (C) DF   : d รายตัว (two-sided โภชนาการ / one-sided-min ต้นทุน) -> D = geometric mean
 (D) SLSQP: หาสัดส่วน x ที่ D สูงสุด ภายใต้ข้อจำกัด (multi-start)
================================================================================
"""

import numpy as np
from scipy.optimize import minimize

import config as C
import constraints as K
import m3_evaluate as m3

RESPONSES = ["Energy", "Protein", "Fat", "Carb", "Cost"]
_TARGET_COL = {"Energy": "Energy_out (kJ)", "Protein": "Protein_out (g)",
               "Fat": "Fat_out (g)", "Carb": "Carb_out (g)", "Cost": "Cost (Baht)"}


################################################################################
# (A) GPR  --  ordinary kriging ด้วย Gaussian correlation
################################################################################
def _corr(A, B, xi):
    """R = prod_k exp(-xi*(a_k-b_k)^2)  ระหว่างชุดจุด A (na x d) และ B (nb x d)"""
    diff = A[:, None, :] - B[None, :, :]
    return np.exp(-xi * (diff ** 2).sum(axis=-1))


def fit_gpr(X, y, xi=None, nugget=None):
    """ฟิต kriging 1 response -> dict {mu, w, Rinv, X, xi}"""
    xi = C.GP_XI if xi is None else xi
    nugget = C.GP_NUGGET if nugget is None else nugget
    n = len(X)
    R = _corr(X, X, xi) + nugget * np.eye(n)
    Rinv = np.linalg.inv(R)
    one = np.ones(n)
    mu = (one @ Rinv @ y) / (one @ Rinv @ one)
    w = Rinv @ (y - mu * one)
    return {"mu": mu, "w": w, "Rinv": Rinv, "X": X, "xi": xi}


def gpr_predict(model, Xstar):
    """พยากรณ์ค่าเฉลี่ย ŷ ที่จุดใหม่ Xstar (m x d) -> ndarray (m,)"""
    Xstar = np.atleast_2d(Xstar)
    r = _corr(Xstar, model["X"], model["xi"])
    return model["mu"] + r @ model["w"]


def fit_all_responses(X, R_df, xi=None):
    """ฟิต GPR ทุก response -> dict {response: model}"""
    return {resp: fit_gpr(X, R_df[_TARGET_COL[resp]].to_numpy(float), xi=xi)
            for resp in RESPONSES}


################################################################################
# (B) LOOCV  --  สูตรปิด (Dubrule 1983) : เร็ว O(n^3) ครั้งเดียว
################################################################################
def loocv_r2(X, R_df, xi=None):
    """คืน dict {response: R^2} ตาม R^2 = 1 - PRESS/SSTO
       Q = R^{-1} - R^{-1}F (F^T R^{-1} F)^{-1} F^T R^{-1}  (F = 1 : constant trend)
       residual LOO ของจุด i :  e_i = (Q y)_i / Q_ii"""
    xi = C.GP_XI if xi is None else xi
    n = len(X)
    R = _corr(X, X, xi) + C.GP_NUGGET * np.eye(n)
    Rinv = np.linalg.inv(R)
    F = np.ones((n, 1))
    M = F.T @ Rinv @ F
    Q = Rinv - Rinv @ F @ np.linalg.inv(M) @ F.T @ Rinv
    diagQ = np.clip(np.diag(Q), 1e-12, None)

    r2 = {}
    for resp in RESPONSES:
        y = R_df[_TARGET_COL[resp]].to_numpy(float)
        e = (Q @ y) / diagQ
        press = float(np.sum(e ** 2))
        ssto = float(((y - y.mean()) ** 2).sum())
        r2[resp] = 1.0 - press / ssto if ssto > 1e-12 else np.nan
    return r2


################################################################################
# (C) DESIRABILITY FUNCTIONS
################################################################################
def d_two_sided(y, L, T, U, r1=1.0, r2=1.0):
    """d สองด้าน : 0 นอกช่วง [L,U], ไต่ถึง 1 ที่ T"""
    if y < L or y > U:
        return 0.0
    if y <= T:
        return ((y - L) / (T - L)) ** r1 if T > L else 1.0
    return ((U - y) / (U - T)) ** r2 if U > T else 1.0


def d_min(y, T, U, r=1.0):
    """d ต้องการค่าต่ำสุด (ต้นทุน) : 1 เมื่อ y<T, ลดเหลือ 0 ที่ y=U"""
    if y < T:
        return 1.0
    if y > U:
        return 0.0
    return ((U - y) / (U - T)) ** r if U > T else 1.0


def individual_desirabilities(responses, group_name, cost_u=None):
    """d รายตัวของทุก response ; two-sided ใช้ T = กึ่งกลาง (L+U)/2
       cost_u : ขอบบนต้นทุน (None = ใช้ config.COST_U) -- ปรับได้จาก dashboard"""
    cu = C.COST_U if cost_u is None else cost_u
    bounds = C.PRODUCT_GROUPS[group_name]
    d = {}
    for resp in ["Energy", "Protein", "Fat", "Carb"]:
        L, U = bounds[resp]
        T = 0.5 * (L + U)
        w = C.DF_WEIGHTS[resp]
        d[resp] = d_two_sided(responses[resp], L, T, U, r1=w, r2=w)
    d["Cost"] = d_min(responses["Cost"], C.COST_T, cu, r=C.DF_WEIGHTS["Cost"])
    return d


def overall_D(responses, group_name, cost_u=None):
    """D รวม = geometric mean ของ d ทั้งหมด (ถ้ามีตัวใด=0 -> D=0)"""
    d = individual_desirabilities(responses, group_name, cost_u=cost_u)
    vals = np.array(list(d.values()), float)
    if np.any(vals <= 0):
        return 0.0, d
    return float(np.exp(np.mean(np.log(vals)))), d


################################################################################
# (D) SLSQP  --  maximize D (multi-start)
################################################################################
def optimize_desirability(models, group_name, n_starts=40, seed=None, cost_u=None):
    """ใช้ GPR surrogate พยากรณ์ response ที่จุด x แล้ว maximize D ผ่าน SLSQP
       (minimize -D) จากหลายจุดเริ่ม -> คืน dict {x, D, d, responses}
       cost_u : ขอบบนต้นทุนของ DF (None = config.COST_U)"""
    rng = np.random.default_rng(C.RANDOM_SEED if seed is None else seed)

    def responses_at(x):
        return {resp: float(gpr_predict(models[resp], np.atleast_2d(x))[0])
                for resp in RESPONSES}

    def neg_D(x):
        return -overall_D(responses_at(x), group_name, cost_u=cost_u)[0]

    cons = K.scipy_constraints()
    lows, highs, _ = K.get_bounds()
    bounds = list(zip(lows, highs))
    starts = K.sample_feasible(n_starts, rng=rng)

    best = None
    for x0 in starts:
        res = minimize(neg_D, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
        if not res.success or not K.is_feasible(res.x):
            continue
        D, d = overall_D(responses_at(res.x), group_name, cost_u=cost_u)
        if (best is None) or (D > best["D"]):
            best = {"x": res.x, "D": D, "d": d, "responses": responses_at(res.x)}

    if best is None:   # เผื่อ SLSQP ล้มทุกจุด : ใช้จุดเริ่มที่ดีสุด
        scored = [(overall_D(responses_at(x), group_name, cost_u=cost_u)[0], x) for x in starts]
        D, x = max(scored, key=lambda t: t[0])
        best = {"x": x, "D": D,
                "d": individual_desirabilities(responses_at(x), group_name, cost_u=cost_u),
                "responses": responses_at(x)}
    return best


def format_best(best):
    """จัดผลลัพธ์ให้อ่านง่าย -> dict พร้อมสัดส่วนรายวัตถุดิบ (%)"""
    x = best["x"]
    return {
        "D": best["D"],
        "formulation_%": {ing: round(100 * xi, 2) for ing, xi in zip(C.INGREDIENTS, x)},
        "responses": {k: round(v, 2) for k, v in best["responses"].items()},
        "individual_d": {k: round(v, 3) for k, v in best["d"].items()},
    }


def maybe_subsample(design_df, evald, max_n, seed=0):
    """ถ้าจำนวนจุด > max_n : สุ่มตัวอย่างย่อยสำหรับ GPR/LOOCV/DF (kriging O(n^3))
       คืน (design_sub, evald_sub, was_subsampled)"""
    if len(design_df) <= max_n:
        return design_df, evald, False
    idx = np.sort(np.random.default_rng(seed).choice(len(design_df), max_n, replace=False))
    return (design_df.iloc[idx].reset_index(drop=True),
            evald.iloc[idx].reset_index(drop=True), True)


if __name__ == "__main__":
    import m1_nutrition as m1, m2_design as m2
    matrix, is_mock = m1.load_matrix()
    design, info = m2.generate_design("Maximin (Morris-Mitchell)", 30)
    evald = m3.evaluate_design(design, matrix, save=False, water_fraction=0.55)
    X = design[C.INGREDIENTS].to_numpy(float)
    r2 = loocv_r2(X, evald)
    print("LOOCV R^2:", {k: round(v, 3) for k, v in r2.items()})
    models = fit_all_responses(X, evald)
    best = optimize_desirability(models, "Burger (เบอร์เกอร์)")
    import json
    print(json.dumps(format_best(best), ensure_ascii=False, indent=2))

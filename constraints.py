"""
================================================================================
 constraints.py  --  เรขาคณิตพื้นที่ออกแบบ (feasible polytope) + ตัวช่วยสุ่ม
--------------------------------------------------------------------------------
 พื้นที่ออกแบบ = ส่วนของ simplex S^4 ที่ถูกตัดด้วย:
   Sum(x)=1 , และ per-ingredient bounds (min_i <= x_i <= max_i)
   group : PROTEIN_SUM_MIN <= xSPI+xGluten+xPea <= PROTEIN_SUM_MAX
 * bounds เก็บใน 'active state' ที่ปรับสด ๆ ได้ (dashboard เรียก set_ingredient_min)
 * ตัวสุ่มใช้ 'L-pseudocomponent' : เติมขอบล่าง L_i ก่อน แล้วกระจายมวลที่เหลือ
   (1 - sum L) บน simplex -> ขอบล่างถูกเสมอ, reject เฉพาะขอบบน+group -> เร็วมาก
================================================================================
"""

import numpy as np
import config as C


################################################################################
# 0) ACTIVE BOUNDS  --  สถานะที่ปรับได้ (เริ่มจากค่าใน config)
################################################################################
_ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}
_PROT = [C.PROTEIN_SUM_MIN, C.PROTEIN_SUM_MAX]


def set_ingredient_min(ing, lo):
    """ตั้งขอบล่างของวัตถุดิบ ing (ใช้บังคับให้ใส่ gluten/binder ฯลฯ)"""
    _ING_BOUNDS[ing][0] = float(lo)


def set_ingredient_max(ing, hi):
    _ING_BOUNDS[ing][1] = float(hi)


def set_protein_sum(lo, hi):
    _PROT[0], _PROT[1] = float(lo), float(hi)


def reset_bounds():
    """คืนค่ากลับเป็นค่าเริ่มต้นใน config"""
    global _ING_BOUNDS, _PROT
    _ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}
    _PROT = [C.PROTEIN_SUM_MIN, C.PROTEIN_SUM_MAX]


def get_bounds():
    """คืน (lows, highs) เป็น ndarray เรียงตาม INGREDIENTS + (prot_lo, prot_hi)"""
    lows = np.array([_ING_BOUNDS[k][0] for k in C.INGREDIENTS])
    highs = np.array([_ING_BOUNDS[k][1] for k in C.INGREDIENTS])
    return lows, highs, tuple(_PROT)


################################################################################
# 1) FEASIBILITY CHECK
################################################################################
def is_feasible(x, tol=1e-9):
    x = np.asarray(x, float)
    if x.shape[-1] != C.D:
        return False
    lows, highs, (plo, phi) = get_bounds()
    prot = x[C.PROTEIN_IDX].sum()
    return (abs(x.sum() - 1.0) <= 1e-6
            and np.all(x >= lows - tol) and np.all(x <= highs + tol)
            and (plo - tol <= prot <= phi + tol))


def feasible_mask(X):
    """vectorized : X (n x 5) -> boolean array (n,)"""
    X = np.asarray(X, float)
    lows, highs, (plo, phi) = get_bounds()
    prot = X[:, C.PROTEIN_IDX].sum(axis=1)
    m = (np.abs(X.sum(axis=1) - 1.0) <= 1e-6)
    m &= np.all(X >= lows - 1e-9, axis=1)
    m &= np.all(X <= highs + 1e-9, axis=1)
    m &= (prot >= plo - 1e-9) & (prot <= phi + 1e-9)
    return m


################################################################################
# 2) LINEAR CONSTRAINTS  A x <= b  (สำหรับ SLSQP)  -- ไม่รวม Sum=1
################################################################################
def linear_ineq_Ab():
    lows, highs, (plo, phi) = get_bounds()
    A, b = [], []
    # per-ingredient : x_i <= hi ,  -x_i <= -lo
    for i in range(C.D):
        e = np.zeros(C.D); e[i] = 1.0
        A.append(e.copy());  b.append(highs[i])
        A.append(-e.copy()); b.append(-lows[i])
    # protein group
    prot = np.zeros(C.D); prot[C.PROTEIN_IDX] = 1.0
    A.append(prot.copy());  b.append(phi)
    A.append(-prot.copy()); b.append(-plo)
    return np.array(A), np.array(b)


def scipy_constraints():
    A, b = linear_ineq_Ab()
    return [
        {"type": "eq",   "fun": lambda x: np.sum(x) - 1.0},         # Sum = 1
        {"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x},     # A x <= b
    ]


################################################################################
# 3) SAMPLER  --  L-pseudocomponent + rejection ขอบบน/group (เร็วแม้ n ใหญ่)
################################################################################
def sample_feasible(n, rng=None, max_rounds=2000):
    """คืน (n x 5) จุด feasible :
       x = L + (1 - sum L) * Dirichlet(1)   -> ขอบล่างถูกเสมอ
       แล้ว reject จุดที่ละเมิดขอบบน หรือ group protein-sum"""
    rng = rng or np.random.default_rng(C.RANDOM_SEED)
    lows, highs, _ = get_bounds()
    s = lows.sum()
    if s > 1.0 + 1e-9:
        raise ValueError("ผลรวมขอบล่าง > 1 : ข้อจำกัดขัดกัน (ลดค่า min ลง)")
    free = 1.0 - s
    out, rounds = [], 0
    while len(out) < n and rounds < max_rounds:
        batch_n = max(int(n * 2.5), 500)
        base = rng.dirichlet(np.ones(C.D), size=batch_n) * free
        X = base + lows                                  # ขอบล่างถูกโดยสร้าง
        X = X[feasible_mask(X)]                          # กรองขอบบน + group
        out.extend(list(X))
        rounds += 1
    if len(out) < n:
        raise RuntimeError(f"สุ่มได้ {len(out)}/{n} -- พื้นที่แคบ ลองผ่อนขอบล่าง/บน")
    return np.array(out[:n])


def acceptance_rate(rng=None, n_test=20000):
    """สัดส่วนจุดที่ผ่านหลังเติมขอบล่างแล้ว (บอกความแคบของขอบบน+group)"""
    rng = rng or np.random.default_rng(0)
    lows, _, _ = get_bounds()
    free = 1.0 - lows.sum()
    X = rng.dirichlet(np.ones(C.D), size=n_test) * free + lows
    return float(feasible_mask(X).mean())

import numpy as np
import config as C

_ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}
_PROT = [C.PROTEIN_SUM_MIN, C.PROTEIN_SUM_MAX]

def set_ingredient_min(ing, lo):

    _ING_BOUNDS[ing][0] = float(lo)

def set_ingredient_max(ing, hi):
    _ING_BOUNDS[ing][1] = float(hi)

def set_protein_sum(lo, hi):
    _PROT[0], _PROT[1] = float(lo), float(hi)

def reset_bounds():

    global _ING_BOUNDS, _PROT
    _ING_BOUNDS = {k: list(v) for k, v in C.INGREDIENT_BOUNDS.items()}
    _PROT = [C.PROTEIN_SUM_MIN, C.PROTEIN_SUM_MAX]

def get_bounds():

    lows = np.array([_ING_BOUNDS[k][0] for k in C.INGREDIENTS])
    highs = np.array([_ING_BOUNDS[k][1] for k in C.INGREDIENTS])
    return lows, highs, tuple(_PROT)

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

    X = np.asarray(X, float)
    lows, highs, (plo, phi) = get_bounds()
    prot = X[:, C.PROTEIN_IDX].sum(axis=1)
    m = (np.abs(X.sum(axis=1) - 1.0) <= 1e-6)
    m &= np.all(X >= lows - 1e-9, axis=1)
    m &= np.all(X <= highs + 1e-9, axis=1)
    m &= (prot >= plo - 1e-9) & (prot <= phi + 1e-9)
    return m

def linear_ineq_Ab():
    lows, highs, (plo, phi) = get_bounds()
    A, b = [], []

    for i in range(C.D):
        e = np.zeros(C.D); e[i] = 1.0
        A.append(e.copy());  b.append(highs[i])
        A.append(-e.copy()); b.append(-lows[i])

    prot = np.zeros(C.D); prot[C.PROTEIN_IDX] = 1.0
    A.append(prot.copy());  b.append(phi)
    A.append(-prot.copy()); b.append(-plo)
    return np.array(A), np.array(b)

def scipy_constraints():
    A, b = linear_ineq_Ab()
    return [
        {"type": "eq",   "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x},
    ]

def sample_feasible(n, rng=None, max_rounds=2000):

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
        X = base + lows
        X = X[feasible_mask(X)]
        out.extend(list(X))
        rounds += 1
    if len(out) < n:
        raise RuntimeError(f"สุ่มได้ {len(out)}/{n} -- พื้นที่แคบ ลองผ่อนขอบล่าง/บน")
    return np.array(out[:n])

def acceptance_rate(rng=None, n_test=20000):

    rng = rng or np.random.default_rng(0)
    lows, _, _ = get_bounds()
    free = 1.0 - lows.sum()
    X = rng.dirichlet(np.ones(C.D), size=n_test) * free + lows
    return float(feasible_mask(X).mean())

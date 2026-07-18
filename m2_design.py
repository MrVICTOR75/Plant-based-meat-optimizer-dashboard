import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize

import config as C
import constraints as K

def _lhd(n, dim, rng):

    X = np.empty((n, dim))
    for k in range(dim):
        perm = rng.permutation(n) + 1
        U = rng.random(n)
        X[:, k] = (perm - 1 + U) / n
    return X

def _cube_to_simplex(U):

    n, dm1 = U.shape
    d = dm1 + 1
    X = np.zeros((n, d))
    remaining = np.ones(n)
    for k in range(dm1):
        power = 1.0 / (d - 1 - k)
        xk = remaining * (1.0 - U[:, k] ** power)
        X[:, k] = xk
        remaining = remaining - xk
    X[:, d - 1] = remaining
    return X

def _pairwise_dist(X):

    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def phi_p(X, p=None, lam=None, max_n=3000):

    if len(X) > max_n:
        return float("nan")
    lam = C.MORRIS_MITCHELL_LAMBDA if lam is None else lam
    Dm = _pairwise_dist(X)
    d = np.maximum(Dm[np.triu_indices(len(X), k=1)], 1e-12)
    return (np.sum((1.0 / d) ** lam)) ** (1.0 / lam)

def min_pairwise_distance(X):

    if len(X) < 2:
        return 0.0
    if len(X) <= 2000:
        Dm = _pairwise_dist(X)
        return float(Dm[np.triu_indices(len(X), k=1)].min())
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(X).query(X, k=2)
    return float(dist[:, 1].min())

def _greedy_farthest_point(pool, n, rng):

    n = min(n, len(pool))
    idx0 = int(rng.integers(len(pool)))
    chosen = [idx0]
    d_to_set = np.linalg.norm(pool - pool[idx0], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d_to_set))
        chosen.append(nxt)
        d_to_set = np.minimum(d_to_set, np.linalg.norm(pool - pool[nxt], axis=1))
    return pool[chosen]

def lhd_rejection(n, rng):

    got, factor = [], 20
    while len(got) < n and factor <= 80000:
        M = max(n * factor, 400)
        U = _lhd(M, C.D - 1, rng)
        X = np.column_stack([U, 1.0 - U.sum(axis=1)])
        X = X[K.feasible_mask(X)]
        got.extend(list(X))
        factor *= 4
    if not got:
        raise RuntimeError("LHD-Rejection: หาจุด feasible ไม่ได้เลย (ข้อจำกัดแคบเกิน)")
    return np.array(got[:n])

def lhd_transformation(n, rng):

    lows, _, _ = K.get_bounds()
    free = 1.0 - lows.sum()
    got, factor = [], 6
    while len(got) < n and factor <= 80000:
        M = max(n * factor, 400)
        U = _lhd(M, C.D - 1, rng)
        X = _cube_to_simplex(U) * free + lows
        X = X[K.feasible_mask(X)]
        got.extend(list(X))
        factor *= 4
    if len(got) < n:
        return np.array(got) if got else lhd_rejection(n, rng)
    return np.array(got[:n])

def _compositions(total, parts):

    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest

def _pseudo_to_x(Z):

    lows, _, _ = K.get_bounds()
    return np.atleast_2d(Z) * (1.0 - lows.sum()) + lows

def simplex_lattice(n=None, degree=None, max_degree=25):

    n = n or C.DEFAULT_N_POINTS
    m_list = [degree] if degree else range(2, max_degree + 1)
    best = np.zeros((0, C.D))
    for m in m_list:
        Z = np.array([c for c in _compositions(m, C.D)], float) / m
        pts = _pseudo_to_x(Z)
        pts = pts[K.feasible_mask(pts)]
        if len(pts) >= len(best):
            best = pts
        if degree is None and len(pts) >= n:
            break
    return best

def _subset_centroids(d):

    Z = []
    for r in range(1, d + 1):
        for S in combinations(range(d), r):
            z = np.zeros(d)
            for k in S:
                z[k] = 1.0 / r
            Z.append(z)
    return np.array(Z)

def simplex_centroid(n=None, max_degree=25):

    n = n or C.DEFAULT_N_POINTS
    d = C.D
    base = _subset_centroids(d)
    best = _pseudo_to_x(base)
    best = np.unique(np.round(best[K.feasible_mask(best)], 9), axis=0)
    for m in range(2, max_degree + 1):
        cells = np.array([a for a in _compositions(m - 1, d)], float)
        subc = (cells + 1.0 / d) / m
        pts = _pseudo_to_x(np.vstack([base, subc]))
        pts = np.unique(np.round(pts[K.feasible_mask(pts)], 9), axis=0)
        if len(pts) >= len(best):
            best = pts
        if len(pts) >= n:
            break
    return best

def _phi_contrib(xi, others, lam):
    d = np.maximum(np.linalg.norm(others - xi, axis=1), 1e-12)
    return np.sum((1.0 / d) ** lam)

def maximin_design(n, rng, sweeps=2):

    lam = C.MORRIS_MITCHELL_LAMBDA
    pool_size = int(min(max(20 * n, 2000), 60000))
    pool = K.sample_feasible(pool_size, rng=rng)
    X = _greedy_farthest_point(pool, n, rng).copy()

    if n <= C.MAXIMIN_REFINE_MAX_N:
        lows, highs, _ = K.get_bounds()
        bounds = list(zip(lows, highs))
        cons = K.scipy_constraints()
        for _ in range(sweeps):
            for i in range(len(X)):
                others = np.delete(X, i, axis=0)
                res = minimize(lambda x, o=others: _phi_contrib(x, o, lam), X[i],
                               method="SLSQP", bounds=bounds, constraints=cons,
                               options={"maxiter": 60, "ftol": 1e-9})
                if res.success and K.is_feasible(res.x) and                   _phi_contrib(res.x, others, lam) < _phi_contrib(X[i], others, lam):
                    X[i] = res.x
    return X

def _corr_between(A, B, xi):

    diff = A[:, None, :] - B[None, :, :]
    return np.exp(-xi * (diff ** 2).sum(axis=-1))

def _corr_matrix(X, xi):
    return _corr_between(X, X, xi)

def _rank1_inv_update(Rinv, b, c):

    Rb = Rinv @ b
    s = max(c - b @ Rb, 1e-12)
    k = len(b)
    new = np.empty((k + 1, k + 1))
    new[:k, :k] = Rinv + np.outer(Rb, Rb) / s
    new[:k, k] = -Rb / s
    new[k, :k] = -Rb / s
    new[k, k] = 1.0 / s
    return new

def maxentropy_design(n, rng, xi=None):

    xi = C.GP_XI if xi is None else xi
    capped = n > C.MAXENT_MAX_N
    n = min(n, C.MAXENT_MAX_N)
    pool = K.sample_feasible(C.MAXENT_CANDIDATE_POOL, rng=rng)
    c = 1.0 + C.GP_NUGGET
    P = len(pool)

    start = int(rng.integers(P))
    chosen = [start]
    Rinv = np.array([[1.0 / c]])
    rem = np.delete(np.arange(P), start)
    while len(chosen) < n and len(rem) > 0:
        Rc = _corr_between(pool[rem], pool[chosen], xi)
        v = c - np.einsum("ij,ij->i", Rc @ Rinv, Rc)
        jl = int(np.argmax(v))
        j = int(rem[jl])
        b = _corr_between(pool[[j]], pool[chosen], xi).ravel()
        Rinv = _rank1_inv_update(Rinv, b, c)
        chosen.append(j)
        rem = np.delete(rem, jl)

    X = pool[chosen]
    R = _corr_matrix(X, xi) + C.GP_NUGGET * np.eye(len(X))
    return X, float(np.linalg.det(R)), capped

def dirichlet_design(n, rng):

    return K.sample_feasible(n, rng=rng)

def generate_design(method, n, seed=None):

    n = int(min(max(n, 2), C.MAX_POINTS))
    rng = np.random.default_rng(C.RANDOM_SEED if seed is None else seed)
    info = {"method": method, "requested_n": n}

    if method == "Latin Hypercube (Rejection)":
        X = lhd_rejection(n, rng)
    elif method == "Latin Hypercube (Transformation)":
        X = lhd_transformation(n, rng)
    elif method == "Simplex Lattice":
        X = simplex_lattice(n=n)
        info["note"] = "จำนวนจุดกำหนดโดยโครงสร้าง (degree) + ข้อจำกัด จึงไม่เท่า n พอดี"
    elif method == "Simplex Centroid":
        X = simplex_centroid(n)
        info["note"] = "Augmented SCD : centroid ของเซตย่อย + ซอยซิมเพล็กซ์ (subdivision) จนได้ feasible >= n"
    elif method == "Maximin (Morris-Mitchell)":
        X = maximin_design(n, rng)
        if n > C.MAXIMIN_REFINE_MAX_N:
            info["note"] = f"n>{C.MAXIMIN_REFINE_MAX_N}: ใช้ greedy farthest-point อย่างเดียว (ข้าม SLSQP polish)"
    elif method == "Maximum Entropy":
        X, detR, capped = maxentropy_design(n, rng)
        info["det_R"] = detR
        if capped:
            info["note"] = f"Max-Entropy จำกัดที่ {C.MAXENT_MAX_N} จุด (เกณฑ์ det(R) มีต้นทุนสูง)"
    elif method == "Dirichlet":
        X = dirichlet_design(n, rng)
    else:
        raise ValueError(f"ไม่รู้จักวิธี: {method}")

    if len(X) < n and "note" not in info:
        info["note"] = f"ได้ {len(X)}/{n} จุด (ข้อจำกัดแคบ) -- ลองผ่อนขอบล่าง/บน หรือใช้วิธีอื่น"
    info["actual_n"] = len(X)
    info["min_distance"] = min_pairwise_distance(X)
    info["phi_p"] = phi_p(X) if len(X) > 1 else float("nan")
    return to_dataframe(X), info

def to_dataframe(X):
    df = pd.DataFrame(X, columns=C.INGREDIENTS)
    df.insert(0, "Point", [f"P{i+1:04d}" for i in range(len(df))])
    return df

def save_design_xlsx(df, path=None):
    path = path or C.DESIGN_POINTS_XLSX
    df.to_excel(path, index=False, engine="openpyxl")
    return path

def make_simplex_figures(df, max_points=1500):

    import plotly.graph_objects as go
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    figs = []
    for combo in combinations(C.INGREDIENTS, 3):
        sub = df[list(combo)].copy()
        s = sub.sum(axis=1).replace(0, np.nan)
        sub = sub.div(s, axis=0).fillna(0.0)
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1],
                                i=[0], j=[1], k=[2], opacity=0.12, color="royalblue",
                                hoverinfo="skip", showscale=False))
        fig.add_trace(go.Scatter3d(x=sub[combo[0]], y=sub[combo[1]], z=sub[combo[2]],
                                   mode="markers", marker=dict(size=3, color="crimson"),
                                   text=df.get("Point"), name="points"))
        fig.update_layout(title=f"{combo[0]} : {combo[1]} : {combo[2]}",
                          scene=dict(xaxis_title=combo[0], yaxis_title=combo[1], zaxis_title=combo[2],
                                     xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1])),
                          margin=dict(l=0, r=0, t=40, b=0), height=420)
        figs.append((f"{combo[0]}-{combo[1]}-{combo[2]}", fig))
    return figs

if __name__ == "__main__":
    for meth in C.DESIGN_METHODS:
        d, info = generate_design(meth, 50)
        print(f"{meth:38s} n={info['actual_n']:4d} min_dist={info['min_distance']:.4f}"
              + ("  | " + info["note"] if "note" in info else ""))

"""
================================================================================
 main_inpho.py  --  DASHBOARD (Streamlit)
--------------------------------------------------------------------------------
 run:   streamlit run main_inpho.py
 * รันทั้ง 7 วิธี space-filling ในคลิกเดียว แล้วเปรียบเทียบด้วย D (overall desirability)
 tabs: 1)Ingredients 2)Method Comparison 3)Point Distribution
       4)Nutritional Values 5)GPR+LOOCV 6)Best Points 7)Contour
================================================================================
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import config as C
import constraints as K
import m1_nutrition as m1
import m2_design as m2
import m3_evaluate as m3
import m4_gpr_df as m4
import m5_contour as m5

st.set_page_config(page_title="Plant-Based Meat Optimizer", layout="wide", page_icon="leaf")


def df_to_xlsx_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def get_matrix():
    return m1.load_matrix()


def run_one_method(method, n, seed, matrix, water, xi, group, cost_u):
    """รัน pipeline เต็มของ 1 วิธี -> dict ผลลัพธ์ (ไม่เขียนไฟล์)"""
    design_df, info = m2.generate_design(method, int(n), seed=int(seed))
    evald = m3.evaluate_design(design_df, matrix, save=False, water_fraction=water)

    gdesign, gevald, subsampled = m4.maybe_subsample(design_df, evald, C.GPR_MAX_N, seed=int(seed))
    Xg = gdesign[C.INGREDIENTS].to_numpy(float)

    r2 = m4.loocv_r2(Xg, gevald, xi=xi) if len(Xg) >= 3 else {}
    models, best = None, None
    if len(Xg) >= 2:
        models = m4.fit_all_responses(Xg, gevald, xi=xi)
        best = m4.optimize_desirability(models, group, seed=int(seed), cost_u=cost_u)

    return dict(design=design_df, info=info, evald=evald, r2=r2,
                models=models, best=best, subsampled=subsampled, gpr_n=len(Xg))


# ================================ HEADER =====================================
st.title("Optimization of Plant-Based Meat Formulations")
st.caption("Runs all 7 space-filling methods in one click, then compares them by overall desirability (D)")

# ================================ SIDEBAR ====================================
with st.sidebar:
    st.header("Parameters")
    n_points = st.number_input(f"Maximum points (n)  [cap {C.MAX_POINTS:,} points]",
                               min_value=4, max_value=C.MAX_POINTS,
                               value=C.DEFAULT_N_POINTS, step=1)
    seed = st.number_input("Random seed", min_value=0, value=C.RANDOM_SEED, step=1)
    st.caption("Fixed seed -> reproducible designs, so the 7 methods are compared fairly.")

    st.divider()
    st.subheader("Ingredient Proportions")
    st.caption("Drag to set both lower and upper bounds for each ingredient")
    bounds_pct = {}
    for ing in C.INGREDIENTS:
        lo_def, hi_def = C.INGREDIENT_BOUNDS[ing]
        bounds_pct[ing] = st.slider(f"{ing} (%)", 0.0, 100.0,
                                    (float(lo_def * 100), float(hi_def * 100)),
                                    0.5, key=f"bd_{ing}")

    st.divider()
    st.subheader("Nutrition / Desirability")
    water = st.slider("Water fraction", 0.0, 0.80, 0.55, 0.05)
    group = st.selectbox("Product Group (DF Bounds)", list(C.PRODUCT_GROUPS.keys()))
    cost_u = st.slider("Upper Cost Bound (Baht/100g)", 5.0, 60.0, float(C.COST_U), 1.0)

    with st.expander("Advanced (GPR)"):
        xi = st.number_input("Correlation xi", min_value=0.1, value=float(C.GP_XI), step=1.0)

    run = st.button("Run Full Process (all 7 methods)", type="primary", width="stretch")

if "ran" not in st.session_state:
    st.session_state.ran = False

# ================================ RUN ========================================
if run:
    K.reset_bounds()
    for ing, (lo, hi) in bounds_pct.items():
        K.set_ingredient_min(ing, lo / 100.0)
        K.set_ingredient_max(ing, hi / 100.0)

    if sum(lo for lo, _ in bounds_pct.values()) / 100.0 > 1.0:
        st.error("Sum of lower bounds > 100% : Constraints are conflicting. Reduce the minimum values.")
        st.stop()

    matrix, is_mock = get_matrix()
    results = {}
    prog = st.progress(0.0, text="Running all methods...")
    for i, meth in enumerate(C.DESIGN_METHODS):
        prog.progress(i / len(C.DESIGN_METHODS), text=f"Running: {meth}")
        results[meth] = run_one_method(meth, n_points, seed, matrix, water, xi, group, cost_u)
    prog.progress(1.0, text="Done")
    prog.empty()

    scored = {m: (r["best"]["D"] if r["best"] else -1.0) for m, r in results.items()}
    best_method = max(scored, key=scored.get)

    matrix.to_excel(C.NUTRITION_MATRIX_XLSX, index=False)
    m2.save_design_xlsx(results[best_method]["design"])
    results[best_method]["evald"].to_excel(C.PROJECT_DATA_XLSX, index=False, engine="openpyxl")

    st.session_state.update(dict(
        ran=True, results=results, matrix=matrix, is_mock=is_mock,
        group=group, water=water, best_method=best_method, n_req=int(n_points)))

# ================================ DISPLAY ====================================
if not st.session_state.ran:
    st.info("Set the parameters on the left and click **Run Full Process** — the app runs all 7 methods and compares them.")
    st.stop()

S = st.session_state
results = S["results"]
matrix = S["matrix"]
best_method = S["best_method"]

if S["is_mock"]:
    st.warning("Did not find real data file (.zip) — Using **mock data** for demonstration. "
               "Place the zip file with 3 files next to the code to use real data")


def _mean_r2(rdict):
    vals = [v for v in rdict.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


comp_rows = []
for meth in C.DESIGN_METHODS:
    R = results[meth]
    D = R["best"]["D"] if R["best"] else np.nan
    comp_rows.append({
        "Method": meth,
        "Points": R["info"]["actual_n"],
        "Min distance": R["info"]["min_distance"],
        "Mean R2": _mean_r2(R["r2"]),
        "D (overall)": D,
    })
comp_df = pd.DataFrame(comp_rows)
bestD = results[best_method]["best"]["D"] if results[best_method]["best"] else float("nan")

st.success(f"Best method by overall desirability: **{best_method}**  ->  D = {bestD:.4f}")

best_idx = C.DESIGN_METHODS.index(best_method)

tabs = st.tabs(
    ["1) Ingredients", "2) Method Comparison", "3) Point Distribution",
     "4) Nutritional Values", "5) GPR + LOOCV", "6) Best Points", "7) Contour"])

# ---- TAB 1 ----
with tabs[0]:
    st.subheader("Ingredient Database (Nutrition Matrix) — Per 100 g")
    st.dataframe(matrix, width="stretch")
    st.download_button("Nutrition_Matrix.xlsx", df_to_xlsx_bytes(matrix), "Nutrition_Matrix.xlsx")

# ---- TAB 2 : Method Comparison ----
with tabs[1]:
    st.subheader("Comparison of the 7 Space-Filling Methods")
    st.caption("Same n, seed, bounds and product group for every method -> a fair head-to-head. "
               "Higher D = better formulation; higher Mean R2 = more reliable GPR; "
               "larger Min distance = more even coverage.")

    show = comp_df.copy()
    show["Min distance"] = show["Min distance"].round(4)
    show["Mean R2"] = show["Mean R2"].round(3)
    show["D (overall)"] = show["D (overall)"].round(4)

    def _hl(row):
        return ["background-color: #d4edda" if row["Method"] == best_method else "" for _ in row]

    st.dataframe(show.style.apply(_hl, axis=1), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    dfin = comp_df.dropna(subset=["D (overall)"])
    if not dfin.empty:
        dfin = dfin.sort_values("D (overall)")
        figD = px.bar(dfin, x="D (overall)", y="Method", orientation="h",
                      text=dfin["D (overall)"].round(4), title="Overall Desirability (D) by Method")
        figD.update_traces(marker_color="seagreen")
        c1.plotly_chart(figD, width="stretch", key="cmp_D")

    r2fin = comp_df.dropna(subset=["Mean R2"])
    if not r2fin.empty:
        r2fin = r2fin.sort_values("Mean R2")
        figR = px.bar(r2fin, x="Mean R2", y="Method", orientation="h",
                      range_x=[min(0, float(r2fin["Mean R2"].min())), 1],
                      text=r2fin["Mean R2"].round(3), title="Mean LOOCV R2 by Method")
        figR.update_traces(marker_color="steelblue")
        c2.plotly_chart(figR, width="stretch", key="cmp_R2")

    st.download_button("Method_Comparison.xlsx", df_to_xlsx_bytes(comp_df), "Method_Comparison.xlsx")

# ---- TAB 3 : Point Distribution ----
with tabs[2]:
    st.subheader("Point Distribution (Space-Filling)")
    st.caption("Pick a space-filling method to see how it spreads points, then choose a 3-in-5 view. "
               "This selector is independent of the one above, so you can browse every method's coverage.")

    dc1, dc2 = st.columns([1, 1])
    dist_method = dc1.selectbox("1) Space-filling method", C.DESIGN_METHODS,
                                index=best_idx, key="dist_method")
    VD = results[dist_method]
    info = VD["info"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of points (actual)", info["actual_n"])
    c2.metric("Min pairwise distance", f"{info['min_distance']:.4f}")
    if "det_R" in info:
        c3.metric("det(R) (Max-Entropy)", f"{info['det_R']:.2e}")
    else:
        c3.metric("Acceptance rate", f"{K.acceptance_rate():.1%}")

    if "note" in info:
        st.caption("Note: " + info["note"])
    if info["actual_n"] < 8:
        st.warning("Few points — GPR/LOOCV may not be reliable (e.g., Simplex Centroid).")

    st.dataframe(VD["design"].round(4), width="stretch", height=260)
    st.download_button("Design_Points.xlsx", df_to_xlsx_bytes(VD["design"]),
                       f"Design_Points_{dist_method}.xlsx", key="dl_design")

    st.markdown("Simplex 3 dimensional (3 in 5 proportions)")
    figs = m2.make_simplex_figures(VD["design"])
    labels = [t for t, _ in figs]
    pick = st.selectbox("2) Select a set of 3 components", ["(Show all 10 plots)"] + labels,
                        key="simplex_pick")
    if pick == "(Show all 10 plots)":
        cols = st.columns(2)
        for i, (_, fig) in enumerate(figs):
            cols[i % 2].plotly_chart(fig, width="stretch", key=f"simplex_all_{dist_method}_{i}")
    else:
        st.plotly_chart(figs[labels.index(pick)][1], width="stretch", key=f"simplex_one_{dist_method}")

# ---- TAB 4 : Nutritional Values ----
with tabs[3]:
    vm = st.selectbox("Space-filling method", C.DESIGN_METHODS, index=best_idx, key="vm_nutri")
    V = results[vm]
    st.subheader(f"Project Data — {vm} — Per 100 g After Processing + Water")
    st.caption(f"Water fraction = {S['water']:.2f}  |  Processing loss: Protein 10%, Fat 15%, Carb 5%")
    st.dataframe(V["evald"].drop(columns=["Energy_out (kcal)"]).round(3), width="stretch", height=320)
    st.download_button("Project_Data.xlsx", df_to_xlsx_bytes(V["evald"]),
                       "Project_Data.xlsx", key="dl_project")

# ---- TAB 5 : GPR + LOOCV ----
with tabs[4]:
    vm = st.selectbox("Space-filling method", C.DESIGN_METHODS, index=best_idx, key="vm_gpr")
    V = results[vm]
    st.subheader(f"Accuracy of GPR with LOOCV (R2) — {vm}")
    if V.get("subsampled"):
        st.info(f"Many design points — GPR/LOOCV/DF use a subsample of {V['gpr_n']:,} points "
                f"(kriging is O(n^3), limited to {C.GPR_MAX_N:,})")
    r2 = V["r2"]
    if not r2:
        st.error("Few points — LOOCV may not be reliable")
    else:
        cols = st.columns(len(r2))
        for col, (k, v) in zip(cols, r2.items()):
            col.metric(f"R2 {k}", f"{v:.3f}")
        r2df = pd.DataFrame({"Response": list(r2.keys()), "R2": list(r2.values())})
        fig = px.bar(r2df, x="Response", y="R2",
                     range_y=[min(0, float(r2df["R2"].min())), 1], text=r2df["R2"].round(3))
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Good criterion 0.8")
        st.plotly_chart(fig, width="stretch", key=f"loocv_bar_{vm}")

# ---- TAB 6 : Best Points ----
with tabs[5]:
    vm = st.selectbox("Space-filling method", C.DESIGN_METHODS, index=best_idx, key="vm_best")
    V = results[vm]
    st.subheader(f"Best Points — {vm} — Group: {S['group']}")
    best = V["best"]
    if best is None:
        st.error("Few points — not enough for optimization")
    else:
        fb = m4.format_best(best)
        D = fb["D"]
        if D <= 1e-9:
            st.error("D = 0 : At least one objective is out of bounds (d=0). "
                     "If water fraction = 0, try increasing it to ~0.55")
        st.metric("Overall Desirability  D", f"{D:.4f}")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Best Proportions of Ingredients (%)**")
            form_df = pd.DataFrame({"Ingredient": list(fb["formulation_%"].keys()),
                                    "Proportion (%)": list(fb["formulation_%"].values())})
            st.plotly_chart(px.pie(form_df, names="Ingredient", values="Proportion (%)", hole=0.35),
                            width="stretch", key=f"best_pie_{vm}")
            st.dataframe(form_df, width="stretch", hide_index=True)
        with cB:
            st.markdown("**Individual Desirability  (d)**")
            d_df = pd.DataFrame({"Response": list(fb["individual_d"].keys()),
                                 "d": list(fb["individual_d"].values())})
            st.plotly_chart(px.bar(d_df, x="Response", y="d", range_y=[0, 1], text=d_df["d"]),
                            width="stretch", key=f"best_dbar_{vm}")
            st.markdown("**Predicted Values (response)**")
            resp_df = pd.DataFrame({
                "Response": ["Energy (kJ)", "Protein (g)", "Fat (g)", "Carb (g)", "Cost (Baht)"],
                "Value": [fb["responses"][k] for k in ["Energy", "Protein", "Fat", "Carb", "Cost"]]})
            st.dataframe(resp_df, width="stretch", hide_index=True)

        st.success("Best formulation: " +
                   ", ".join(f"{k} {v}%" for k, v in fb["formulation_%"].items()) + f"  ->  D = {D:.4f}")

# ---- TAB 7 : Contour ----
with tabs[6]:
    vm = st.selectbox("Space-filling method", C.DESIGN_METHODS, index=best_idx, key="vm_contour")
    V = results[vm]
    st.subheader(f"Contour of Responses (3 of 5 Proportions) — {vm}")
    models, best = V["models"], V["best"]
    if models is None or best is None:
        st.error("GPR model and optimized point are required (too few points)")
    else:
        cc1, cc2 = st.columns(2)
        resp = cc1.selectbox("Select Response", ["Protein", "Energy", "Fat", "Carb", "Cost"],
                             key=f"cont_resp_{vm}")
        figs_c = m5.make_contour_figures(models, resp, np.asarray(best["x"], float))
        labels_c = [t for t, _ in figs_c]
        pick_c = cc2.selectbox("Select Set of 3 Components", ["(Show All 10 Figures)"] + labels_c,
                               key=f"cont_pick_{vm}")
        if pick_c == "(Show All 10 Figures)":
            cols = st.columns(2)
            for i, (_, fig) in enumerate(figs_c):
                cols[i % 2].plotly_chart(fig, width="stretch", key=f"contour_all_{vm}_{resp}_{i}")
        else:
            st.plotly_chart(figs_c[labels_c.index(pick_c)][1], width="stretch",
                            key=f"contour_one_{vm}_{resp}")

import numpy as np
from itertools import combinations

import config as C
import m4_gpr_df as m4

GREEN_RED = [[0.0, "rgb(0,130,60)"], [0.25, "rgb(150,200,0)"], [0.5, "rgb(255,215,0)"],
             [0.75, "rgb(255,140,0)"], [1.0, "rgb(200,0,0)"]]

RESPONSE_LABEL = {"Protein": "Protein (g)", "Energy": "Energy (kJ)",
                  "Fat": "Fat (g)", "Carb": "Carb (g)", "Cost": "Cost (บาท)"}

_H = np.sqrt(3.0) / 2.0

def make_contour_figures(models, response, fixed_x, res=64):

    import plotly.graph_objects as go

    model = models[response]
    fixed_x = np.asarray(fixed_x, float)

    xs = np.linspace(0.0, 1.0, res)
    ys = np.linspace(0.0, _H, res)
    XX, YY = np.meshgrid(xs, ys)
    cc = YY / _H
    bb = XX - 0.5 * cc
    aa = 1.0 - bb - cc
    inside = (aa >= -1e-9) & (bb >= -1e-9) & (cc >= -1e-9)

    ai, bi, ci = aa[inside], bb[inside], cc[inside]
    figs = []
    for combo in combinations(range(C.D), 3):
        others = [i for i in range(C.D) if i not in combo]
        mass = max(1.0 - float(fixed_x[others].sum()), 0.0)

        Xfull = np.tile(fixed_x, (ai.size, 1))
        Xfull[:, combo[0]] = ai * mass
        Xfull[:, combo[1]] = bi * mass
        Xfull[:, combo[2]] = ci * mass
        vals = m4.gpr_predict(model, Xfull)

        Z = np.full(XX.shape, np.nan)
        Z[inside] = vals

        names = [C.INGREDIENTS[i] for i in combo]
        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=Z, colorscale=GREEN_RED, connectgaps=False,
            contours=dict(coloring="fill", showlines=True),
            colorbar=dict(title=RESPONSE_LABEL[response], thickness=14)))

        fig.add_trace(go.Scatter(x=[0, 1, 0.5, 0], y=[0, 0, _H, 0], mode="lines",
                                 line=dict(color="black", width=1), hoverinfo="skip",
                                 showlegend=False))

        for (vx, vy), nm in zip([(0, 0), (1, 0), (0.5, _H)], names):
            fig.add_annotation(x=vx, y=vy, text=f"<b>{nm}</b>", showarrow=False,
                               font=dict(size=13), yshift=(14 if vy > 0 else -14))
        fig.update_xaxes(visible=False, range=[-0.08, 1.08])
        fig.update_yaxes(visible=False, scaleanchor="x", range=[-0.1, _H + 0.1])
        fig.update_layout(title=f"{RESPONSE_LABEL[response]} : {'-'.join(names)}",
                          height=430, margin=dict(l=10, r=10, t=40, b=10))
        figs.append(("-".join(names), fig))
    return figs

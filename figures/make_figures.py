"""Make every data figure of the paper from the experiment CSVs in ./data.

python make_figures.py            -> figs/*.pdf (vector, for LaTeX) and figs/*.png (previews)

All numbers come from the Kaggle runs (Exps 1-5 and the full-val runs). Nothing is synthesised.
Sizes follow the CVPR template: one column = 3.25 in, full width = 6.875 in.
"""
import os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
D, OUT = os.path.join(HERE, "..", "results"), os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
COL, FULL = 3.25, 6.875

# Okabe-Ito colour-blind-safe palette, one colour per design family
C = {"DETR": "#0072B2", "two-stage CNN": "#D55E00", "one-stage CNN": "#009E73", "plain ViT": "#CC79A7",
     "open-vocab.": "#E69F00", "fixed-size": "#999999", "fix": "#000000"}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NAME = {"detr_r50": "DETR", "conditional_detr": "Cond. DETR", "deformable_detr": "Deform. DETR",
        "frcnn_v2": "Faster R-CNN", "maskrcnn_v2": "Mask R-CNN", "retinanet_v2": "RetinaNet", "fcos": "FCOS",
        "yolos_small": "YOLOS-S", "grounding_dino_tiny": "G-DINO", "rtdetr_r50": "RT-DETR",
        "rtdetr_v2_r50": "RT-DETRv2", "mask2former_swin_s": "Mask2Former"}
FAM = {"detr_r50": "DETR", "conditional_detr": "DETR", "deformable_detr": "DETR", "frcnn_v2": "two-stage CNN",
       "maskrcnn_v2": "two-stage CNN", "retinanet_v2": "one-stage CNN", "fcos": "one-stage CNN",
       "yolos_small": "plain ViT", "grounding_dino_tiny": "open-vocab.", "rtdetr_r50": "fixed-size",
       "rtdetr_v2_r50": "fixed-size", "mask2former_swin_s": "fixed-size"}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=300); plt.close(fig)
    print("wrote", name)

S = pd.concat([pd.read_csv(f"{D}/summary_group1.csv"), pd.read_csv(f"{D}/summary_group2.csv")], ignore_index=True)
X = pd.read_csv(f"{D}/exp_extracted.csv", comment="#",
                names=["source", "experiment", "model", "variant", "setting", "metric", "value"])
rand = S[(S.variant == "none") & S.batching.str.startswith("bs8_rand")]
per_model = rand.groupby("model").agg(img=("images_changed", "mean"), det=("detections_changed", "mean"),
                                      dmin=("dAP", "min"), dmax=("dAP", "max")).reset_index()
per_model["img"] *= 100; per_model["det"] *= 100

# ---------------------------------------------------------------- Fig: churn across 12 detectors
def fig_churn():
    d = per_model.copy(); d["fam"] = d.model.map(FAM)
    d = d.sort_values("img", ascending=True)
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    y = np.arange(len(d))
    ax.barh(y, d.img, color=[C[f] for f in d.fam], height=0.72, label="images changed")
    ax.barh(y, d.det, color="white", alpha=0.55, height=0.32, label="detections changed")
    for yi, (_, r) in zip(y, d.iterrows()):
        txt = f"{r.img:.1f}%" if r.img >= 0.05 else "0.0%"
        ax.text(r.img + 1.2, yi, txt, va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in d.model])
    ax.set_xlim(0, 112); ax.set_xlabel("% of COCO val images whose detections change\n(random batches of 8 vs. batch 1)")
    handles = [mpl.patches.Patch(color=C[f], label=f) for f in ["DETR", "two-stage CNN", "one-stage CNN", "plain ViT",
                                                                  "open-vocab.", "fixed-size"]]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="0.5", lw=0.5, label="share of detections changed"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, handlelength=0.9,
              columnspacing=0.6, fontsize=5.4)
    ax.axvline(0, c="k", lw=0.6)
    save(fig, "fig_churn")

# ---------------------------------------------------------------- Fig: AP hides churn
def fig_ap_vs_churn():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw={"width_ratios": [4, 1.1], "wspace": 0.08})
    for _, r in per_model.iterrows():
        pts = rand[rand.model == r.model]
        tgt = ax2 if r.model == "yolos_small" else ax
        tgt.plot(pts.images_changed * 100, pts.dAP, "o", color=C[FAM[r.model]], ms=3.2, alpha=0.9)
        off = {"frcnn_v2": (4, 4), "maskrcnn_v2": (4, -7), "retinanet_v2": (-34, 2), "conditional_detr": (4, -2),
               "deformable_detr": (4, -3), "detr_r50": (4, 1), "fcos": (-18, -8), "grounding_dino_tiny": (-8, -10)}
        if r.model in off:
            ax.annotate(NAME[r.model], (r.img, (r.dmin + r.dmax) / 2), fontsize=5.3, xytext=off[r.model],
                        textcoords="offset points")
    ax.annotate("3 fixed-size\ncontrols", (0, 0), fontsize=5.3, xytext=(4, 6), textcoords="offset points")
    ax.axhline(0, c="0.6", lw=0.6, ls="--")
    ax.set_xlim(-3, 60); ax.set_ylim(-0.45, 0.3)
    ax.set_xlabel("% images changed"); ax.set_ylabel(r"$\Delta$AP (batch 8 $-$ batch 1)")
    ax2.set_xlim(94, 100); ax2.set_ylim(-37, -32); ax2.yaxis.tick_right(); ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(False); ax2.set_xticks([95, 99])
    ax2.annotate("YOLOS-S\n36.1→1.1 AP", (97.6, -35), fontsize=5.3, xytext=(-30, 8), textcoords="offset points")
    ax.set_title("Large per-image churn, small or systematic AP shift", loc="left")
    save(fig, "fig_ap_vs_churn")

# ---------------------------------------------------------------- Fig: the padding switch (Exp 2 sweep)
def fig_switch():
    sw = pd.read_csv(f"{D}/A_sweep.csv")
    g = sw.groupby(["model", "variant", "px"]).flip.mean().reset_index()
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    col = {"detr_r50": C["DETR"], "frcnn_v2": C["two-stage CNN"], "retinanet_v2": C["one-stage CNN"]}
    for m in ["detr_r50", "frcnn_v2", "retinanet_v2"]:
        for v, ls, mk in [("none", "-", "o"), ("full", "--", "s")]:
            d = g[(g.model == m) & (g.variant == v)].sort_values("px")
            x0 = [0] if m == "detr_r50" else []      # torchvision canvases are multiples of 32: 32 px is the smallest step
            ax.plot(np.r_[x0, d.px], np.r_[[0] * len(x0), d.flip * 100], ls, marker=mk, color=col[m],
                    label=f"{NAME[m]}" + (" + fix" if v == "full" else ""), mfc="white" if v == "full" else col[m])
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, 1, 4, 16, 64, 256]); ax.set_xticklabels(["0", "1", "4", "16", "64", "256"])
    ax.set_xlabel("padding added to the right and bottom (pixels)"); ax.set_ylabel("% images changed")
    ax.annotate("1 px changes\n42% of DETR outputs", (1, 42.5), xytext=(-8, 16), textcoords="offset points", fontsize=5.8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    ax.set_ylim(-2, 60); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), columnspacing=0.8, handlelength=1.8)
    save(fig, "fig_switch")

# ---------------------------------------------------------------- Fig: local vs global leak (Exp 1 B2)
def fig_locality():
    bd = pd.read_csv(f"{D}/B_per_detection.csv")
    bins = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.01]; mids = [0.025, 0.075, 0.15, 0.275, 0.425, 0.75]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6), sharey=True)
    col = {"detr_r50": C["DETR"], "deformable_detr": "#56B4E9", "frcnn_v2": C["two-stage CNN"]}
    for ax, test, dcol, lab in [(axes[0], "B1_right", "d_right", "right"), (axes[1], "B1_bottom", "d_bottom", "bottom")]:
        for m in ["frcnn_v2", "detr_r50", "deformable_detr"]:
            d = bd[(bd.model == m) & (bd.test == test)].copy()
            d["bin"] = pd.cut(d[dcol], bins, right=False)
            gg = d.groupby("bin", observed=True).changed.agg(["mean", "size"])
            ax.plot(mids[:len(gg)], gg["mean"] * 100, "o-", color=col[m], label=NAME[m])
        ax.set_xlabel(f"distance to {lab} edge"); ax.set_title(f"{lab} padding (25%)")
    axes[0].set_ylabel("% detections changed"); axes[0].legend(loc="upper right")
    save(fig, "fig_locality")

# ---------------------------------------------------------------- Fig: four coupled leaks (Exp 2 attribution)
def fig_attribution():
    a = X[X.experiment == "exp2_attribution"].copy(); a["value"] = a.value.astype(float)
    models = ["detr_r50", "frcnn_v2", "retinanet_v2", "fcos"]
    order = ["none", "exactmask", "norm", "strides", "conv", "full"]
    lab = {"none": "no fix", "exactmask": "exact mask only", "norm": "masked GroupNorm only", "strides": "anchor stride only",
           "conv": "zeroed padding only", "full": "all together"}
    col = {"none": "0.75", "exactmask": "#56B4E9", "norm": "#009E73", "strides": "#E69F00", "conv": "#D55E00", "full": "#000000"}
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    w = 0.14
    for k, v in enumerate(order):
        ys = [a[(a.model == m) & (a.variant == v)].value.max() if ((a.model == m) & (a.variant == v)).any() else np.nan for m in models]
        xs = np.arange(len(models)) + (k - 2.5) * w
        ax.bar(xs, ys, w, color=col[v], label=lab[v])
        if v == "full":
            for x in xs:
                ax.text(x, 2, "0", ha="center", fontsize=5.5, fontweight="bold")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models])
    ax.set_ylabel("% images changed\n(batch 8 vs. batch 1, same setting)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.32), columnspacing=0.8, handlelength=1.0)
    ax.set_ylim(0, 100)
    save(fig, "fig_attribution")

# ---------------------------------------------------------------- Fig: remedies (full val)
def fig_remedies():
    models = ["detr_r50", "conditional_detr", "frcnn_v2", "maskrcnn_v2", "retinanet_v2", "fcos"]
    pol = [("none", "bs1", "batch 1", "o", "white"), ("none", "bs8_rand_s0", "batch 8 (default)", "X", None),
           ("self", "bs8_rand_s0", "batch 8 + our fix", "D", "k"), ("none", "buckets", "exact-shape buckets", "s", "0.5")]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for m in models:
        d = S[S.model == m].set_index(["variant", "batching"]); t1 = d.loc[("none", "bs1"), "img_per_s"]
        xs, ys = [], []
        for v, b, lab, mk, fc in pol:
            x = d.loc[(v, b), "img_per_s"] / t1; y = 0.0 if b == "bs1" else d.loc[(v, b), "images_changed"] * 100
            xs.append(x); ys.append(y)
            ax.plot(x, y, mk, color=C[FAM[m]], mfc=fc if fc else C[FAM[m]], ms=4.2, mew=0.7)
    for v, b, lab, mk, fc in pol:
        ax.plot([], [], mk, color="0.3", mfc=fc if fc else "0.3", label=lab)
    ax.set_xlabel("throughput relative to batch 1 (T4, fp32, full resolution)"); ax.set_ylabel("% images changed")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55)); ax.set_ylim(-3, 56); ax.set_xlim(0.5, 1.05)
    ax.text(0.62, 49, "default batching:\nslower and inexact", fontsize=5.6, ha="center", color="0.3")
    ax.text(0.62, 4, "our fix: exact", fontsize=5.6, ha="center", color="0.3")
    ax.text(0.955, 4, "buckets / batch 1:\nexact", fontsize=5.6, ha="center", color="0.3")
    save(fig, "fig_remedies")

# ---------------------------------------------------------------- Fig: YOLOS collapse
def fig_yolos():
    y = S[S.model == "yolos_small"].set_index(["variant", "batching"])
    vals = [y.loc[("none", "bs1"), "AP"], y.loc[("none", "bs8_rand_s0"), "AP"], y.loc[("rescale", "bs8_rand_s0"), "AP"]]
    labs = ["bs 1", "bs 8", "bs 8 +\nrescale"]
    fig, ax = plt.subplots(figsize=(1.55, 1.75))
    bars = ax.bar(range(3), vals, color=[C["plain ViT"], "#e7b8d8", C["plain ViT"]], width=0.65)
    bars[2].set_hatch("////"); bars[2].set_edgecolor("white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=6.3)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=5.8); ax.set_ylabel("COCO box AP"); ax.set_ylim(0, 42)
    ax.set_title("YOLOS-S", loc="left")
    save(fig, "fig_yolos")

# ---------------------------------------------------------------- Fig: auto-labelling disagreement
def fig_autolabel():
    a = X[(X.experiment == "exp5_autolabel") & (X.metric == "images_with_different_labels")].copy(); a["value"] = a.value.astype(float)
    groups = [("frcnn_v2", "none", "F-RCNN", C["two-stage CNN"]),
              ("frcnn_v2", "self", "F-RCNN\n+ fix", "k"),
              ("grounding_dino_tiny", "none", "G-DINO", C["open-vocab."])]
    fig, ax = plt.subplots(figsize=(1.75, 1.75))
    for i, (m, v, lab, c) in enumerate(groups):
        vals = a[(a.model == m) & (a.variant == v)].value.values
        ax.bar(i, vals.mean(), color=c, width=0.62)
        ax.plot([i] * len(vals), vals, "o", color="k", ms=2, mfc="white", mew=0.5)
        ax.text(i, vals.max() + 1.2, f"{vals.mean():.1f}%", ha="center", fontsize=6.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[2] for g in groups], fontsize=6)
    ax.set_ylabel("% images whose pseudo-labels\ndiffer after re-shuffling"); ax.set_ylim(0, 38)
    ax.set_title("Auto-labelling", loc="left")
    save(fig, "fig_autolabel")

# ---------------------------------------------------------------- Supp: online serving
def fig_serving():
    s = X[X.experiment == "exp5_serving"].copy(); s["value"] = s.value.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True)
    mk = {"bs1": "o", "dynamic": "X", "buckets": "s", "dynamic+self": "D"}
    for ax, st in zip(axes, ["coco", "diverse"]):
        for p in mk:
            t = s[(s.setting == st) & (s.variant == p) & (s.metric == "throughput")].value.iloc[0]
            f = s[(s.setting == st) & (s.variant == p) & (s.metric == "answers_changed")].value.iloc[0]
            ax.plot(t, f, mk[p], ms=5, color="k" if p != "dynamic" else "#D55E00", mfc="white" if p == "bs1" else None, label=p)
        ax.set_title(f"request stream: {st}"); ax.set_xlabel("throughput (req/s)")
    axes[0].set_ylabel("% answers changed"); axes[1].legend(loc="center right")
    save(fig, "fig_serving")

# ---------------------------------------------------------------- Supp: per-condition breakdown (Exp 1, 500 images)
def fig_conditions():
    a = pd.read_csv(f"{D}/A_summary.csv")
    a = a[a.condition.isin(["bs8_sorted", "bs8_rand_s0", "bs8_maxpad"])]
    models = ["detr_r50", "deformable_detr", "frcnn_v2", "maskrcnn_v2", "retinanet_v2", "fcos", "rtdetr_r50"]
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 1.7))
    w = 0.26
    for k, (c, lab, col) in enumerate([("bs8_sorted", "shape-sorted batches", "#8ecae6"),
                                       ("bs8_rand_s0", "random batches", "#219ebc"), ("bs8_maxpad", "max-padding batches", "#023047")]):
        ys = [a[(a.model == m) & (a.condition == c)].flip_rate.mean() * 100 for m in models]
        ax.bar(np.arange(len(models)) + (k - 1) * w, ys, w, color=col, label=lab)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], rotation=15); ax.set_ylabel("% images changed")
    ax.legend(ncol=3, loc="upper left"); ax.set_ylim(0, 65)
    ax.text(6, 1.5, "0 (control)", ha="center", fontsize=5.8)
    save(fig, "fig_conditions")

if __name__ == "__main__":
    for f in [fig_churn, fig_ap_vs_churn, fig_switch, fig_locality, fig_attribution, fig_remedies, fig_yolos,
              fig_autolabel, fig_serving, fig_conditions]:
        f()


# ================================================================ Section 4 additions (full val, per image)
def _per_image():
    P = pd.concat([pd.read_csv(f"{D}/per_image_group1.csv"), pd.read_csv(f"{D}/per_image_group2.csv")], ignore_index=True)
    return P[(P.variant == "none") & P.batching.str.startswith("bs8_rand")]

MCOL = {"detr_r50": "#0072B2", "conditional_detr": "#3A9AD9", "deformable_detr": "#8CC5EB",
        "frcnn_v2": "#D55E00", "maskrcnn_v2": "#F09A62", "retinanet_v2": "#009E73", "fcos": "#66C2A5",
        "yolos_small": "#CC79A7", "grounding_dino_tiny": "#E69F00", "rtdetr_r50": "#7F7F7F",
        "rtdetr_v2_r50": "#B0B0B0", "mask2former_swin_s": "#D0D0D0"}

def fig_anatomy():
    R = _per_image()
    g = R.groupby("model").agg(img=("flip", "mean"), stable=("stable", "sum"), moved=("moved", "sum"),
                               lost=("lost", "sum"), gained=("gained", "sum"))
    base = g.stable + g.moved + g.lost
    for c in ["moved", "lost", "gained"]:
        g[c] = 100 * g[c] / base
    g["img"] *= 100
    order = g.sort_values("img").index.tolist()
    fig = plt.figure(figsize=(FULL, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.42)
    y = np.arange(len(order))
    # (a) images changed
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y, g.loc[order, "img"], color=[MCOL[m] for m in order], height=0.72)
    for yi, m in zip(y, order):
        ax.text(g.loc[m, "img"] + 1.5, yi, f"{g.loc[m, 'img']:.1f}", va="center", fontsize=5.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in order]); ax.set_xlim(0, 112)
    ax.set_xlabel("% images changed"); ax.set_title("(a) images whose output changes", loc="left")
    # (b) composition of the change
    ax = fig.add_subplot(gs[0, 1])
    left = np.zeros(len(order))
    for c, col, lab in [("moved", "#56B4E9", "moved (IoU < 0.9)"), ("lost", "#D62728", "lost"), ("gained", "#FFB000", "new")]:
        v = g.loc[order, c].values
        ax.barh(y, np.minimum(v, 40 - left), left=left, color=col, height=0.72, label=lab)
        left = np.minimum(left + v, 40)
    yi = order.index("yolos_small")
    ax.text(39, yi, "YOLOS: 11 / 87 / 84 →", ha="right", va="center", fontsize=5.4, color="white", weight="bold")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 40)
    ax.set_xlabel("% of batch-1 detections"); ax.set_title("(b) what happens to detections", loc="left")
    ax.legend(loc="lower right", handlelength=1.0)
    # (c) how large the change is: survival of the per-image max score change
    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0, 1, 201)
    for m in ["yolos_small", "detr_r50", "conditional_detr", "deformable_detr", "frcnn_v2", "maskrcnn_v2",
              "retinanet_v2", "fcos", "grounding_dino_tiny", "rtdetr_r50", "rtdetr_v2_r50"]:
        v = R[R.model == m].max_score_delta.values
        ax.plot(xs, [(v > x).mean() * 100 for x in xs], color=MCOL[m], lw=1.1,
                ls="--" if m.startswith("rtdetr") else "-", label=NAME[m])
    ax.set_xlabel("largest score change in the image"); ax.set_ylabel("% images exceeding it")
    ax.set_yscale("log"); ax.set_ylim(0.05, 100); ax.set_xlim(0, 1)
    ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_title("(c) how large the change is", loc="left")
    ax.text(0.98, 0.97, "colours as in (a)\ncontrols: at 0", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color="0.35")
    save(fig, "fig_anatomy")

def fig_fragility():
    R = _per_image()
    piv0 = R[R.batching == "bs8_rand_s0"].pivot_table(index="image_id", columns="model", values="flip", aggfunc="first").astype(float)
    piv1 = R[R.batching == "bs8_rand_s1"].pivot_table(index="image_id", columns="model", values="flip", aggfunc="first").astype(float)
    models = ["detr_r50", "conditional_detr", "deformable_detr", "frcnn_v2", "maskrcnn_v2", "retinanet_v2", "fcos",
              "yolos_small", "grounding_dino_tiny"]
    M = np.zeros((len(models), len(models)))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            M[i, j] = np.corrcoef(piv0[a], (piv1 if i == j else piv0)[b])[0, 1]
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if M[i, j] > 0.55 else "black", weight="bold" if i == j else None)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], rotation=45, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NAME[m] for m in models])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("correlation (φ) of 'image changed'", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("Diagonal: same model, two different shuffles\noff-diagonal: two models, same shuffle", fontsize=6.3, loc="left")
    save(fig, "fig_fragility")
    # numbers quoted in the text
    out = []
    for m in models:
        a, b = piv0[m].astype(bool), piv1[m].astype(bool)
        out.append({"model": m, "p_changed": a.mean(), "p_again_given_changed": (a & b).sum() / max(a.sum(), 1),
                    "phi_shuffles": np.corrcoef(a, b)[0, 1]})
    pd.DataFrame(out).to_csv(f"{D}/fragility_stats.csv", index=False)

if __name__ == "__main__":
    fig_anatomy(); fig_fragility()


def fig_batchsize():
    A = pd.read_csv(f"{D}/exp7_batchsize.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True)
    col = {"detr_r50": C["DETR"], "frcnn_v2": C["two-stage CNN"], "yolo11m": "#56B4E9"}
    lab = {"detr_r50": "DETR", "frcnn_v2": "Faster R-CNN", "yolo11m": "YOLO11m"}
    for m in ["detr_r50", "frcnn_v2", "yolo11m"]:
        d = A[(A.model == m) & (A.variant == "none")].sort_values("batch")
        axes[0].plot(d.batch, d.images_changed * 100, "o-", color=col[m], label=lab[m])
        e = d[d.batch > 1]
        axes[1].plot(e.batch, e.changed_among_padded * 100, "o-", color=col[m])
        f = A[(A.model == m) & (A.variant == "self")]
        if len(f):
            axes[0].plot(f.batch, f.images_changed * 100, "D", color=col[m], mfc="white", ms=4)
    for ax, t in [(axes[0], "all images"), (axes[1], "padded images only")]:
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("batch size"); ax.set_title(t); ax.set_ylim(-2, 55)
    axes[0].set_ylabel("% images changed"); axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=5.5)
    axes[0].text(32, 4, "with fix", fontsize=5.3, ha="right")
    save(fig, "fig_batchsize")

if __name__ == "__main__":
    fig_batchsize()

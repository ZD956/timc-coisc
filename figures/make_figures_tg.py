"""Redraw Figures 1-3 of the Technical Gazette manuscript (black-and-white print) from the released summary results in ../results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
FD = RES / "figures_data"
SCR = RES / "screening"
CL = RES / "closedloop"

SEEDS = [20260922, 20260923, 20260924]
T = np.arange(10, 121, 10)
TAU_P, G = 39.1, 10.0
W2 = 6.9   # two-column figure width (in)
W1 = 3.3   # one-column figure width (in)
PANEL_B_PLANTS = [0, 5, 11, 8, 20, 23]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.formatter.use_locale": False,
})


def comma(ax, axis="both", fmt="{:g}"):
    """Decimal commas on tick labels, as required by the journal."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: fmt.format(v).replace(".", ",").replace("-", "−"))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def save(fig, name):
    for ext in ("png", "tif"):
        fig.savefig(OUT / f"{name}.{ext}", **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
    plt.close(fig)
    print("saved", name)


def fig1():
    qd = pd.read_csv(FD / "fig2_holdout_Tp_sensitivity_quantiles.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    check = []
    for n, (ax, seed) in enumerate(zip(axes, SEEDS)):
        ax.axvspan(0, TAU_P - G, color="0.85", lw=0)
        ax.axvspan(TAU_P - G, TAU_P + G, color="0.94", lw=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(TAU_P, color="k", ls=":", lw=0.8)
        for var, style in (("R0", dict(color="k", ls="-", marker="o", mfc="k")),
                           ("COISC", dict(color="0.35", ls="--", marker="s", mfc="white"))):
            g = qd[(qd.seed == seed) & (qd.variant == var)].sort_values("t_s")
            q = g[["q25_S_Tp", "q50_S_Tp", "q75_S_Tp"]].to_numpy()
            ax.fill_between(T, q[:, 0], q[:, 2], color="0.5", alpha=0.18 if var == "R0" else 0.0,
                            hatch=None if var == "R0" else "////", edgecolor="0.55", lw=0)
            ax.plot(T, q[:, 1], ms=3, mew=0.7, label=var, **style)
            check.append((seed, var, round(q[1, 1], 4)))
        ax.set_title(f"Seed {n + 1}")
        ax.set_xlim(5, 125)
        ax.set_xticks([10, 30, 50, 70, 90, 110])
        ax.set_xlabel("Lead time $t$ / s")
        comma(ax, "y")
    ymin, ymax = axes[0].get_ylim()
    for ax in axes:
        ax.text(11, ymax - 0.10 * (ymax - ymin), "pre-lag", fontsize=6.5)
        ax.text(TAU_P + 1.5, ymin + 0.05 * (ymax - ymin), "prior lag", fontsize=6.5)
        ax.text(78, ymax - 0.10 * (ymax - ymin), "post-lag", fontsize=6.5)
    axes[0].set_ylabel("$S_{T_p}$ / °C per full scale")
    axes[2].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig1")
    print("  median S at 20 s:", check)


def fig2():
    res = json.loads((SCR / "SCREENING_RESULTS.json").read_text(encoding="utf-8"))
    per, wrong = res["per_seed"], res["result"]["wrong_per_seed"]
    thr = res["result"]["threshold_rule"]["threshold"]
    x = np.array([per[s]["Tp_MAE_120"] for s in per])
    y = np.array([per[s]["Tp_WSR_ONSET"] for s in per])
    pos = np.array([wrong[s] > 0 for s in per])
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.scatter(x[~pos], y[~pos], s=14, facecolors="white", edgecolors="k", linewidths=0.7,
               label=f"No reversed move (n = {int((~pos).sum())})")
    ax.scatter(x[pos], y[pos], s=22, marker="^", color="k",
               label=f"Reversed first move (n = {int(pos.sum())})")
    ax.axhline(thr, color="k", ls="--", lw=0.8)
    ax.text(x.max(), thr - 0.003, "threshold fixed in advance", ha="right", va="top", fontsize=6.5)
    auc = res["result"]["auc"]
    ax.text(0.98, 0.40, "AUC: WSR {:.2f}, MAE {:.2f}".format(auc["Tp_WSR_ONSET"], auc["Tp_MAE_120"]).replace(".", ","),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.set_xlabel("R0 holdout $T_p$ MAE at 120 s / °C")
    ax.set_ylabel("R0 holdout $T_p$ WSR")
    comma(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig2")
    assert int(pos.sum()) == res["result"]["n_positive"]


def seed_level_panel_b(d, wrong_pairs, a, b):
    sub = d[d.plant_index.isin(PANEL_B_PLANTS)]
    p = sub.pivot_table(index=["seed", "scenario", "plant_index", "family"], columns="variant", values="MAE_track").reset_index()
    p = p[[(s, sc) not in wrong_pairs for s, sc in zip(p.seed, p.scenario)]].copy()
    p["e"] = p[a] - p[b]
    pair = p.groupby(["seed", "scenario", "family"]).e.mean().groupby(["seed", "scenario"]).mean()
    return pair.groupby("seed").mean()


def fig3():
    res = json.loads((CL / "CLOSEDLOOP_RESULTS.json").read_text(encoding="utf-8"))
    d = pd.read_csv(CL / "CLOSEDLOOP_RUN_METRICS.csv")
    wp = {(w["seed"], w["scenario"]) for w in res["wrong_pairs"]}
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.1), gridspec_kw=dict(width_ratios=[1.0, 1.15, 0.85, 0.85]))

    ax = axes[0]
    pa = res["panel_A"]
    xs = np.arange(len(res["wrong_pairs"]))
    ax.bar(xs - 0.19, pa["COISC - R0"]["per_pair"], 0.38, color="k", label="COISC − R0")
    ax.bar(xs + 0.19, pa["L0 - R0"]["per_pair"], 0.38, color="white", edgecolor="k", lw=0.6, hatch="////", label="L0 − R0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"P{i + 1}" for i in xs])
    ax.set_xlabel("R0-reversed pair")
    ax.set_ylabel("$\\Delta$ tracking MAE / °C")
    ax.set_title("(a) Panel A")
    ax.set_ylim(-3.9, 1.7)          # top band holds the legend
    ax.legend(loc="upper right", frameon=False, fontsize=6, handlelength=0.8, handletextpad=0.3, borderaxespad=0.1,
              bbox_to_anchor=(0.0, 0.0, 1.0, 1.0))   # above P4-P6, whose bars are all below 0
    comma(ax, "y")

    ax = axes[1]
    for k, (a, b, mk, fc) in enumerate((("COISC", "R0", "o", "k"), ("L0", "R0", "s", "white"))):
        s = seed_level_panel_b(d, wp, a, b).sort_values()
        point = res["panel_B_unaffected"][f"{a} - {b}"]["point"]
        assert abs(s.mean() - point) < 1e-12, (a, b, s.mean(), point)
        ax.plot(np.arange(len(s)), s.values, ls="none", marker=mk, ms=3, mfc=fc, mec="k", mew=0.6, label=f"{a} − {b}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(res["ni_margin_C"], color="k", ls="--", lw=0.8)
    ax.text(0.5, res["ni_margin_C"] - 0.004, "NI margin", va="top", fontsize=6.5)
    ax.set_xlabel("Seed (sorted by difference)")
    ax.set_ylabel("Seed-level $\\Delta$ tracking MAE / °C")
    ax.set_title("(b) Panel B")
    ax.set_xticks([0, 10, 20, 30, 39])
    ax.set_xticklabels(["1", "11", "21", "31", "40"])
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    comma(ax, "y")

    ax = axes[2]
    m = res["panel_B_descriptive_means"]
    names = ["R0", "L0", "COISC"]
    tv = [m[v]["total_variation_u"] for v in names]
    sat = [m[v]["saturation_fraction"] for v in names]
    xs = np.arange(3)
    ax.bar(xs, tv, 0.55, color=["0.3", "white", "k"], edgecolor="k", lw=0.6, hatch=[None, "////", None])
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 560)          # bars use the lower half, the saturation line the upper half
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("Total variation of $u_h$ / %")
    ax.set_title("(c) Command activity")
    ax2 = ax.twinx()
    ax2.plot(xs, sat, color="k", marker="D", ms=3.5, mfc="white", lw=0.8)
    ax2.set_ylim(-0.75, 0.85)
    ax2.set_yticks([0.4, 0.6, 0.8])
    ax2.set_ylabel("Saturation fraction")
    comma(ax2, "y")
    for i, v in enumerate(sat):
        ax2.text(i, v + 0.04, f"{v:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    ax = axes[3]
    r = pd.read_csv(FD / "derived_reversal_seed.csv")   # derive_reversal.py (post-hoc, descriptive)
    rng = np.random.default_rng(0)
    for i, v in enumerate(names):
        rv = r[r["variant"] == v]
        ax.bar(i - 0.18, rv["reversal_plant"].mean(), 0.34, color="white", edgecolor="k", lw=0.6)
        ax.bar(i + 0.18, rv["reversal_internal"].mean(), 0.34, color="0.75", edgecolor="k", lw=0.6)
        ax.plot(i - 0.18 + rng.uniform(-0.1, 0.1, len(rv)), rv["reversal_plant"], ls="none", marker="o", ms=1.6, mfc="k", mec="none")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.36)          # dots stay below 0,25; legend uses the top band
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_ylabel("Reversal fraction")
    ax.set_title("(d) Reversed updates")
    ax.legend(handles=[matplotlib.patches.Patch(facecolor="white", edgecolor="k", lw=0.6, label="plant $T_p$"),
                       matplotlib.patches.Patch(facecolor="0.75", edgecolor="k", lw=0.6, label=r"internal $\hat T_p$")],
              loc="upper center", ncol=1, frameon=False, fontsize=6.5, handlelength=1.0, borderaxespad=0.1)
    comma(ax, "y")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()

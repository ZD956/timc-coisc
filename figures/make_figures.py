"""Redraw Figures 1-7 of the manuscript from the released summary results in ../results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
D23 = RES / "holdout23"
CV = RES / "control"
LOCK = REPO / "protocols/CONTROL_REACHABILITY_STRATA_LOCK.csv"
PA = RES / "posthoc_ablation"
FD = RES / "figures_data"

SEEDS = [20260922, 20260923, 20260924]
SEED_LABEL = {s: f"Seed {i + 1}" for i, s in enumerate(SEEDS)}
SEED_MARK = {20260922: "o", 20260923: "s", 20260924: "^"}
C_R0, C_CO = "#4d4d4d", "#c0392b"
VAR_COLOR = {"L0": "#7f7f7f", "PRE_B": "#1f77b4", "SIGN_A": "#ff7f0e", "SIGN_B": "#9467bd", "COISC": "#c0392b"}
VAR_LABEL = {"L0": "L0", "PRE_B": "Pre-only", "SIGN_A": r"Sign-only ($\lambda$)",
             "SIGN_B": "Sign-only (own weight)", "COISC": "COISC"}
T = np.arange(10, 121, 10)
TAU_P, G = 39.1, 10.0
W2 = 7.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


def csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig = plt.figure(figsize=(W2, 2.9))
    axa = fig.add_axes([0.0, 0.0, 0.34, 1.0])
    axb = fig.add_axes([0.37, 0.0, 0.63, 1.0])
    for ax in (axa, axb):
        ax.set_axis_off()

    axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.set_aspect("equal")
    axa.add_patch(Rectangle((1, 1.2), 8, 7.6, fill=False, lw=1.4, ec="k"))
    axa.text(5, 9.55, "Thermal enclosure (schematic)", ha="center", fontsize=7.5)
    for i, x in enumerate(np.linspace(2.0, 8.0, 6)):
        hl = i == 1
        axa.add_patch(Rectangle((x - 0.42, 1.55), 0.84, 0.45, fc="#c0392b" if hl else "#d9d9d9",
                                ec="k", lw=0.5))
    axa.text(2.0 + 1.2, 2.25, "$u_h$", ha="center", va="bottom", fontsize=8, color="#c0392b")
    axa.text(5, 0.75, "six heaters; $u_h$ actuated, others held", ha="center", fontsize=6.5)
    axa.add_patch(Circle((5.0, 5.4), 0.38, fc="white", ec="#c0392b", lw=1.3, ls="--"))
    axa.text(5.0, 5.4, "$T_p$", ha="center", va="center", fontsize=7.5)
    axa.text(5.0, 4.65, "internal, unmeasured\nin deployment", ha="center", va="top", fontsize=6)
    per = [(1, 6.9), (1, 4.2), (3.4, 8.8), (6.6, 8.8), (9, 6.9), (9, 4.2)]
    for k, (x, y) in enumerate(per):
        axa.add_patch(Circle((x, y), 0.3, fc="#1f77b4", ec="k", lw=0.5))
        dx = -0.55 if x <= 1 else (0.55 if x >= 9 else 0)
        dy = -0.6 if y >= 8.8 else 0
        axa.text(x + dx, y + dy, f"$T_{k + 1}$", ha="center", va="center", fontsize=7)
    axa.text(5, 6.9, "peripheral points $T_1,\\dots,T_6$\n(measured; positions illustrative)",
             ha="center", fontsize=6, color="#1f77b4")
    axa.text(0.2, 9.7, "(a)", fontsize=9, fontweight="bold")

    axb.set_xlim(0, 10); axb.set_ylim(0, 10)
    axb.text(0.0, 9.7, "(b)", fontsize=9, fontweight="bold")
    stages = [
        ("Development", "91 fit / 18 selection\nsources",
         "four gates on 3/3 seeds\n(window-pooled)", "passed; accuracy\npenalty under\nequal-source weights"),
        ("Frozen holdout", "23 independent sources,\nhashed before inference",
         "NI margin 0.005 °C;\nPLE, WSR ≤ 0.9 × R0;\nPSM floor", "sensitivity met 3/3;\nNI not established 0/3"),
        ("Model-in-the-loop\nMPC", "24 learned plants\n× 12 scenarios",
         "upper 95 % CI of Δ < 0\non each seed", "supported 1/3;\ncriterion not met"),
        ("Post-hoc checks", "development, holdout\nand MIL",
         "rule hashed in each\nprotocol", "attribution and\nrobustness; never\nrevises a verdict"),
    ]
    xs = [0.1, 2.6, 5.1, 7.6]
    wbox = 2.25
    rows = [("", 7.6, 1.5), ("Population", 5.55, 1.45), ("Decision rule", 3.5, 1.45), ("Outcome", 1.45, 1.45)]
    boxes = []
    for (title, pop, rule, outc), x in zip(stages, xs):
        fc = "#f7e1de" if title.startswith("Post") is False else "#eeeeee"
        axb.add_patch(FancyBboxPatch((x, 7.6), wbox, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                     fc=fc, ec="k", lw=0.7))
        axb.text(x + wbox / 2, 8.35, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
        for txt, (_, y, h) in zip((pop, rule, outc), rows[1:]):
            axb.add_patch(FancyBboxPatch((x, y), wbox, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                         fc="white", ec="#888888", lw=0.5))
            boxes.append((axb.add_patch(FancyBboxPatch((x, y), wbox, h,
                                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                                        fc="none", ec="none")),
                          axb.text(x + wbox / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3)))
    for x in xs[:-1]:
        axb.annotate("", xy=(x + wbox + 0.24, 8.35), xytext=(x + wbox + 0.02, 8.35),
                     arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k"))
    for name, y, h in rows[1:]:
        axb.text(-0.05, y + h / 2, name, ha="right", va="center", fontsize=6.3, rotation=90)
    axb.text(5.0, 0.6, "chronological order; no later stage revised an earlier verdict",
             ha="center", fontsize=6.5, style="italic")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for box, txt in boxes:
        bb, tb = box.get_window_extent(rend), txt.get_window_extent(rend)
        pad = 2.0
        assert (tb.x0 >= bb.x0 + pad and tb.x1 <= bb.x1 - pad and tb.y0 >= bb.y0 + pad
                and tb.y1 <= bb.y1 - pad), f"text overflows box: {txt.get_text()!r}"
    save(fig, "fig1")


# ---------------------------------------------------------------- Figure 2
def fig2():
    # equal-source-weighted quantiles over the 23 holdout sources, precomputed from window-level files
    qd = csv(FD / "fig2_holdout_Tp_sensitivity_quantiles.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    check = []
    for ax, seed in zip(axes, SEEDS):
        ax.axvspan(0, TAU_P - G, color="#fde0dd", lw=0)
        ax.axvspan(TAU_P - G, TAU_P + G, color="#f0f0f0", lw=0)
        ax.axvspan(TAU_P + G, 125, color="#e5f5e0", lw=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(TAU_P, color="k", ls=":", lw=0.8)
        for var, col in (("R0", C_R0), ("COISC", C_CO)):
            g = qd[(qd.seed == seed) & (qd.variant == var)].sort_values("t_s")
            assert g.t_s.tolist() == T.tolist()
            q = g[["q25_S_Tp", "q50_S_Tp", "q75_S_Tp"]].to_numpy()
            ax.fill_between(T, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0)
            ax.plot(T, q[:, 1], color=col, marker="o", ms=2.5, label=var)
            frac = g.frac_negative_S_Tp_20s.iloc[0]
            check.append((seed, var, round(frac, 3), round(q[1, 1], 4)))
        ax.set_title(SEED_LABEL[seed])
        ax.set_xlim(5, 125)
        ax.set_xticks([10, 30, 50, 70, 90, 110])
        ax.set_xlabel("Lead time $t$ (s)")
    ymin, ymax = axes[0].get_ylim()
    for ax in axes:
        for t0, n in ((10, 2), (20, 2), (30, 1)):
            ax.plot([t0], [ymax * 0.92], marker="v", color="#2c7fb8", ms=4 + 1.5 * n, ls="none")
        ax.text(TAU_P + 1.5, ymin + 0.06 * (ymax - ymin), "prior\n39.1 s", fontsize=6)
        ax.text(14, ymin + 0.06 * (ymax - ymin), "pre-lag", fontsize=6, color="#99000d")
        ax.text(80, ymin + 0.06 * (ymax - ymin), "post-lag", fontsize=6, color="#006d2c")
    axes[0].set_ylabel("$S_{T_p}(t)$ (°C per full-scale fraction)")
    axes[0].plot([], [], marker="v", color="#2c7fb8", ls="none", label="measured onset")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), frameon=False, fontsize=6.3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2")
    print("  check frac(S<-1e-4) and median at 20 s:", check)


# ---------------------------------------------------------------- Figure 4 (holdout)
def fig3():
    b = csv(D23 / "delivered23_cluster_bootstrap.csv")
    b = b[(b.cohort == "all23") & (b.output == "Tp")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1.6]})
    x = np.arange(3)
    for i, seed in enumerate(SEEDS):
        row = b[(b.seed == seed) & (b.endpoint == "MAE") & (b.horizon_s == 120)].iloc[0]
        lo, hi = json.loads(row.absolute_ci)
        a1.errorbar(i, row.absolute_change, yerr=[[row.absolute_change - lo], [hi - row.absolute_change]],
                    fmt=SEED_MARK[seed], color="k", ms=4, capsize=3, lw=0.9)
    a1.axhline(0, color="k", lw=0.5)
    a1.axhline(0.005, color="#c0392b", ls="--", lw=0.8)
    a1.text(1.5, 0.0065, "NI margin\n0.005 °C", color="#c0392b", fontsize=6, ha="center", va="bottom")
    a1.set_xticks(x, [SEED_LABEL[s] for s in SEEDS])
    a1.set_xlim(-0.5, 2.5)
    a1.set_ylabel(r"$\Delta$MAE of $T_p$ at 120 s (°C)")
    a1.set_title("(a) Prediction: COISC − R0, 95 % CI")

    metrics = ["PLE", "WSR", "PSM"]
    for j, m in enumerate(metrics):
        for i, seed in enumerate(SEEDS):
            row = b[(b.seed == seed) & (b.endpoint == m) & (b.gate_contrast.isna())].iloc[0]
            ratio = row.COISC / row.R0
            lo, hi = [1 + v / 100 for v in json.loads(row.relative_ci)]
            xp = j + (i - 1) * 0.22
            a2.errorbar(xp, ratio, yerr=[[ratio - lo], [hi - ratio]], fmt=SEED_MARK[seed], color=C_CO,
                        ms=4, capsize=2.5, lw=0.9, label=SEED_LABEL[seed] if j == 0 else None)
    a2.axhline(1, color="k", lw=0.5)
    a2.plot([-0.4, 1.4], [0.9, 0.9], color="#2c7fb8", ls="--", lw=0.8)
    a2.text(1.42, 0.9, "0.9", color="#2c7fb8", fontsize=6.5, va="center")
    a2.plot([1.6, 2.4], [0.2, 0.2], color="#2c7fb8", ls="--", lw=0.8)
    a2.text(2.42, 0.2, "0.2", color="#2c7fb8", fontsize=6.5, va="center")
    a2.set_xticks(range(3), [f"{m} of $T_p$" for m in metrics])
    a2.set_ylim(0, 1.45)
    a2.set_ylabel("COISC / R0")
    a2.set_title("(b) Sensitivity diagnostics under the prior, 95 % CI")
    a2.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4")


# ---------------------------------------------------------------- Figure 3 (ablation)
def fig4():
    dev = csv(PA / "ablation_summary.csv")
    dev["variant"] = dev.variant.replace({"COISC_REPRO": "COISC"})
    t6 = csv(PA / "ablation_sensitivity_all_outputs.csv")
    t6 = t6[t6.output == "T6"].copy()
    t6["variant"] = t6.variant.replace({"COISC_REPRO": "COISC"})
    hold = csv(PA / "holdout_ablation_summary.csv")
    rows = []
    for seed in SEEDS:
        d = dev[dev.seed == seed].set_index("variant")
        yy = t6[t6.seed == seed].set_index("variant")
        h = hold[hold.seed == seed].set_index("variant")
        for v in VAR_COLOR:
            rows.append(dict(split="Development", seed=seed, variant=v,
                             dmae=d.loc[v, "Tp_MAE_120s"] - d.loc["R0", "Tp_MAE_120s"],
                             ple=d.loc[v, "Tp_PLE"] / d.loc["R0", "Tp_PLE"],
                             wsr=d.loc[v, "Tp_WSR"] / d.loc["R0", "Tp_WSR"],
                             t6=yy.loc[v, "PLE"] / yy.loc["R0", "PLE"]))
            rows.append(dict(split="Holdout", seed=seed, variant=v, dmae=h.loc[v, "dTp_MAE_120_vs_R0"],
                             ple=h.loc[v, "ratio_PLE_PRIOR"], wsr=h.loc[v, "ratio_WSR_PRIOR"],
                             t6=h.loc[v, "ratio_T6_PLE_PRIOR"]))
    df = pd.DataFrame(rows)
    cols = [("ple", "PLE ratio of $T_p$"), ("wsr", "WSR ratio of $T_p$"), ("t6", "PLE ratio of $T_6$")]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.2), sharex="row")
    for r_i, split in enumerate(("Development", "Holdout")):
        sub = df[df.split == split]
        for c_i, (key, lab) in enumerate(cols):
            ax = axes[r_i, c_i]
            ax.axhline(1, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            if key != "t6":
                ax.axhline(0.9, color="#2c7fb8", ls="--", lw=0.7)
            for _, p in sub.iterrows():
                ax.plot(p.dmae, p[key], marker=SEED_MARK[p.seed], color=VAR_COLOR[p.variant], ms=4.5,
                        ls="none", mec="k", mew=0.3)
            ax.set_yscale("log")
            ax.set_title(f"{split}: {lab}")
            if r_i == 1:
                ax.set_xlabel(r"$\Delta$MAE of $T_p$ at 120 s vs R0 (°C)")
            if c_i == 0:
                ax.set_ylabel("Variant / R0 (log scale)")
    for ax in axes.ravel():
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[:, 0]:
        ax.set_yticks([0.5, 0.7, 1.0])
    for ax in axes[:, 1]:
        ax.set_yticks([0.01, 0.1, 1.0])
    for ax in axes[:, 2]:
        ax.set_yticks([0.3, 1, 3, 10])
    handles = [plt.Line2D([], [], marker="o", color=c, ls="none", mec="k", mew=0.3, label=VAR_LABEL[v])
               for v, c in VAR_COLOR.items()]
    handles += [plt.Line2D([], [], marker=m, color="w", mec="k", ls="none", label=SEED_LABEL[s])
                for s, m in SEED_MARK.items()]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8)
    save(fig, "fig3")
    print(df.round(3).to_string())


# ---------------------------------------------------------------- Figure 5
def fig5():
    lag = csv(PA / "lag_scenarios_holdout23.csv")
    lag = lag[lag.split == "holdout23"]
    panels = {
        "Tp": ("$T_p$", "Tp", [("PRIOR", "prior\n(39.1 s)"), ("T30", "τ = 30 s"),
                               ("T20_measured_median", "τ = 20 s\n(median)"),
                               ("T10_measured_min", "τ = 10 s\n(min)"), ("ALL_STEPS", "lag-free")]),
        "T6": ("$T_6$", "T6", [("PRIOR", "prior\n(180 s)"), ("T60_measured_median", "τ = 60 s\n(median)"),
                                ("ALL_STEPS", "lag-free")]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(W2, 3.9), gridspec_kw={"hspace": 0.75})
    for r_i, (sym, out, scen) in enumerate(panels.values()):
        for c_i, m in enumerate(("PLE", "WSR", "PSM")):
            ax = axes[r_i, c_i]
            ax.axhline(1, color="k", lw=0.5)
            if m != "PSM":
                ax.axhline(0.9, color="#2c7fb8", ls="--", lw=0.7)
            for x, (code, _lab) in enumerate(scen):
                for i, seed in enumerate(SEEDS):
                    row = lag[(lag.seed == seed) & (lag.output == out) & (lag.scenario == code)]
                    val = row[f"ratio_{m}"].iloc[0] if len(row) else np.nan
                    if np.isfinite(val):
                        ax.plot(x + (i - 1) * 0.18, val, marker=SEED_MARK[seed], color=C_CO, ms=4,
                                ls="none", mec="k", mew=0.3)
                    elif i == 1:
                        ax.text(x, 0.5, "n/a", ha="center", va="center", fontsize=6, color="#777777",
                                transform=ax.get_xaxis_transform())
            ax.set_xticks(range(len(scen)), [lab for _, lab in scen], fontsize=6)
            ax.set_xlim(-0.5, len(scen) - 0.5)
            ax.set_ylim(0, 1.4)
            ax.set_title(f"{m} of {sym}: COISC / R0")
    handles = [plt.Line2D([], [], marker=mk, color=C_CO, mec="k", mew=0.3, ls="none", label=SEED_LABEL[s])
               for s, mk in SEED_MARK.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.14, wspace=0.25)
    save(fig, "fig5")


# ---------------------------------------------------------------- Figure 6
STRATA = [("BRACKETED_AT_600S", "bracketed"), ("BOTH_ABOVE_37_AT_600S", "both above"),
          ("BOTH_BELOW_37_AT_600S", "both below")]


def fig6():
    fam = csv(CV / "CONTROL_FAMILY_RESULTS.csv")
    st = csv(CV / "CONTROL_REACHABILITY_STRATA_RESULTS.csv")
    st = st[st.panel == "primary"]
    fams = ["F1", "F2", "F3", "F4"]
    M = np.array([[fam[(fam.seed == s) & (fam.family == f)].Delta_COISC_minus_R0.iloc[0] for f in fams]
                  for s in SEEDS])
    M = np.column_stack([M, M.mean(1)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    lim = np.abs(M).max()
    im = a1.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(3):
        for j in range(5):
            a1.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.55 * lim else "k")
    a1.axvline(3.5, color="k", lw=1.0)
    a1.set_xticks(range(5), [f"{f}\n(n={n})" for f, n in zip(fams, (17, 3, 3, 1))] + ["family-\nequal"])
    a1.set_yticks(range(3), [SEED_LABEL[s] for s in SEEDS])
    a1.set_title(r"(a) $\Delta$ tracking MAE of $T_p$ (°C), COISC − R0")
    cb = fig.colorbar(im, ax=a1, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    rng = np.random.default_rng(0)
    a2.axhline(0, color="k", lw=0.5)
    means = {}
    for j, (code, lab) in enumerate(STRATA):
        for i, seed in enumerate(SEEDS):
            v = st[(st.seed == seed) & (st.reachability_stratum == code)].Delta_COISC_minus_R0.to_numpy()
            xp = j + (i - 1) * 0.26
            a2.scatter(xp + rng.uniform(-0.07, 0.07, len(v)), v, s=3, color="#999999", lw=0, alpha=0.6)
            a2.plot(xp, v.mean(), marker=SEED_MARK[seed], color=C_CO, ms=5, mec="k", mew=0.4, ls="none",
                    label=SEED_LABEL[seed] if j == 0 else None)
            means[(seed, lab)] = (len(v), round(v.mean(), 4))
    n = {lab: means[(SEEDS[0], lab)][0] for _, lab in STRATA}
    a2.set_xticks(range(3), [f"{lab}\n(n = {n[lab]})" for _, lab in STRATA])
    a2.set_ylabel(r"$\Delta$ per plant–scenario pair (°C)")
    a2.set_title("(b) By reachability stratum (pairs and unweighted mean)")
    a2.legend(frameon=False, loc="lower left", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig6")
    print("  family-equal:", M[:, 4].round(4), " strata means:", means)


# ---------------------------------------------------------------- Figure 7
def fig7():
    tr = csv(FD / "fig7_control_traces.csv")
    lock = csv(LOCK)
    scen_label = {sid: f"S{int(sid[1:])}" for sid in lock.scenario_id.unique()}
    plant = "P_seed20260922"
    pl = lock[lock.plant_name == plant].set_index("scenario_id")
    cases = [("bracketed", "S05"), ("bracketed", "S07"), ("both below (unreachable)", "S04")]

    def trace(sid, seed, var):
        g = tr[(tr.plant == plant) & (tr.scenario_id == sid) & (tr.controller_seed == seed)
               & (tr.controller_variant == var) & (tr.panel == "primary")].sort_values("step")
        assert g.step.tolist() == list(range(61)), (sid, seed, var)
        return g.Tp_C.iloc[0], g.Tp_C.to_numpy()[1:], g.applied_UH_percent.to_numpy()[1:]

    tt = np.arange(0, 601, 10)
    fig, axes = plt.subplots(4, 3, figsize=(W2, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 0.9, 1.5, 0.9]})
    for c_i, (lab, sid) in enumerate(cases):
        row = pl.loc[sid]
        for s_i, seed in enumerate(SEEDS[:2]):
            ay, au = axes[2 * s_i, c_i], axes[2 * s_i + 1, c_i]
            ay.axhline(37.0, color="k", ls=":", lw=0.8)
            for var, col in (("R0", C_R0), ("COISC", C_CO)):
                c0, y, u = trace(sid, seed, var)
                mae = np.mean(np.abs(y - 37.0))
                ay.plot(tt, np.r_[c0, y], color=col, label=f"{var} (MAE {mae:.2f} °C)")
                au.step(tt[1:], u, where="pre", color=col)
            ay.set_title(f"{scen_label[sid]}, {lab}\n{SEED_LABEL[seed]} internal models", fontsize=7.5)
            ay.legend(frameon=False, loc={0: "best", 1: "lower right", 2: "center left"}[c_i], fontsize=6)
            if c_i == 0:
                ay.set_ylabel("$T_p$ (°C)")
                au.set_ylabel("$u_h$ (%)")
            au.set_ylim(-5, 105)
        print(f"  {lab}: {scen_label[sid]}  T_p(600 s) at u_h 0/100 % = "
              f"{row.Tp_UH_0_600s:.2f} / {row.Tp_UH_100_600s:.2f} °C")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, 600)
    fig.tight_layout(h_pad=0.5, w_pad=1.0)
    save(fig, "fig7")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()

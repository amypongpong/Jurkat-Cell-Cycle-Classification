"""Draw paper Figures 2, 3 and 4 from the saved results in results/.

Usage:  python make_paper_figures.py
Writes PNG files to paper_figures/. Figure 1 (degradation steps) needs the BBBC048
images and is produced by the notebook (Section 15).

Figure 4 uses results/confusion_matrices_figure4.csv, the confusion matrices of the
three runs the notebook selects in Section 15 (Model A seed 42; Model C seeds 25 and 2).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "DejaVu Sans"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "paper_figures")
os.makedirs(OUT, exist_ok=True)

NAMES = ["Clean images, correct labels (baseline)", "Degraded images, correct labels",
         "Clean images, noisy labels", "Degraded images, noisy labels (combined)"]
STAGES = ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"]
ABBREV = ["G1", "S", "G2", "Pro", "Meta", "Ana", "Telo"]

runs = pd.read_csv(os.path.join(RES, "all_runs.csv"))
summary = pd.read_csv(os.path.join(RES, "summary_by_condition.csv"), header=[0, 1], index_col=0)
MAJORITY = 1433 / 3233 * 100  # share of G1 in the shared test set (Table 1)

# Figure 2: mean +/- 1 SD of each metric across the 30 seeds
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
colors = ["#194d68", "#537996", "#c88435", "#a63337"]
xt = ["Model A\nclean images,\ncorrect labels", "Model B\ndegraded images,\ncorrect labels",
      "Model C\nclean images,\nnoisy labels", "Model D\ndegraded images,\nnoisy labels"]
x = np.arange(4); width = 0.20
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
for i, (m, lab) in enumerate(zip(metrics, labels)):
    means = [summary.loc[n, (m, "mean")] * 100 for n in NAMES]
    stds = [summary.loc[n, (m, "std")] * 100 for n in NAMES]
    ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=lab, color=colors[i], capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": "#4a4a4a"})
ax.axhline(y=MAJORITY, color="gray", linestyle="--", linewidth=1.1)
ax.text(3.35, MAJORITY + 0.7, f"majority-class rate ({MAJORITY:.1f}%)", color="#5a5a5a", fontsize=8.5)
ax.axhline(y=100 / 7, color="gray", linestyle=":", linewidth=1.1)
ax.text(3.35, 100 / 7 + 0.9, "uniform-random rate (14.3%)", color="#5a5a5a", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=9.5)
ax.set_ylabel("Score (%)", fontsize=11); ax.set_ylim(0, 80); ax.set_xlim(-0.6, 3.8)
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#cccccc"); ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figure2_performance.png"), dpi=300); plt.close()

# Figure 3: accuracy of every run; same jitter as the notebook (Section 14)
short = ["A: clean images,\ncorrect labels", "B: degraded images,\ncorrect labels",
         "C: clean images,\nnoisy labels", "D: degraded images,\nnoisy labels"]
rng_jit = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(9, 5))
for i, n in enumerate(NAMES):
    sub = runs[runs["condition"] == n]
    xj = i + rng_jit.uniform(-0.15, 0.15, len(sub))
    ax.scatter(xj, sub["accuracy"] * 100, s=28, alpha=0.85,
               c=np.where(sub["collapsed"], "#a63337", "#194d68"))
    ax.hlines(sub["accuracy"].mean() * 100, i - 0.28, i + 0.28, color="black", linewidth=1.5)
ax.axhline(MAJORITY, color="gray", linestyle="--", linewidth=1)
ax.text(3.3, MAJORITY + 1, f"majority-class rate ({MAJORITY:.1f}%)", fontsize=8, color="#555555")
ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("Test accuracy (%)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figure3_every_run.png"), dpi=300); plt.close()

# Figure 4: confusion matrices, stages in cell-cycle order
cms = pd.read_csv(os.path.join(RES, "confusion_matrices_figure4.csv"))
panels = [("Clean images, correct labels (baseline)", 42, "Blues", "Model A"),
          ("Clean images, noisy labels", 25, "Reds", "Model C"),
          ("Clean images, noisy labels", 2, "Reds", "Model C")]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, (cond, seed, cmap, tag) in zip(axes, panels):
    sub = cms[(cms["condition"] == cond) & (cms["seed"] == seed)]
    M = sub.pivot(index="true_stage", columns="predicted_stage", values="count").loc[STAGES, STAGES].values
    run = runs[(runs["condition"] == cond) & (runs["seed"] == seed)].iloc[0]
    assert np.trace(M) / M.sum() == run["accuracy"] or abs(np.trace(M) / M.sum() - run["accuracy"]) < 1e-9
    flag = f" (collapsed to {run['dominant_class']})" if run["collapsed"] else ""
    sns.heatmap(M, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax, xticklabels=ABBREV, yticklabels=ABBREV,
                annot_kws={"fontsize": 10}, linewidths=0.4, linecolor="#e6e6e6")
    ax.set_title(f"{tag}, seed {seed}{flag}\naccuracy {run['accuracy']*100:.2f}%", fontsize=11.5)
    ax.set_xlabel("Predicted stage", fontsize=11); ax.set_ylabel("True stage", fontsize=11)
    ax.tick_params(labelsize=10.5); ax.tick_params(axis="y", rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figure4_confusion_matrices.png"), dpi=300); plt.close()
print("Figures written to", OUT)

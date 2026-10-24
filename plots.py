"""Figures for the report. All are saved as PNG in outputs/<domain>/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fuzzy

BG, FG, GRID = "#0c1116", "#dce6ec", "#243038"
CYAN, PINK, VIOLET, GREEN, AMBER, RED = "#35d0ff", "#ff5ec4", "#a78bfa", "#3ddc84", "#ffb020", "#ff3b3b"
LEVEL_COLORS = [GREEN, AMBER, RED]


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    ax.grid(color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, ax


def ga_convergence(ga_hist, pso_hist, path):
    fig, ax = _fig()
    ax.plot(range(1, len(ga_hist["best"]) + 1), ga_hist["best"], color=CYAN, lw=2, marker="o", ms=4, label="Genetic algorithm")
    if "mean" in ga_hist:
        ax.plot(range(1, len(ga_hist["mean"]) + 1), ga_hist["mean"], color=CYAN, lw=1, ls="--", alpha=0.5, label="GA population mean")
    ax.plot(range(1, len(pso_hist["best"]) + 1), pso_hist["best"], color=PINK, lw=2, marker="s", ms=4, label="Binary PSO")
    _style(ax, "Feature selection: fitness per generation (lower is better)", "Generation / iteration", "Validation RMSE + feature penalty")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def pred_vs_true(true, pred, path, title):
    fig, ax = _fig(5.5, 5)
    ax.scatter(true, pred, s=14, color=CYAN, alpha=0.8)
    m = max(np.max(true), np.max(pred))
    ax.plot([0, m], [0, m], color=FG, ls="--", lw=1)
    _style(ax, title, "True life left", "Predicted life left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def ablation_bars(names, means, stds, path):
    fig, ax = _fig(8, 4)
    colors = [VIOLET, VIOLET, PINK, CYAN, GREEN][: len(names)]
    ax.barh(range(len(names)), means, xerr=stds, color=colors, ecolor=FG, capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=FG, fontsize=9)
    ax.invert_yaxis()
    _style(ax, "Test RMSE by model (mean ± std over seeds, lower is better)", "RMSE", "")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def fuzzy_membership(path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG)
    x = np.linspace(0, 125, 400)
    for name, col in zip(("short", "medium", "long"), (RED, AMBER, GREEN)):
        a.plot(x, fuzzy.trap(x, *fuzzy.LIFE_SETS[name]), color=col, lw=2, label=name)
    _style(a, "Life left (rescaled 0-125)", "Predicted life left", "Membership")
    a.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    w = np.linspace(0, 1, 300)
    for name, col in zip(("slow", "fast"), (GREEN, RED)):
        b.plot(w, fuzzy.trap(w, *fuzzy.WEAR_SETS[name]), color=col, lw=2, label=name)
    _style(b, "Wear rate", "Wear rate", "Membership")
    b.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def fuzzy_surface(path):
    fig, ax = _fig(6, 4.5)
    life = np.linspace(0, 125, 120)
    wear = np.linspace(0, 1, 120)
    L, W = np.meshgrid(life, wear)
    S = fuzzy.risk_score(L, W)
    cs = ax.contourf(L, W, S, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r")
    cb = fig.colorbar(cs, ax=ax)
    cb.ax.tick_params(colors=FG)
    cb.set_label("Risk score", color=FG)
    _style(ax, "Fuzzy rule surface", "Predicted life left (0-125 scale)", "Wear rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def alert_timeline(label, true, pred_s, level, cap, path):
    fig, ax = _fig(9, 3.6)
    t = np.arange(1, len(true) + 1)
    for lv in (0, 1, 2):
        ax.fill_between(t, 0, cap, where=(level == lv), color=LEVEL_COLORS[lv], alpha=0.16, step="mid", linewidth=0)
    ax.plot(t, true, color=FG, ls="--", lw=1.4, label="True life left")
    ax.plot(t, pred_s, color=CYAN, lw=2, label="ANN prediction")
    _style(ax, f"{label}: alert level over the run (green normal, amber warning, red critical)", "Time step", "Life left")
    ax.set_ylim(0, cap)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)

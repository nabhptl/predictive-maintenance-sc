#!/usr/bin/env python
"""Train and evaluate the hybrid GA + ANN + fuzzy system for one machine type.

    python train.py --domain aircraft --data-dir data/CMAPSS
    python train.py --domain factory  --data-dir data/IMS
    python train.py --domain aircraft --data-dir data/synthetic/CMAPSS --quick     # smoke test

Steps
  1  load data and scale it
  2  baselines: Linear Regression, Random Forest, ANN on all sensors
  3  GA (and binary PSO for comparison) choose the useful sensors
  4  GA tunes the ANN (layers, regularisation, learning rate)
  5  final ANN trained for several random seeds -> test RMSE mean +- std
  6  fuzzy alert engine evaluated on held-out run-to-failure units (lead time, false alarms)
  7  save models, results.json, replay.json for the dashboard, and PNG figures
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import config
import datasets
import fuzzy
import plots
from optimizers import GeneticAlgorithm, binary_pso

warnings.filterwarnings("ignore")

DEFAULT_HP = {"hidden": (64, 32), "alpha": 1e-4, "lr": 1e-3}
HIDDEN = [(32,), (64,), (64, 32), (128, 64), (64, 64, 32)]
ALPHA = [1e-5, 1e-4, 1e-3, 1e-2]
LR = [1e-3, 3e-3, 1e-2]


# ----------------------------------------------------------------------------- metrics
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def nasa_score(true, pred):
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


# ----------------------------------------------------------------------------- models
def fit_mlp(X, y, hp, cap, seed, max_iter=200):
    m = MLPRegressor(hidden_layer_sizes=tuple(hp["hidden"]), alpha=hp["alpha"], learning_rate_init=hp["lr"],
                     max_iter=max_iter, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                     random_state=seed)
    m.fit(X, y / cap)
    return lambda Z: np.clip(m.predict(Z) * cap, 0, cap), m


def fit_rf(X, y, cap, seed, trees=100):
    m = RandomForestRegressor(n_estimators=trees, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    m.fit(X, y)
    return lambda Z: np.clip(m.predict(Z), 0, cap)


def fit_lr(X, y, cap):
    m = LinearRegression().fit(X, y)
    return lambda Z: np.clip(m.predict(Z), 0, cap)


# ----------------------------------------------------------------------------- alerts
def first_index(level, k):
    hit = np.flatnonzero(level >= k)
    return int(hit[0]) if len(hit) else None


def alert_summary(units):
    """units: list of dicts with 'level', 'true' (capped life left), 'cap'."""
    lead_w, lead_c, fa_w, fa_c, n_h = [], [], 0, 0, 0
    for u in units:
        L = len(u["level"])
        iw, ic = first_index(u["level"], 1), first_index(u["level"], 2)
        lead_w.append(L - (iw + 1) if iw is not None else 0)
        lead_c.append(L - (ic + 1) if ic is not None else 0)
        healthy = u["true"] >= u["cap"]
        n_h += int(healthy.sum())
        fa_w += int((u["level"][healthy] >= 1).sum())
        fa_c += int((u["level"][healthy] >= 2).sum())
    lead_c = np.array(lead_c)
    return {
        "units": len(units),
        "mean_warning_lead": float(np.mean(lead_w)),
        "mean_critical_lead": float(lead_c.mean()),
        "min_critical_lead": int(lead_c.min()),
        "critical_missed": int((lead_c == 0).sum()),
        "critical_in_time_pct": float(100 * np.mean(lead_c >= 10)),
        "false_warning_rate_pct": float(100 * fa_w / max(1, n_h)),
        "false_critical_rate_pct": float(100 * fa_c / max(1, n_h)),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(config.DOMAINS), required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seeds", type=int, default=5, help="random seeds for the final ANN comparison")
    ap.add_argument("--quick", action="store_true", help="small search budgets, for a fast test run")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--test-run", default="set2_bearing1")
    ap.add_argument("--val-run", default="set1_bearing4")
    args = ap.parse_args()

    t_start = time.time()
    cfg = config.DOMAINS[args.domain]
    cap = cfg["cap"]
    data_dir = args.data_dir or str(config.DATA_DIR / ("CMAPSS" if args.domain == "aircraft" else "IMS"))
    pop = args.pop or (8 if args.quick else 20)
    gens = args.gens or (5 if args.quick else 15)
    hp_pop, hp_gens, hp_iter = (4, 3, 40) if args.quick else (8, 5, 80)
    seeds = list(range(args.seeds if not args.quick else min(args.seeds, 2)))
    model_dir = config.MODELS_DIR / args.domain
    out_dir = config.OUTPUT_DIR / args.domain
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. data
    print(f"[1/7] Loading {args.domain} data from {data_dir}")
    b = datasets.load_domain(args.domain, data_dir, subset=args.subset, seed=args.seed,
                             test_run=args.test_run, val_run=args.val_run)
    print("      " + b.note)
    feats = b.features
    scaler = MinMaxScaler().fit(b.train[feats].values)
    Xtr, ytr = scaler.transform(b.train[feats].values), b.train["rul"].to_numpy()
    Xva, yva = scaler.transform(b.val[feats].values), b.val["rul"].to_numpy()

    if b.domain == "aircraft":
        last = b.test.sort_values(["unit", "t"]).groupby("unit").tail(1)
        Xte, truth_raw = scaler.transform(last[feats].values), b.test_truth
    else:
        Xte, truth_raw = scaler.transform(b.test[feats].values), b.test["rul"].to_numpy()
    truth_cap = np.minimum(truth_raw, cap)

    def test_metrics(predict, cols=None):
        p = predict(Xte if cols is None else Xte[:, cols])
        return {"rmse": rmse(truth_raw, p), "rmse_capped": rmse(truth_cap, p), "nasa": nasa_score(truth_raw, p)}, p

    all_cols = np.arange(len(feats))

    # ---- 3. feature selection (done before the multi-seed comparison, needs only train + val)
    print(f"[3/7] Feature selection: GA (pop {pop}, gens {gens}) and binary PSO")
    stride = max(1, len(Xtr) // 8000)
    Xs, ys = Xtr[::stride], ytr[::stride]

    def feature_fitness(mask):
        cols = np.flatnonzero(mask)
        if len(cols) == 0:
            return 1e6
        m = RandomForestRegressor(n_estimators=30, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=0)
        m.fit(Xs[:, cols], ys)
        p = np.clip(m.predict(Xva[:, cols]), 0, cap)
        return rmse(yva, p) + cfg["feature_penalty"] * len(cols)

    ones = np.ones(len(feats), dtype=int)
    ga = GeneticAlgorithm([2] * len(feats), feature_fitness, pop_size=pop, generations=gens, seed=args.seed, seeded=[ones])
    ga_best, ga_fit, ga_hist = ga.run(verbose=True)
    pso_best, pso_fit, pso_hist = binary_pso(len(feats), feature_fitness, swarm=pop, iters=max(1, gens - 1),
                                             seed=args.seed, seeded=[ones])
    sel = np.flatnonzero(ga_best)
    sel_names = [feats[i] for i in sel]
    print(f"      GA selected {len(sel)}/{len(feats)}: {sel_names}")
    print(f"      PSO selected {int(pso_best.sum())}/{len(feats)} (fitness {pso_fit:.3f} vs GA {ga_fit:.3f})")

    # ---- 4. ANN hyper-parameters by GA
    print(f"[4/7] ANN tuning with GA (pop {hp_pop}, gens {hp_gens})")

    def hp_from(genes):
        return {"hidden": HIDDEN[genes[0]], "alpha": ALPHA[genes[1]], "lr": LR[genes[2]]}

    def hp_fitness(genes):
        pr, _ = fit_mlp(Xtr[:, sel], ytr, hp_from(genes), cap, seed=0, max_iter=hp_iter)
        return rmse(yva, pr(Xva[:, sel]))

    hp_ga = GeneticAlgorithm([len(HIDDEN), len(ALPHA), len(LR)], hp_fitness, pop_size=hp_pop, generations=hp_gens,
                             seed=args.seed, seeded=[np.array([2, 1, 0])])
    hp_best, hp_fit, hp_hist = hp_ga.run(verbose=True)
    best_hp = hp_from(hp_best)
    print(f"      best hyper-parameters: {best_hp}  (val RMSE {hp_fit:.3f})")

    # ---- 2/5. baselines and ablation over several seeds
    print(f"[2/7] and [5/7] Comparing models over {len(seeds)} seed(s)")
    variants = {
        "Linear Regression (all sensors)": lambda s: (fit_lr(Xtr, ytr, cap), all_cols),
        "Random Forest (all sensors)": lambda s: (fit_rf(Xtr, ytr, cap, s), all_cols),
        "ANN (all sensors)": lambda s: (fit_mlp(Xtr, ytr, DEFAULT_HP, cap, s)[0], all_cols),
        "ANN + GA sensors": lambda s: (fit_mlp(Xtr[:, sel], ytr, DEFAULT_HP, cap, s)[0], sel),
        "ANN + GA sensors + GA-tuned (proposed)": lambda s: (fit_mlp(Xtr[:, sel], ytr, best_hp, cap, s)[0], sel),
    }
    table, deployed = [], None
    for name, make in variants.items():
        runs = []
        for s in seeds:
            predict, cols = make(s)
            m, p = test_metrics(predict, cols)
            runs.append(m)
            if name.endswith("(proposed)") and s == seeds[0]:
                deployed = (predict, p)
        row = {"name": name, "seeds": len(seeds)}
        for k in ("rmse", "rmse_capped", "nasa"):
            vals = [r[k] for r in runs]
            row[k] = float(np.mean(vals))
            row[k + "_std"] = float(np.std(vals))
        table.append(row)
        print(f"      {name:42s} RMSE {row['rmse']:7.2f} +- {row['rmse_std']:.2f}   (capped {row['rmse_capped']:.2f})")

    # final deployed ANN: refit seed 0 with the proposed settings and keep the sklearn object
    predict_final, mlp = fit_mlp(Xtr[:, sel], ytr, best_hp, cap, seeds[0])
    pred_test = predict_final(Xte[:, sel])

    # ---- 6. fuzzy alerts on held-out units
    print("[6/7] Fuzzy alert engine on held-out run-to-failure units")
    wear = fuzzy.WearEstimator().fit(Xtr[:, sel], ytr, b.train["unit"].to_numpy())
    units_fuzzy, units_thr, replay_units = [], [], []
    for key, (label, df) in b.heldout.items():
        X = scaler.transform(df[feats].values)
        pred = predict_final(X[:, sel])
        w = wear.transform(X[:, sel])
        ps, score, level = fuzzy.alert_series(pred, w, cap)
        true = df["rul"].to_numpy()
        units_fuzzy.append({"level": level, "true": true, "cap": cap})
        units_thr.append({"level": fuzzy.threshold_levels(pred, cap), "true": true, "cap": cap})
        replay_units.append((key, label, df, X, ps, w, score, level))
    alerts = {"fuzzy": alert_summary(units_fuzzy), "threshold_baseline": alert_summary(units_thr)}
    for k, v in alerts.items():
        print(f"      {k:20s} critical lead {v['mean_critical_lead']:.1f} {cfg['unit']}, false warnings {v['false_warning_rate_pct']:.1f}%, "
              f"critical in time {v['critical_in_time_pct']:.0f}%")

    # ---- 7. save
    print("[7/7] Saving models, results and figures")
    joblib.dump({"scaler": scaler, "features": feats, "selected": [int(i) for i in sel], "model": mlp, "cap": cap,
                 "wear": wear, "hp": {**best_hp, "hidden": list(best_hp["hidden"])}}, model_dir / "bundle.joblib")

    results = {
        "domain": args.domain, "label": cfg["label"], "note": b.note, "unit": cfg["unit"], "cap": cap,
        "quick_run": bool(args.quick), "seeds": len(seeds), "train_minutes": round((time.time() - t_start) / 60, 1),
        "features_total": len(feats), "selected": sel_names, "selected_count": int(len(sel)),
        "ga": {"best_fitness": ga_fit, "history": ga_hist, "evals": ga_hist["evals"]},
        "pso": {"best_fitness": pso_fit, "history": pso_hist, "selected_count": int(pso_best.sum())},
        "hyperparameters": {**best_hp, "hidden": list(best_hp["hidden"]), "val_rmse": hp_fit, "history": hp_hist},
        "models": table, "alerts": alerts, "fuzzy_rules": [r for r, _ in fuzzy.RULES],
        "test_metric_note": ("RMSE against the official RUL_FD001 values (last cycle of each test engine)"
                             if b.domain == "aircraft" else "RMSE over every reading of the test run (capped life left)"),
    }
    (model_dir / "results.json").write_text(json.dumps(results, indent=2))

    # replay file for the dashboard
    corr = {}
    ytr_fit = ytr
    for i in sel:
        c = np.corrcoef(Xtr[:, i], ytr_fit)[0, 1] if Xtr[:, i].std() > 1e-9 else 0.0
        corr[i] = abs(np.nan_to_num(c))
    shown = sorted(sel, key=lambda i: -corr[i])[:3]
    while len(shown) < 3:
        shown.append(next(i for i in range(len(feats)) if i not in shown))
    maxlen = cfg["replay_max_len"]
    units_out = []
    for key, label, df, X, ps, w, score, level in replay_units:
        sl = slice(-maxlen, None) if maxlen else slice(None)
        units_out.append({
            "id": str(key), "label": label, "length": int(len(df[sl])),
            "pred": [round(float(v), 2) for v in ps[sl]],
            "wear": [round(float(v), 3) for v in w[sl]],
            "sensors": [[round(float(np.clip(v, 0, 1)), 3) for v in X[sl, i]] for i in shown],
        })
    replay = {
        "domain": args.domain, "label": cfg["label"], "unit": cfg["unit"], "unit_singular": cfg["unit_singular"], "cap": cap,
        "features": [b.feature_labels[f] for f in feats],
        "selected": [b.feature_labels[feats[i]] for i in sel],
        "shown": [b.feature_labels[feats[i]] for i in shown],
        "shown_names": [b.long_labels[feats[i]] for i in shown],
        "units": units_out,
        "metrics": {"rmse": table[-1]["rmse"], "rmse_std": table[-1]["rmse_std"],
                    "mean_critical_lead": alerts["fuzzy"]["mean_critical_lead"],
                    "quick_run": bool(args.quick), "seeds": len(seeds), "synthetic": "synthetic" in str(data_dir)},
        "note": b.note,
    }
    (model_dir / "replay.json").write_text(json.dumps(replay, separators=(",", ":")))

    plots.ga_convergence(ga_hist, pso_hist, out_dir / "ga_vs_pso.png")
    plots.pred_vs_true(truth_cap, pred_test, out_dir / "pred_vs_true_test.png", "Test set: predicted vs true life left")
    plots.ablation_bars([r["name"] for r in table], [r["rmse"] for r in table], [r["rmse_std"] for r in table], out_dir / "model_comparison.png")
    plots.fuzzy_membership(out_dir / "fuzzy_membership.png")
    plots.fuzzy_surface(out_dir / "fuzzy_surface.png")
    key, label, df, X, ps, w, score, level = replay_units[0]
    plots.alert_timeline(label, df["rul"].to_numpy(), ps, level, cap, out_dir / "alert_timeline_example.png")
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Results: {model_dir / 'results.json'}")
    if args.quick or "synthetic" in str(data_dir):
        print("NOTE: this was a quick or synthetic run. Do not report these numbers.")


if __name__ == "__main__":
    main()

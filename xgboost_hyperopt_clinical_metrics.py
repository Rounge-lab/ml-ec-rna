#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic XGBoost (Hyperopt-tuned) + clinical metrics pipeline for EC miRNA
VST-normalized data.

Nothing about a specific subset is hardcoded -- you pass the training CSV
and however many held-out test CSVs you have on the command line, and it
trains XGBoost with Hyperopt (TPE) tuned via 5-fold cross-validation 
(case-control pairs, the "set" column, kept within the
same fold), refits on the full training set, saves the model, 
plots the fold ROC curves, runs SHAP for feature
importance, evaluates on every test set you gave it, and writes one
clinical-metrics CSV covering all of them.

Usage:
    python xgboost_hyperopt_clinical_metrics.py \\
        --train ec_training.csv \\
        --test testing2=ec_testing2.csv \\
        --test testing1=ec_testing1.csv \\
        --tag first-quart

--test can be repeated any number of times (including zero), 
and --train can point at any subset's training CSV.
--tag controls the filenames for the saved model/plots/metrics; if you
don't pass one, it's derived from the --train filename.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import xgboost as xgb
from hyperopt import hp, fmin, tpe, pyll, STATUS_OK, Trials
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, balanced_accuracy_score, precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 912437
N_SPLITS = 5
N_JOBS = 100

# Dropped from any file that has them; harmless if a file lacks one.
COLUMNS_TO_DROP = ["sampleid"]

SPACE4XGB = {
    "n_estimators": pyll.scope.int(hp.quniform("n_estimators", 1, 1000, 1)),
    "eta": hp.loguniform("eta", np.log(10 ** -4), np.log(10 ** 2)),
    "max_depth": pyll.scope.int(hp.quniform("max_depth", 1, 16, 1)),
    "subsample": hp.uniform("subsample", 0.3, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 20, 1),
    "gamma": hp.quniform("gamma", 0, 20, 0.1),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-5), np.log(20)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-5), np.log(20)),
}


# ---------------------------------------------------------------------------
# Clinical metrics
# ---------------------------------------------------------------------------

def sensitivity_at_specificity(y_true, y_probs, target_specificity):
    """Highest sensitivity achievable at >= target_specificity, and the threshold that gives it."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificity = 1 - fpr
    valid = specificity >= target_specificity
    if not valid.any():
        return np.nan, np.nan
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(tpr[valid_idx])]
    return tpr[best], thresholds[best]


def bootstrap_auc_ci(y_true, y_probs, n_boot=2000, ci=0.95, seed=RANDOM_STATE):
    """Percentile bootstrap CI for AUC (plain row-level resampling)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    boot_aucs = []

    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, yp))

    lower = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return float(np.mean(boot_aucs)), float(lower), float(upper)


def ppv_npv_at_prevalence(sensitivity, specificity, prevalence):
    """PPV/NPV adjusted to an assumed population prevalence, via Bayes' theorem."""
    ppv = (sensitivity * prevalence) / (
        sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    )
    npv = (specificity * (1 - prevalence)) / (
        specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
    )
    return ppv, npv


def full_metrics_report(
    y_true,
    y_probs,
    label="",
    threshold=0.5,
    # Annual EC incidence, all-ages: World-standardized (GLOBOCAN 2024, 14.4/100,000) and
    # Norwegian-standardized (Cancer in Norway 2025 report, 28.4/100,000, 2021-2025).
    # Both are all-age averages, not specific to our cohort's median age.
    prevalence_scenarios=(0.000144, 0.000284),
):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    report = {"label": label, "n": len(y_true), "n_cases": int(y_true.sum())}

    report["AUC"], report["AUC_CI_low"], report["AUC_CI_high"] = bootstrap_auc_ci(y_true, y_probs)
    report["PR_AUC"] = average_precision_score(y_true, y_probs)

    for target_spec in (0.90, 0.95, 0.98):
        sens, thr = sensitivity_at_specificity(y_true, y_probs, target_spec)
        report[f"Sens_at_{int(target_spec * 100)}pct_spec"] = sens
        report[f"Threshold_at_{int(target_spec * 100)}pct_spec"] = thr

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens_default = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec_default = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    report[f"Sensitivity_at_{threshold}"] = sens_default
    report[f"Specificity_at_{threshold}"] = spec_default
    report[f"Precision_at_{threshold}"] = precision_score(y_true, y_pred, zero_division=0)
    report[f"BalancedAccuracy_at_{threshold}"] = balanced_accuracy_score(y_true, y_pred)

    for prev in prevalence_scenarios:
        ppv, npv = ppv_npv_at_prevalence(sens_default, spec_default, prev)
        report[f"PPV_prevalence_{prev}"] = ppv
        report[f"NPV_prevalence_{prev}"] = npv

    return report


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(csv_path, group_col="set"):
    """Load one CSV, map Case/Control, drop whichever of the fixed covariate
    columns are actually present, and split into X, y (and the group/pair
    column if the file has one)."""
    df = pd.read_csv(csv_path)
    df["cacostat"] = df["cacostat"].map({"Case": 1, "Control": 0})
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    pair = df[group_col] if group_col in df.columns else None
    drop_cols = [c for c in (group_col, "cacostat") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["cacostat"]
    return X, y, pair


# ---------------------------------------------------------------------------
# XGBoost + Hyperopt
# ---------------------------------------------------------------------------

def cross_val_auc(X, y, pair, params, n_splits=N_SPLITS, save_path=None):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = None
    if save_path:
        plt.figure(figsize=(8, 6))
        fig = plt.gcf()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=pair)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(n_jobs=N_JOBS, **params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = y_proba

        fpr, tpr, _ = roc_curve(y_val, y_proba)
        fold_auc = roc_auc_score(y_val, y_proba)
        fold_aucs.append(fold_auc)

        if fig:
            plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f"Fold {fold + 1} (AUC = {fold_auc:.3f})")
            plt.fill_between(fpr, tpr - 0.05, tpr + 0.05, alpha=0.1)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    overall_auc = roc_auc_score(y, oof_preds)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    std_auc = np.std(fold_aucs)

    if fig:
        plt.plot(mean_fpr, mean_tpr, color="black", linestyle="--",
                  label=f"OOF AUC = {overall_auc:.3f} ± {std_auc:.3f}", lw=2)
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="gray", alpha=0.3)
        plt.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{n_splits}-Fold ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    return overall_auc, oof_preds, fold_aucs


def hyperopt_objective(params, X, y, pair):
    params = dict(params)
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    auc, _, _ = cross_val_auc(X, y, pair, params, n_splits=N_SPLITS, save_path=None)
    return {"loss": -auc, "status": STATUS_OK}


def train_xgboost(X, y, pair, tag, model_dir, max_evals):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"EVAL-{max_evals}_best_model_simplecv-{tag}.joblib")
    roc_path = f"{tag}_roc_folds_best_trial.pdf"

    trials = Trials()
    best = fmin(
        lambda params: hyperopt_objective(params, X, y, pair),
        SPACE4XGB,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    best_trial_idx = np.argmin([t["result"]["loss"] for t in trials.trials])
    print(f"\nBest Trial Index: {best_trial_idx + 1}")
    print(f"Best Trial Loss (mean OOF AUC): {-trials.trials[best_trial_idx]['result']['loss']:.4f}")
    print("Best Trial Parameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["min_child_weight"] = int(best["min_child_weight"])

    model = xgb.XGBClassifier(n_jobs=N_JOBS, **best)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    overall_auc, _, fold_aucs = cross_val_auc(X, y, pair, best, save_path=roc_path)
    print(f"Fold AUCs of best trial: {fold_aucs}")
    print(f"Mean AUC: {overall_auc:.4f}")

    return model


def run_shap(model, X, tag, model_dir):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(80, 80))
    shap.plots.bar(shap_values, show=False)
    plt.savefig(os.path.join(model_dir, f"shap_bar_plot_{tag}.pdf"), bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Mean SHAP Value": abs(shap_values.values).mean(axis=0),
    }).sort_values(by="Mean SHAP Value", ascending=False)
    shap_importance.to_csv(os.path.join(model_dir, f"shap_feature_importance_{tag}.csv"), index=False)


def evaluate_test_sets(model, test_specs, out_dir, tag):
    """test_specs: list of (label, csv_path) tuples, as many as you gave on the command line."""
    all_metrics = []
    for label, csv_path in test_specs:
        X_test, y_test, _ = load_xy(csv_path)
        y_probs = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_probs)
        print(f"Test AUC ({label}): {auc_score:.4f}")
        all_metrics.append(full_metrics_report(y_test, y_probs, label=label))

    metrics_df = pd.DataFrame(all_metrics)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"clinical_metrics_{tag}.csv")
    metrics_df.to_csv(out_path, index=False)
    print(metrics_df)
    print(f"Clinical metrics written to {out_path}")
    return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_test_arg(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--test must be given as label=path, got: {spec!r}")
    label, path = spec.split("=", 1)
    return label, path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument(
        "--test", action="append", default=[], type=parse_test_arg, metavar="LABEL=PATH",
        help="A held-out test CSV as label=path. Repeat for multiple test sets. Optional.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Filename tag for saved model/plots/metrics (default: derived from --train filename)",
    )
    parser.add_argument("--model-dir", default="./xgboost-models", help="Directory for the saved model + SHAP outputs")
    parser.add_argument("--out-dir", default=".", help="Directory for the clinical metrics CSV")
    parser.add_argument("--max-evals", type=int, default=3000, help="Hyperopt max_evals (default: 3000)")
    args = parser.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.train))[0]

    X, y, pair = load_xy(args.train)
    print(f"[{tag}] train n = {len(y)}, case fraction = {sum(y) / len(y):.3f}")

    model = train_xgboost(X, y, pair, tag, args.model_dir, args.max_evals)
    run_shap(model, X, tag, args.model_dir)

    if args.test:
        evaluate_test_sets(model, args.test, args.out_dir, tag)
    else:
        print("No --test sets given; skipping evaluation and clinical metrics.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Logistic Regression + Random Forest + SVM training pipeline for EC
miRNA VST-normalized data. No file paths are hardcoded. During 5-fold
cross-validation in the training dataset, case–control pairs (the "set" column)
are kept within the same fold to preserve pairing structure. Scored ROC AUC.
Training only -- no test-set evaluation and no clinical metrics here.

Usage:
    python lr_rf_svm_generic.py --train ec_training.csv
    python lr_rf_svm_generic.py --train <file>.csv --models rf svm --tag early
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import parallel_backend, dump

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = 912437
N_SPLITS = 5

# Dropped from the file if present; harmless if a file lacks one.
COLUMNS_TO_DROP = ["sampleid"]

def load_xy(csv_path, group_col="set"):
    """Load one CSV, map Case/Control, drop whichever of the fixed covariate
    columns are actually present, and split into X, y, and the group/pair
    column (if the file has one)."""
    df = pd.read_csv(csv_path)
    df["cacostat"] = df["cacostat"].map({"Case": 1, "Control": 0})
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    pair = df[group_col] if group_col in df.columns else None
    drop_cols = [c for c in (group_col, "cacostat") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["cacostat"]
    return X, y, pair


def make_cv():
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def report_grid_result(grid, model_label, tag):
    print(f"\n---- {model_label} | {tag} ----")
    print(f"Grid best score ({grid.scoring}): {grid.best_score_}")
    print("Grid best parameters:")
    for k, v in grid.best_params_.items():
        print(f"  {k:>20}: {v}")

    best_index = grid.best_index_
    mean_auc = grid.cv_results_["mean_test_score"][best_index]
    fold_auc_scores = [grid.cv_results_[f"split{i}_test_score"][best_index] for i in range(N_SPLITS)]
    print(f"Mean AUC for the best model: {mean_auc}")
    print(f"AUC for each fold of the best model: {fold_auc_scores}")


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def run_lr(X, y, pair, tag, model_dir, n_jobs):
    sgkf = make_cv()

    pipeline_lr = Pipeline([
        ("scalar", MinMaxScaler()),
        ("model", LogisticRegression(solver="saga", max_iter=10000)),
    ])
    grid_values = {
        "model__C": np.logspace(-4, 4, 1000),
        "model__l1_ratio": np.logspace(-1, 0, 20),
        "model__penalty": ["l1", "elasticnet"],
    }

    with parallel_backend("multiprocessing"):
        grid_lr = GridSearchCV(
            pipeline_lr,
            param_grid=grid_values,
            scoring="roc_auc",
            cv=sgkf.split(X, y, pair),
            n_jobs=n_jobs,
        )
        grid_lr.fit(X, y)

    report_grid_result(grid_lr, "LR", tag)

    os.makedirs(model_dir, exist_ok=True)
    dump(grid_lr.best_estimator_, os.path.join(model_dir, f"lr_{tag}_model.joblib"))
    return grid_lr


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

def run_rf(X, y, pair, tag, model_dir, n_jobs):
    sgkf = make_cv()

    grid_values = {
        "criterion": ["entropy", "gini"],
        "n_estimators": [50, 100, 250, 500],
        "max_depth": np.arange(2, X.shape[1], 2),
        "min_samples_split": [4, 8, 16],
        "min_samples_leaf": [2, 4, 8],
    }

    with parallel_backend("multiprocessing"):
        grid_rf = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid=grid_values,
            scoring="roc_auc",
            cv=sgkf.split(X, y, pair),
            n_jobs=n_jobs,
        )
        grid_rf.fit(X, y)

    report_grid_result(grid_rf, "RF", tag)

    os.makedirs(model_dir, exist_ok=True)
    dump(grid_rf.best_estimator_, os.path.join(model_dir, f"rf_{tag}_model.joblib"))

    best_rf = grid_rf.best_estimator_
    feature_importance = best_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(12, 45))
    plt.barh(pos, feature_importance[sorted_idx], xerr=std[sorted_idx][::-1], align="center")
    plt.yticks(pos, np.array(list(X.columns))[sorted_idx])
    plt.title("Feature Importance (MDI)", fontsize=10)
    plt.xlabel("Mean decrease in impurity")
    plt.savefig(os.path.join(model_dir, f"rf_{tag}_mdi.pdf"), bbox_inches="tight")
    plt.close()

    return grid_rf


# ---------------------------------------------------------------------------
# SVM
# three kernel branches, each tuned independently
# ---------------------------------------------------------------------------

def run_svm(X, y, pair, tag, model_dir, n_jobs):
    sgkf = make_cv()

    pipe = Pipeline([
        ("scalar", MinMaxScaler()),
        ("classifier", SVC()),
    ])
    grid_param = [
        {"classifier": [SVC(class_weight=None, probability=True, kernel="linear")],
         "classifier__C": np.logspace(-2, 2, 100)},
        {"classifier": [SVC(class_weight=None, probability=True, kernel="rbf")],
         "classifier__C": np.logspace(-2, 2, 100),
         "classifier__gamma": np.logspace(-2, 1, 100)},
        {"classifier": [SVC(class_weight=None, probability=True, kernel="poly")],
         "classifier__C": np.logspace(-2, 2, 100),
         "classifier__degree": [2, 3, 4, 5]},
    ]

    with parallel_backend("multiprocessing"):
        grid_svm = GridSearchCV(
            pipe,
            grid_param,
            cv=sgkf.split(X, y, pair),
            verbose=0,
            n_jobs=n_jobs,
            scoring="roc_auc",
        )
        grid_svm.fit(X, y)

    print(f"\n---- SVM | {tag} ----")
    print(f"Grid best kernel: {grid_svm.best_params_['classifier'].kernel}")
    report_grid_result(grid_svm, "SVM", tag)

    os.makedirs(model_dir, exist_ok=True)
    dump(grid_svm.best_estimator_, os.path.join(model_dir, f"svm_{tag}_model.joblib"))
    return grid_svm


MODEL_RUNNERS = {
    "lr": run_lr,
    "rf": run_rf,
    "svm": run_svm,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_RUNNERS.keys()), default=list(MODEL_RUNNERS.keys()),
        help="Which model(s) to train (default: all three)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Filename tag for saved outputs (default: derived from --train filename)",
    )
    parser.add_argument("--model-dir", default=".", help="Directory to save trained models (and the RF plot) into")
    parser.add_argument("--n-jobs", type=int, default=100, help="n_jobs for GridSearchCV (default: 100)")
    args = parser.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.train))[0]

    X, y, pair = load_xy(args.train)
    print(f"[{tag}] n = {len(y)}, case fraction = {sum(y) / len(y):.3f}")

    for model_name in args.models:
        MODEL_RUNNERS[model_name](X, y, pair, tag, args.model_dir, args.n_jobs)


if __name__ == "__main__":
    main()

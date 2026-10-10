"""
Prepare the purchase-intention models for the app.

Run once before starting the app:
    python train_model.py

Same steps as the notebook used for the paper: duplicates removed ->
stratified 70:30 split (random_state=42) -> one-hot encoding of the
categorical features -> class-weighted Random Forest and Logistic Regression.

Two feature sets are trained:

  "without"  PageValues removed. PageValues is computed from completed
             transactions, so it leaks the outcome (target leakage). This is
             the honest early-session model and the app's default.
  "with"     All features, as in the original paper results. Kept only so the
             effect of PageValues can be shown.

If best_random_forest_model.joblib / best_logistic_regression_model.joblib
(the models saved by the notebook) are in this folder, they are used as-is for
the "with" feature set, so the app shows exactly the paper's numbers.
Otherwise new models are trained with the same settings; the numbers can
differ slightly if your scikit-learn version is not the one used in Colab.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA_FILE = HERE / "online_shoppers_intention.csv"
NOTEBOOK_MODEL = HERE / "best_random_forest_model.joblib"
NOTEBOOK_LR_MODEL = HERE / "best_logistic_regression_model.joblib"
MODEL_FILE = HERE / "model.joblib"

RANDOM_STATE = 42
LEAKY_FEATURE = "PageValues"

# Same feature lists and order as the notebook.
CATEGORICAL = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
NUMERICAL = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Weekend",
]
TARGET = "Revenue"

# Best GridSearchCV settings per feature set.
# "with": from the notebook (paper Section 3.2).
# "without": REPLACE these with the best_params_ printed by the PageValues
# ablation cell in your Colab notebook, so the app matches the paper.
RF_PARAMS = {
    "with": dict(n_estimators=150, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=2),
    "without": dict(n_estimators=150, criterion="log_loss", max_depth=20,
                    min_samples_split=2, min_samples_leaf=2),
}
LR_C = {"with": 0.01, "without": 0.1}


def numerical_for(variant):
    if variant == "with":
        return NUMERICAL
    return [c for c in NUMERICAL if c != LEAKY_FEATURE]


def build_rf(variant):
    pre = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", numerical_for(variant)),
    ])
    return Pipeline(steps=[
        ("preprocessor", pre),
        ("classifier", RandomForestClassifier(
            **RF_PARAMS[variant], class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def build_lr(variant):
    pre = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), numerical_for(variant)),
    ])
    return Pipeline(steps=[
        ("preprocessor", pre),
        ("classifier", LogisticRegression(
            C=LR_C[variant], class_weight="balanced", max_iter=2000,
            random_state=RANDOM_STATE)),
    ])


def report(name, y_test, proba):
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  {name:<20} Acc {accuracy_score(y_test, pred):.4f}  "
          f"Prec {precision_score(y_test, pred):.4f}  "
          f"Rec {recall_score(y_test, pred):.4f}  "
          f"F1 {f1_score(y_test, pred):.4f}  "
          f"AUC {roc_auc_score(y_test, proba):.4f}  "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Rows loaded:            {len(df):,}")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed:     {before - len(df):,}")
    print(f"Rows used:              {len(df):,}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print(f"Purchase / no purchase: {int(y.sum()):,} / {int((y == 0).sum()):,}")

    pv_positive = (df[LEAKY_FEATURE] > 0).groupby(y).mean()
    print(f"PageValues > 0:         {pv_positive[1]:.1%} of purchases, "
          f"{pv_positive[0]:.1%} of non-purchases")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    variants = {}
    for variant in ("without", "with"):
        print(f"\nFeature set: {variant.upper()} PageValues (threshold 0.50)")
        if variant == "with" and NOTEBOOK_MODEL.exists():
            rf_pipe = joblib.load(NOTEBOOK_MODEL)
            source = "Notebook model (best_random_forest_model.joblib)"
        else:
            rf_pipe = build_rf(variant).fit(X_train, y_train)
            source = "Trained by train_model.py"
        if variant == "with" and NOTEBOOK_LR_MODEL.exists():
            lr_pipe = joblib.load(NOTEBOOK_LR_MODEL)
        else:
            lr_pipe = build_lr(variant).fit(X_train, y_train)

        rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
        lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
        report("Random Forest", y_test, rf_proba)
        report("Logistic Regression", y_test, lr_proba)
        if variant == "with":
            print("  Paper RF:  TN=2902 FP=188 FN=190 TP=382")
            print("  Paper LR:  TN=2702 FP=388 FN=141 TP=431")

        names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        rf = rf_pipe.named_steps["classifier"]
        variants[variant] = {
            "pipeline": rf_pipe,
            "source": source,
            "test_proba": rf_proba,
            "lr_test_proba": lr_proba,
            "importances": pd.Series(
                rf.feature_importances_, index=names
            ).sort_values(ascending=False),
            "rf_params": {
                k: getattr(rf, k)
                for k in ["n_estimators", "criterion", "max_depth",
                          "min_samples_split", "min_samples_leaf",
                          "class_weight", "random_state"]
            },
        }

    numeric_inputs = [c for c in NUMERICAL if c != "Weekend"]
    bundle = {
        "variants": variants,
        "features": list(X.columns),
        "options": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL},
        "medians": X_train[numeric_inputs].median().to_dict(),
        "modes": {c: X_train[c].mode()[0] for c in CATEGORICAL},
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "pv_positive": {"purchase": float(pv_positive[1]),
                        "no_purchase": float(pv_positive[0])},
    }
    joblib.dump(bundle, MODEL_FILE, compress=3)
    print(f"\nSaved {MODEL_FILE.name}. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
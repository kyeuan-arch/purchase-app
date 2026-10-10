"""
Online Shopping Purchase Intention - Random Forest simulation app.

Start it with:
    streamlit run app.py
(run `python train_model.py` first so that model.joblib exists)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

MODEL_FILE = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="Purchase Intention Simulator", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


if not MODEL_FILE.exists():
    st.error("model.joblib was not found. Run `python train_model.py` first.")
    st.stop()

bundle = load_bundle()
if "variants" not in bundle:
    st.error("model.joblib is from an older version. Run "
             "`python train_model.py` again to rebuild it.")
    st.stop()

options = bundle["options"]
y_test = np.asarray(bundle["y_test"])

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
FEATURE_SETS = {
    "Without PageValues (early-session)": "without",
    "With PageValues (original results)": "with",
}
st.sidebar.title("Settings")
feature_set = st.sidebar.radio(
    "Feature set",
    list(FEATURE_SETS),
    help="PageValues is calculated from completed transactions, so it "
    "partly contains the outcome being predicted (target leakage). The "
    "model without it is the one that could be used before a purchase "
    "happens.",
)
variant_key = FEATURE_SETS[feature_set]
use_pv = variant_key == "with"
variant = bundle["variants"][variant_key]
pipeline = variant["pipeline"]
test_proba = np.asarray(variant["test_proba"])
lr_proba = np.asarray(variant["lr_test_proba"])

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="A session is predicted as a purchase when its probability is at "
    "or above this value. Lower it to catch more buyers (higher recall) at "
    "the cost of more false alarms (lower precision).",
)
st.sidebar.caption("The paper's reported results use a threshold of 0.50.")
st.sidebar.divider()
st.sidebar.subheader("Model")
st.sidebar.caption(variant.get("source", ""))
for name, value in variant["rf_params"].items():
    st.sidebar.text(f"{name} = {value}")

st.title("Online Shopping Purchase Intention Simulator")
st.caption(
    "Class-weighted Random Forest classifier trained on the UCI Online "
    "Shoppers Purchasing Intention Dataset."
)
if use_pv:
    st.warning(
        "This feature set includes PageValues, which leaks the outcome. "
        "Scores here are inflated and are shown for comparison only."
    )

tab_single, tab_batch, tab_perf, tab_cmp, tab_pv, tab_imp = st.tabs(
    ["Simulate a session", "Batch simulation", "Model performance",
     "RF vs Logistic Regression", "PageValues effect", "Feature importance"]
)


def pick(col, wanted):
    """Index of `wanted` inside the option list for a categorical column."""
    opts = options[col]
    return opts.index(wanted) if wanted in opts else 0


def metrics_at(proba, t):
    pred = proba >= t
    act = y_test.astype(bool)
    tp = int((pred & act).sum()); tn = int((~pred & ~act).sum())
    fp = int((pred & ~act).sum()); fn = int((~pred & act).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, acc=(tp + tn) / len(act),
                prec=prec, rec=rec, f1=f1, auc=roc_auc_score(y_test, proba))


def confusion_figure(m, title, cmap):
    grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(grid, cmap=cmap)
    labels = ["No Purchase", "Purchase"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    top = grid.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:,}", ha="center", va="center",
                    fontsize=14,
                    color="white" if grid[i, j] > top / 2 else "black")
    fig.tight_layout()
    return fig


baseline_acc = 1 - y_test.mean()

# --------------------------------------------------------------------------
# Tab 1 - one session
# --------------------------------------------------------------------------
med = bundle["medians"]
modes = bundle["modes"]
PRESETS = {
    "Typical session (dataset medians)": {**med, **modes, "Weekend": 0},
    "Window shopper": {
        "Administrative": 0, "Administrative_Duration": 0.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 3, "ProductRelated_Duration": 45.0,
        "BounceRates": 0.10, "ExitRates": 0.15, "PageValues": 0.0,
        "SpecialDay": 0.0, "Month": "May", "VisitorType": "New_Visitor",
        "Weekend": 0,
    },
    "Engaged buyer": {
        "Administrative": 5, "Administrative_Duration": 120.0,
        "Informational": 1, "Informational_Duration": 30.0,
        "ProductRelated": 60, "ProductRelated_Duration": 2400.0,
        "BounceRates": 0.0, "ExitRates": 0.01, "PageValues": 40.0,
        "SpecialDay": 0.0, "Month": "Nov", "VisitorType": "Returning_Visitor",
        "Weekend": 1,
    },
}

with tab_single:
    preset_name = st.selectbox("Start from an example", list(PRESETS))
    p = {**med, **modes, "Weekend": 0, **PRESETS[preset_name]}
    k = preset_name  # widget keys change with the preset so values reset

    st.subheader("Pages visited")
    c1, c2, c3 = st.columns(3)
    with c1:
        administrative = st.number_input(
            "Administrative pages", 0, 50, int(p["Administrative"]), key=f"a{k}")
        administrative_dur = st.number_input(
            "Time on administrative pages (s)", 0.0, 5000.0,
            float(p["Administrative_Duration"]), step=10.0, key=f"ad{k}")
    with c2:
        informational = st.number_input(
            "Informational pages", 0, 30, int(p["Informational"]), key=f"i{k}")
        informational_dur = st.number_input(
            "Time on informational pages (s)", 0.0, 5000.0,
            float(p["Informational_Duration"]), step=10.0, key=f"id{k}")
    with c3:
        product = st.number_input(
            "Product-related pages", 0, 800, int(p["ProductRelated"]), key=f"p{k}")
        product_dur = st.number_input(
            "Time on product-related pages (s)", 0.0, 70000.0,
            float(p["ProductRelated_Duration"]), step=30.0, key=f"pd{k}")

    st.subheader("Session behaviour")
    cols = st.columns(4 if use_pv else 3)
    bounce = cols[0].slider("Bounce rate", 0.0, 0.20,
                            min(float(p["BounceRates"]), 0.20), 0.005,
                            format="%.3f", key=f"b{k}")
    exit_rate = cols[1].slider("Exit rate", 0.0, 0.20,
                               min(float(p["ExitRates"]), 0.20), 0.005,
                               format="%.3f", key=f"e{k}")
    special_day = cols[2].select_slider(
        "Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        value=float(p["SpecialDay"]), key=f"s{k}")
    if use_pv:
        page_values = cols[3].number_input(
            "Page value", 0.0, 400.0, float(p["PageValues"]), step=1.0,
            key=f"pv{k}")
    else:
        page_values = 0.0  # not an input: this model ignores PageValues

    st.subheader("Visitor and context")
    c1, c2, c3, c4 = st.columns(4)
    month = c1.selectbox("Month", options["Month"],
                         index=pick("Month", p["Month"]), key=f"m{k}")
    visitor = c2.selectbox("Visitor type", options["VisitorType"],
                           index=pick("VisitorType", p["VisitorType"]),
                           key=f"v{k}")
    traffic = c3.selectbox("Traffic type", options["TrafficType"],
                           index=pick("TrafficType", p["TrafficType"]),
                           key=f"t{k}")
    weekend = c4.checkbox("Weekend session", bool(p["Weekend"]), key=f"w{k}")
    c1, c2, c3, _ = st.columns(4)
    op_sys = c1.selectbox("Operating system", options["OperatingSystems"],
                          index=pick("OperatingSystems", p["OperatingSystems"]),
                          key=f"o{k}")
    browser = c2.selectbox("Browser", options["Browser"],
                           index=pick("Browser", p["Browser"]), key=f"br{k}")
    region = c3.selectbox("Region", options["Region"],
                          index=pick("Region", p["Region"]), key=f"r{k}")

    # The pipeline selects its own columns by name, so the model trained
    # without PageValues simply ignores that column.
    session = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_dur,
        "Informational": informational,
        "Informational_Duration": informational_dur,
        "ProductRelated": product,
        "ProductRelated_Duration": product_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": op_sys,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor,
        "Weekend": bool(weekend),
    }])[bundle["features"]]

    prob = float(pipeline.predict_proba(session)[0, 1])

    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Purchase probability", f"{prob:.1%}")
    with right:
        if prob >= threshold:
            st.success(f"Prediction: PURCHASE (probability is at or above "
                       f"the {threshold:.2f} threshold)")
        else:
            st.info(f"Prediction: NO PURCHASE (probability is below the "
                    f"{threshold:.2f} threshold)")
        st.progress(prob)
    if use_pv:
        st.caption(
            "Page value dominates this model, and it is calculated from "
            "completed transactions. A real shop would not know it before "
            "the purchase, so this prediction is not a true forecast."
        )
    else:
        st.caption(
            "This model uses only information available while the visitor "
            "is still browsing. It is less accurate than the model with "
            "PageValues, so read the probability as a rough signal."
        )

# --------------------------------------------------------------------------
# Tab 2 - batch simulation on unseen test sessions
# --------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Replays randomly chosen sessions from the unseen test set through "
        "the model, as if they were visitors arriving at the shop."
    )
    c1, c2 = st.columns([3, 1])
    n = c1.slider("Number of sessions", 10, 500, 100, 10)
    if c2.button("Run simulation", type="primary"):
        st.session_state["batch_seed"] = int(np.random.randint(0, 1_000_000))

    if "batch_seed" in st.session_state:
        rng = np.random.default_rng(st.session_state["batch_seed"])
        idx = rng.choice(len(y_test), size=n, replace=False)
        sim_proba = test_proba[idx]
        sim_pred = sim_proba >= threshold
        sim_actual = y_test[idx].astype(bool)

        buyers = int(sim_actual.sum())
        caught = int((sim_pred & sim_actual).sum())
        false_alarms = int((sim_pred & ~sim_actual).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sessions", n)
        m2.metric("Actual buyers", buyers)
        m3.metric("Buyers detected", f"{caught} of {buyers}")
        m4.metric("False alarms", false_alarms)

        shown = ["ProductRelated", "ProductRelated_Duration", "ExitRates",
                 "BounceRates", "Month", "VisitorType"]
        if use_pv:
            shown.insert(3, "PageValues")
        table = bundle["X_test"].iloc[idx][shown].copy()
        table.insert(0, "Probability", sim_proba.round(3))
        table.insert(1, "Predicted", np.where(sim_pred, "Purchase", "No purchase"))
        table.insert(2, "Actual", np.where(sim_actual, "Purchase", "No purchase"))
        table.insert(3, "Correct", sim_pred == sim_actual)
        table = table.round({"ProductRelated_Duration": 1, "ExitRates": 3,
                             "BounceRates": 3, "PageValues": 2})
        st.dataframe(table.reset_index(drop=True))
    else:
        st.info("Press Run simulation to draw a sample of sessions.")

# --------------------------------------------------------------------------
# Tab 3 - performance on the full test set at the chosen threshold
# --------------------------------------------------------------------------
with tab_perf:
    pm = metrics_at(test_proba, threshold)
    actual = y_test.astype(bool)

    st.write(f"Evaluated on {len(actual):,} unseen test sessions at a "
             f"threshold of {threshold:.2f}.")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{pm['acc']:.2%}")
    m[1].metric("Precision", f"{pm['prec']:.2%}")
    m[2].metric("Recall", f"{pm['rec']:.2%}")
    m[3].metric("F1-score", f"{pm['f1']:.2%}")
    m[4].metric("ROC-AUC", f"{pm['auc']:.4f}")
    st.caption(
        f"For comparison, predicting 'no purchase' for every session would "
        f"score {baseline_acc:.2%} accuracy while detecting no buyers, "
        f"which is why accuracy is not read on its own."
    )
    if pm["acc"] < baseline_acc:
        st.warning(
            f"At this threshold the model's accuracy ({pm['acc']:.2%}) is "
            f"below that baseline ({baseline_acc:.2%}). Its value is in the "
            f"buyers it detects, not in overall accuracy."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(confusion_figure(pm, "Random Forest", "Blues"))
        st.caption(
            "Move the threshold in the sidebar to see how false negatives "
            "trade against false positives."
        )
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {pm['auc']:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        ax.scatter([pm["fp"] / (pm["fp"] + pm["tn"])], [pm["rec"]],
                   color="red", zorder=3, label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.subheader("Effect of the threshold")
    rows = []
    for t in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        tm = metrics_at(test_proba, t)
        rows.append({
            "Threshold": f"{t:.2f}",
            "Precision": f"{tm['prec']:.2%}" if tm["tp"] + tm["fp"] else "-",
            "Recall": f"{tm['rec']:.2%}",
            "F1-score": f"{tm['f1']:.2%}",
            "Buyers detected": tm["tp"],
            "Buyers missed": tm["fn"],
            "False alarms": tm["fp"],
        })
    st.table(pd.DataFrame(rows).set_index("Threshold"))
    st.caption(
        "A lower threshold detects more buyers but raises more false alarms."
    )

# --------------------------------------------------------------------------
# Tab 4 - Random Forest vs Logistic Regression
# --------------------------------------------------------------------------
with tab_cmp:
    rf_m = metrics_at(test_proba, threshold)
    lr_m = metrics_at(lr_proba, threshold)
    st.write(f"Both models evaluated on the same {len(y_test):,} unseen "
             f"test sessions at a threshold of {threshold:.2f}, "
             f"{'with' if use_pv else 'without'} PageValues.")
    names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    keys = ["acc", "prec", "rec", "f1", "auc"]
    st.table(pd.DataFrame({
        "Metric": names,
        "Random Forest": [f"{rf_m[k]:.2%}" for k in keys],
        "Logistic Regression": [f"{lr_m[k]:.2%}" for k in keys],
    }).set_index("Metric"))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    b1 = ax.bar(x - 0.18, [rf_m[k] * 100 for k in keys], 0.36,
                label="Random Forest")
    b2 = ax.bar(x + 0.18, [lr_m[k] * 100 for k in keys], 0.36,
                label="Logistic Regression")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{bar.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.legend(loc="lower right")
    ax.set_title("Performance comparison")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Confusion matrices")
    c1, c2 = st.columns(2)
    c1.pyplot(confusion_figure(rf_m, "Random Forest", "Blues"))
    c2.pyplot(confusion_figure(lr_m, "Logistic Regression", "Greens"))
    st.caption(
        f"Random Forest missed {rf_m['fn']} buyers with {rf_m['fp']} "
        f"false alarms; Logistic Regression missed {lr_m['fn']} buyers "
        f"with {lr_m['fp']} false alarms."
    )

    st.subheader("ROC curve comparison")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for proba, label in ((test_proba, "Random Forest"),
                         (lr_proba, "Logistic Regression")):
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr,
                label=f"{label} (AUC = {roc_auc_score(y_test, proba):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 5 - effect of PageValues (target leakage)
# --------------------------------------------------------------------------
with tab_pv:
    pvp = bundle["pv_positive"]
    st.write(
        "PageValues is calculated by Google Analytics from completed "
        "transactions, so it partly contains the outcome the model is "
        "trying to predict. This is called target leakage."
    )
    c1, c2 = st.columns(2)
    c1.metric("Purchase sessions with PageValues > 0",
              f"{pvp['purchase']:.1%}")
    c2.metric("Non-purchase sessions with PageValues > 0",
              f"{pvp['no_purchase']:.1%}")

    st.subheader(f"Same models, with and without PageValues "
                 f"(threshold {threshold:.2f})")
    v_with, v_without = bundle["variants"]["with"], bundle["variants"]["without"]
    runs = {
        "RF with": metrics_at(np.asarray(v_with["test_proba"]), threshold),
        "RF without": metrics_at(np.asarray(v_without["test_proba"]), threshold),
        "LR with": metrics_at(np.asarray(v_with["lr_test_proba"]), threshold),
        "LR without": metrics_at(np.asarray(v_without["lr_test_proba"]), threshold),
    }
    names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    keys = ["acc", "prec", "rec", "f1", "auc"]
    table = {"Metric": names + ["False alarms", "Buyers missed"]}
    for label, mm in runs.items():
        table[label] = ([f"{mm[k]:.2%}" for k in keys]
                        + [f"{mm['fp']:,}", f"{mm['fn']:,}"])
    st.table(pd.DataFrame(table).set_index("Metric"))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, mm) in enumerate(runs.items()):
        ax.bar(x + (i - 1.5) * 0.2, [mm[k] * 100 for k in keys], 0.2,
               label=label)
    ax.axhline(baseline_acc * 100, linestyle="--", color="grey", linewidth=1)
    ax.text(len(names) - 0.5, baseline_acc * 100 + 1,
            "always 'no purchase' accuracy", ha="right", fontsize=8,
            color="grey")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.legend(ncol=4, loc="lower center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Predicting 'no purchase' for every session scores "
        f"{baseline_acc:.2%} accuracy. Without PageValues the models "
        f"depend on browsing behaviour alone, which is only moderately "
        f"predictive."
    )

# --------------------------------------------------------------------------
# Tab 6 - feature importance
# --------------------------------------------------------------------------
with tab_imp:
    top = variant["importances"].head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top 15 features ({'with' if use_pv else 'without'} "
                 f"PageValues)")
    st.pyplot(fig)
    st.caption(
        "Importance shows how useful a feature was to the model's "
        "predictions. It does not show that the feature causes a purchase."
    )
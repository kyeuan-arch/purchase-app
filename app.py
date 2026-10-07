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
pipeline = bundle["pipeline"]
options = bundle["options"]
y_test = np.asarray(bundle["y_test"])
test_proba = np.asarray(bundle["test_proba"])

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Settings")
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
for name, value in bundle["rf_params"].items():
    st.sidebar.text(f"{name} = {value}")

st.title("Online Shopping Purchase Intention Simulator")
st.caption(
    "Random Forest classifier trained on the UCI Online Shoppers Purchasing "
    "Intention Dataset."
)

tab_single, tab_batch, tab_perf, tab_imp = st.tabs(
    ["Simulate a session", "Batch simulation", "Model performance",
     "Feature importance"]
)


def pick(col, wanted):
    """Index of `wanted` inside the option list for a categorical column."""
    opts = options[col]
    return opts.index(wanted) if wanted in opts else 0


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
    c1, c2, c3, c4 = st.columns(4)
    bounce = c1.slider("Bounce rate", 0.0, 0.20,
                       min(float(p["BounceRates"]), 0.20), 0.005,
                       format="%.3f", key=f"b{k}")
    exit_rate = c2.slider("Exit rate", 0.0, 0.20,
                          min(float(p["ExitRates"]), 0.20), 0.005,
                          format="%.3f", key=f"e{k}")
    page_values = c3.number_input("Page value", 0.0, 400.0,
                                  float(p["PageValues"]), step=1.0, key=f"pv{k}")
    special_day = c4.select_slider(
        "Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        value=float(p["SpecialDay"]), key=f"s{k}")

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
        "Weekend": int(weekend),
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
    st.caption(
        "Page value has the largest influence on the model. In a live shop "
        "it is only known once the visitor has moved toward checkout pages, "
        "so treat predictions that rely on it as late-session predictions."
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

        table = bundle["X_test"].iloc[idx][
            ["ProductRelated", "ProductRelated_Duration", "ExitRates",
             "PageValues", "Month", "VisitorType"]
        ].copy()
        table.insert(0, "Probability", sim_proba.round(3))
        table.insert(1, "Predicted", np.where(sim_pred, "Purchase", "No purchase"))
        table.insert(2, "Actual", np.where(sim_actual, "Purchase", "No purchase"))
        table.insert(3, "Correct", sim_pred == sim_actual)
        st.dataframe(table.reset_index(drop=True))
    else:
        st.info("Press Run simulation to draw a sample of sessions.")

# --------------------------------------------------------------------------
# Tab 3 - performance on the full test set at the chosen threshold
# --------------------------------------------------------------------------
with tab_perf:
    pred = test_proba >= threshold
    actual = y_test.astype(bool)
    tp = int((pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    auc = roc_auc_score(y_test, test_proba)

    st.write(f"Evaluated on {len(actual):,} unseen test sessions at a "
             f"threshold of {threshold:.2f}.")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{accuracy:.2%}")
    m[1].metric("Precision", f"{precision:.2%}")
    m[2].metric("Recall", f"{recall:.2%}")
    m[3].metric("F1-score", f"{f1:.2%}")
    m[4].metric("ROC-AUC", f"{auc:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.table(pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual: No purchase", "Actual: Purchase"],
            columns=["Predicted: No purchase", "Predicted: Purchase"],
        ))
        st.caption(
            "Move the threshold in the sidebar to see how false negatives "
            "trade against false positives."
        )
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        ax.scatter([fp / (fp + tn)], [recall], color="red", zorder=3,
                   label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 4 - feature importance
# --------------------------------------------------------------------------
with tab_imp:
    top = bundle["importances"].head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 15 features")
    st.pyplot(fig)
    st.caption(
        "Importance shows how useful a feature was to the model's "
        "predictions. It does not show that the feature causes a purchase."
    )
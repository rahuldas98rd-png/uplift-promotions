"""Interactive policy explorer for the uplift modeling project.

Run from the project root:
    uv run streamlit run streamlit_app.py

Requires that notebooks/03_evaluation.ipynb has been run end-to-end so
the cached CATE predictions and demo artifacts are on disk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from uplift.evaluation import (
    assign_segments,
    policy_from_cate,
    snips_value,
    topk_policy,
)
from uplift.splits import load_splits
from uplift.treatment import make_binary_treatment

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Uplift Promotions Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading — @st.cache_data prevents reload on every slider change
# ---------------------------------------------------------------------------


@st.cache_data
def load_artifacts():
    """Load everything the app needs. Cached for the session."""
    splits = load_splits()
    test_df = splits["test"]
    cate_test = pd.read_parquet(PROCESSED_DIR / "cate_test.parquet")

    mu0 = np.load(PROCESSED_DIR / "mu0_test.npy")
    response = np.load(PROCESSED_DIR / "response_score_test.npy")

    T = make_binary_treatment(test_df).values
    Y_visit = test_df["visit"].values.astype(float)
    Y_spend = test_df["spend"].values.astype(float)

    # Hillstrom is randomized — propensity is constant. Use the observed rate.
    propensity = np.full(len(test_df), float(T.mean()))

    return test_df, cate_test, mu0, response, T, Y_visit, Y_spend, propensity


try:
    test_df, cate_test, mu0_test, response_score, T_test, Y_visit, Y_spend, prop_test = (
        load_artifacts()
    )
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: `{e.filename}`. "
        "Run `notebooks/03_evaluation.ipynb` end to end first — the final "
        "cell saves the files this demo needs."
    )
    st.stop()


METHODS = {
    "X-learner": "xlearner",
    "DR-learner": "drlearner",
    "Causal Forest": "causalforest",
    "T-learner": "tlearner",
    "S-learner": "slearner",
}

N_CUSTOMERS = len(test_df)
BASELINE_THRESHOLD = float(np.median(mu0_test))


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


st.sidebar.title("Policy controls")
st.sidebar.markdown("Adjust to update the policy live.")

method_label = st.sidebar.selectbox("CATE estimator", list(METHODS.keys()), index=0)
method_key = METHODS[method_label]

st.sidebar.markdown("**Economics**")
cost = st.sidebar.slider(
    "Cost per email ($)", min_value=0.05, max_value=0.30, value=0.10, step=0.01
)
margin = st.sidebar.slider("Margin on spend", min_value=0.10, max_value=0.50, value=0.30, step=0.05)

st.sidebar.markdown("**Scaling**")
scale_factor = st.sidebar.slider(
    "Visit→spend scale",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help=(
        "CATE models are trained on visit (dense signal). "
        "Multiply by this to convert to dollar spend. ~5 was calibrated "
        "from the data in Phase 5."
    ),
)


# ---------------------------------------------------------------------------
# Current-policy computation
# ---------------------------------------------------------------------------


cate_visit = cate_test[method_key].values
cate_spend = cate_visit * scale_factor

policy = policy_from_cate(cate_spend, cost=cost, margin=margin)
fraction_targeted = float(policy.mean())
n_targeted = int(policy.sum())

policy_value = snips_value(policy, T_test, Y_spend, prop_test)
net_value = policy_value - cost * fraction_targeted


# Naive baseline at the same coverage
def topk_from_score(score, k_fraction):
    if k_fraction <= 0:
        return np.zeros(len(score), dtype=int)
    if k_fraction >= 1:
        return np.ones(len(score), dtype=int)
    return topk_policy(score, k_fraction)


naive_policy = topk_from_score(response_score, fraction_targeted)
naive_value = snips_value(naive_policy, T_test, Y_spend, prop_test)
naive_net = naive_value - cost * fraction_targeted

# Reference policies
all_value = snips_value(np.ones(N_CUSTOMERS, dtype=int), T_test, Y_spend, prop_test)
none_value = float(Y_spend[T_test == 0].mean())


# ---------------------------------------------------------------------------
# Header and top metrics
# ---------------------------------------------------------------------------


st.title("Uplift Promotions — Policy Explorer")
st.markdown(
    "Decision rule: send a promotional email if the predicted spend lift × margin "
    "exceeds the email cost. Adjust the controls in the sidebar; this page "
    "updates live."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Targeted",
    f"{fraction_targeted:.1%}",
    delta=f"{n_targeted:,} of {N_CUSTOMERS:,}",
    delta_color="off",
)
col2.metric(
    "Decision threshold",
    f"τ̂ > ${cost / margin:.3f}",
    help="Predicted spend lift required to send.",
)
col3.metric(
    "Spend / customer",
    f"${policy_value:.4f}",
    delta=f"{policy_value - naive_value:+.4f} vs naive",
)
col4.metric(
    "Net of cost",
    f"${net_value:.4f}",
    delta=f"{net_value - naive_net:+.4f} vs naive",
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Value-vs-coverage chart
# ---------------------------------------------------------------------------


st.subheader("Policy value vs. targeting fraction")
st.markdown(
    "How spend per customer changes as the policy targets more people. "
    "Compare the chosen uplift method (blue) against the naive "
    "response-targeting baseline (orange)."
)


@st.cache_data
def compute_curve(method_key: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    cate = cate_test[method_key].values * scale
    grid = np.linspace(0.0, 1.0, 41)
    values = []
    for k in grid:
        p = topk_from_score(cate, k)
        values.append(snips_value(p, T_test, Y_spend, prop_test))
    return grid, np.array(values)


@st.cache_data
def compute_naive_curve() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 41)
    values = []
    for k in grid:
        p = topk_from_score(response_score, k)
        values.append(snips_value(p, T_test, Y_spend, prop_test))
    return grid, np.array(values)


grid, uplift_curve = compute_curve(method_key, scale_factor)
_, naive_curve = compute_naive_curve()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(grid, uplift_curve, color="#4c78a8", linewidth=2.3, label=f"{method_label}")
ax.plot(
    grid,
    naive_curve,
    color="orange",
    linewidth=2,
    linestyle="--",
    label="Naive: target by predicted response",
)
ax.axvline(
    fraction_targeted,
    color="#e45756",
    linestyle=":",
    linewidth=1.5,
    label=f"Current ({fraction_targeted:.0%})",
)
ax.scatter([fraction_targeted], [policy_value], color="#e45756", s=80, zorder=5)
ax.set_xlabel("Fraction of customers targeted")
ax.set_ylabel("Spend per customer (SNIPS, $)")
ax.set_xlim(0, 1)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------------------------
# Segment breakdown
# ---------------------------------------------------------------------------


st.markdown("---")
st.subheader("Who is being targeted? Customer segments")

segments = assign_segments(
    cate_visit,
    mu0_test,
    cate_threshold=0.0,
    baseline_threshold=BASELINE_THRESHOLD,
)

seg_order = ["persuadable", "sure_thing", "lost_cause", "do_not_disturb"]
seg_labels = {
    "persuadable": "Persuadables (visit only if emailed)",
    "sure_thing": "Sure-things (visit regardless)",
    "lost_cause": "Lost causes (don't visit either way)",
    "do_not_disturb": "Do-not-disturbs (respond negatively to email)",
}

seg_target = pd.crosstab(pd.Series(segments), pd.Series(policy))
seg_target.columns = (
    ["Not targeted", "Targeted"] if seg_target.shape[1] == 2 else seg_target.columns
)
if "Targeted" not in seg_target.columns:
    seg_target["Targeted"] = 0
if "Not targeted" not in seg_target.columns:
    seg_target["Not targeted"] = 0
seg_target = seg_target[["Not targeted", "Targeted"]]
seg_target["Total"] = seg_target.sum(axis=1)
seg_target["% targeted"] = (seg_target["Targeted"] / seg_target["Total"] * 100).round(1)

seg_target = seg_target.reindex([s for s in seg_order if s in seg_target.index])
seg_target.index = [seg_labels[s] for s in seg_target.index]

col_a, col_b = st.columns([5, 4])

with col_a:
    st.markdown("**Targeting reach by segment**")
    st.dataframe(seg_target, use_container_width=True)
    st.caption(
        "An ideal policy targets persuadables and spares the others. "
        "Watch the table shift as you change the cost and margin sliders."
    )

with col_b:
    # Stacked-bar visual of the same data
    fractions_targeted_by_seg = seg_target["Targeted"] / seg_target["Total"]
    fractions_spared_by_seg = seg_target["Not targeted"] / seg_target["Total"]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(seg_target))
    ax2.barh(y_pos, fractions_targeted_by_seg, color="#4c78a8", label="Targeted")
    ax2.barh(
        y_pos,
        fractions_spared_by_seg,
        left=fractions_targeted_by_seg,
        color="#cccccc",
        label="Not targeted",
    )
    ax2.set_yticks(y_pos)
    # Shorten labels for the chart
    short_labels = ["Persuadables", "Sure-things", "Lost causes", "Do-not-disturbs"]
    short_labels = [
        short_labels[i]
        for i, s in enumerate(seg_order)
        if s in [k for k in pd.crosstab(pd.Series(segments), pd.Series(policy)).index]
    ]
    ax2.set_yticklabels([s.split(" (")[0] for s in seg_target.index])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction of segment")
    ax2.legend(loc="lower right", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


st.markdown("---")
st.subheader("Strategy comparison")

comparison = pd.DataFrame(
    {
        "Strategy": [
            f"Uplift ({method_label})",
            "Naive (response targeting, same coverage)",
            "Treat everyone",
            "Treat nobody",
        ],
        "Targeted": [
            f"{fraction_targeted:.1%}",
            f"{fraction_targeted:.1%}",
            "100.0%",
            "0.0%",
        ],
        "Spend / customer": [
            f"${policy_value:.4f}",
            f"${naive_value:.4f}",
            f"${all_value:.4f}",
            f"${none_value:.4f}",
        ],
        "Net of email cost": [
            f"${net_value:.4f}",
            f"${naive_net:.4f}",
            f"${all_value - cost:.4f}",
            f"${none_value:.4f}",
        ],
    }
)
st.dataframe(comparison, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


st.markdown("---")
st.markdown(
    f"""
**About this demo.** Built on the [Hillstrom 2008 e-commerce email experiment]
(http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)
— 64,000 customers randomized across no-email / men's email / women's email
arms, outcomes measured over 2 weeks.

CATE estimates from five methods (S/T/X/DR-learner + Causal Forest), trained
on visit probability. Policy values estimated by Self-Normalized IPS on the
{N_CUSTOMERS:,}-customer test split.

Full methodology in `reports/writeup.md`. Source on GitHub.
"""
)

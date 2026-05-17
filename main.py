import streamlit as st
import numpy as np
from scipy.linalg import lu, solve_triangular
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Production Planning Dashboard",
    layout="wide"
)

st.markdown("""
<style>

/* =========================
   MAIN BACKGROUND
========================= */

[data-testid="stAppViewContainer"] {
    background-color: #FCF8F8;
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #F9DFDF 0%,
        #FBEFEF 45%,
        #FCF8F8 100%
    );
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: black;
}

/* =========================
   SLIDER
========================= */

/* Slider handle */
.stSlider [role="slider"]{
    background-color: #FCF8F8 !important;
    border: 4px solid #F5AFAF !important;
    box-shadow: 0 0 12px rgba(192,132,252,0.7);
}

/* Slider active line */
.stSlider div[data-baseweb="slider"] > div {
    background: linear-gradient(
        90deg,
        #F9DFDF ,
        #FBEFEF ,
        #FCF8F8
    );
}

/* =========================
   TITLES
========================= */

h1 {
    color: #FF5555;
    font-weight: 800;
    letter-spacing: -1px;
}

h2, h3 {
    color: ##f77c7c;
}

/* =========================
   ALERT BOXES
========================= */

.stAlert {
    border-radius: 18px;
    border: none;
}

/* =========================
   METRICS / TEXT
========================= */

p, label {
    color: ##f77c7c;
}

/* =========================
   REMOVE EXTRA SPACE
========================= */

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.title("LU-Based Production Planning Tool")

# ==================================================
# SYSTEM DEFINITION
# ==================================================

A = np.array([
    [2, 1, 1],
    [4, 3, 2],
    [2, 2, 2]
])

capacity = np.array([100, 120, 110])

P, L, U = lu(A)

def solve_lu(b):
    y = solve_triangular(L, np.dot(P, b), lower=True)
    x = solve_triangular(U, y)
    return x

# ==================================================
# USER INPUT
# ==================================================

st.sidebar.header("Demand Input")

d1 = st.sidebar.slider("Product 1 Demand", 0, 100, 20)
d2 = st.sidebar.slider("Product 2 Demand", 0, 100, 40)
d3 = st.sidebar.slider("Product 3 Demand", 0, 100, 30)

b = np.array([d1, d2, d3])

# ==================================================
# SOLUTION
# ==================================================

x = solve_lu(b)

st.subheader("Production Plan")
st.write("Production (x):", np.round(x, 2))

# ==================================================
# BOTTLENECK ANALYSIS
# ==================================================

usage = A @ x
utilization = usage / capacity

st.subheader("Machine Utilization")
st.write(utilization) 
for i, u in enumerate(utilization):
    if u > 0.9:
        st.warning(f"Machine {i+1} is reaching bottleneck level.")

# ==================================================
# GRAPH 1 — PRODUCTION
# ==================================================

st.subheader("Production Distribution")

fig1, ax1 = plt.subplots(figsize=(8,4))

colors1 = ["#12BCF5", "#25BDF0", "#2AC1F3"]

ax1.bar(
    ["Product 1", "Product 2", "Product 3"],
    x,
    color=colors1,
    width=0.6
)

# Background
ax1.set_facecolor("#FCF8F8")
fig1.patch.set_facecolor("#FCF8F8")

# Remove borders
for spine in ax1.spines.values():
    spine.set_visible(False)

# Soft grid
ax1.grid(
    axis='y',
    linestyle='--',
    alpha=0.15
)

# Tick styling
ax1.tick_params(
    colors="black",
    labelsize=11
)



st.pyplot(fig1)

# ==================================================
# GRAPH 2 — UTILIZATION
# ==================================================

st.subheader("Machine Utilization")

fig2, ax2 = plt.subplots(figsize=(8,4))

colors2 = ["#09BAF6", "#17BBF2", "#2CC4F6"]

ax2.bar(
    ["Machine 1", "Machine 2", "Machine 3"],
    utilization,
    color=colors2,
    width=0.6
)

# Background
ax2.set_facecolor("#FCF8F8")
fig2.patch.set_facecolor("#FCF8F8")

# Remove borders
for spine in ax2.spines.values():
    spine.set_visible(False)

# Soft grid
ax2.grid(
    axis='y',
    linestyle='--',
    alpha=0.15
)

# Tick styling
ax2.tick_params(
    colors="black",
    labelsize=11
)



st.pyplot(fig2)

# ==================================================
# SENSITIVITY ANALYSIS
# ==================================================

st.subheader("Sensitivity Analysis")

noise = np.random.normal(0, 1, len(b))
b_noisy = b + noise

x_noisy = solve_lu(b_noisy)

difference = x_noisy - x

st.write(
    "Production Change Due to Noise:",
    np.round(difference, 3)
)

# ==================================================
# CONDITION NUMBER
# ==================================================

st.subheader("Numerical Stability")

cond = np.linalg.cond(A)

st.write("Condition Number:", round(cond, 2))

if cond > 100:
    st.error("System is sensitive!")
else:
    st.success("System is stable!")
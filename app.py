import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.datasets import load_iris


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Iris Species Classification System",
    page_icon="🌸",
    layout="wide"
)


# ==========================================================
# PROFESSIONAL CSS
# ==========================================================

st.markdown("""
<style>

/* Hide Streamlit Elements */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Layout */
.block-container{
    max-width:95%;
    padding-top:2rem;
    padding-bottom:3rem;
}

/* Hero Section */
.hero-title{
    font-size:64px;
    font-weight:800;
    text-align:center;
    margin-bottom:0.5rem;
    letter-spacing:-1px;
}

.hero-subtitle{
    font-size:24px;
    text-align:center;
    color:#9ca3af;
    margin-bottom:4rem;
}

/* Section Titles */
.section-title{
    font-size:34px;
    font-weight:700;
    margin-top:2rem;
    margin-bottom:1rem;
    padding-bottom:10px;
    border-bottom:2px solid rgba(255,255,255,0.08);
}

/* Card Titles */
.card-title{
    font-size:22px;
    font-weight:600;
    margin-bottom:1rem;
}

/* Info Text */
.info-text{
    font-size:18px;
    line-height:1.8;
}

/* Metric Spacing */
[data-testid="metric-container"]{
    padding:18px;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.08);
}

/* Remove Excess White Space */
div[data-testid="stVerticalBlock"]{
    gap:0.8rem;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:18px !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/iris_classifier.pkl"
    )


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    iris = load_iris()

    iris_df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    iris_df["species"] = [
        iris.target_names[i]
        for i in iris.target
    ]

    return iris, iris_df


# ==========================================================
# SPECIES INFORMATION
# ==========================================================

SPECIES_INFO = {

    "Iris Setosa":
    """
    • Small petals

    • Broad sepals

    • Easily distinguishable from the other species

    • Highest classification confidence
    """,

    "Iris Versicolor":
    """
    • Medium-sized petals

    • Intermediate characteristics

    • Moderate petal dimensions

    • Often confused with Virginica
    """,

    "Iris Virginica":
    """
    • Largest petals

    • Long petal dimensions

    • Most similar to Versicolor

    • More complex decision boundary
    """
}


# ==========================================================
# LOAD RESOURCES
# ==========================================================

model = load_model()

iris, iris_df = load_dataset()

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
    """
    <div class="hero-title">
    🌸 Iris Species Classification System
    </div>

    <div class="hero-subtitle">
    Interactive Classification & Analytics Platform for Iris Species Identification
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# PREDICTION DASHBOARD
# ==========================================================

st.markdown(
    """
    <div class="section-title">
    Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)

# ==========================================================
# INPUT CARD
# ==========================================================

with left_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
            Input Parameters
            </div>
            """,
            unsafe_allow_html=True
        )

        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.1
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=5.0,
            value=3.5
        )

        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.4
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=3.0,
            value=0.2
        )

# ==========================================================
# MODEL PREDICTION
# ==========================================================

input_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(
    input_data
)[0]

probabilities = model.predict_proba(
    input_data
)[0]

confidence = (
    probabilities[prediction] * 100
)

predicted_species = species[
    prediction
]

# ==========================================================
# PREDICTION CARD
# ==========================================================

with right_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
            Prediction Results
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            f"Predicted Species: {predicted_species}"
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        probability_df = pd.DataFrame({

            "Species": [
                "Iris Setosa",
                "Iris Versicolor",
                "Iris Virginica"
            ],

            "Probability": [
                probabilities[0] * 100,
                probabilities[1] * 100,
                probabilities[2] * 100
            ]

        })

        fig = px.bar(
            probability_df,
            x="Species",
            y="Probability",
            text="Probability",
            title="Prediction Probability Distribution"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================================
# INPUT SUMMARY
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        """
        <div class="card-title">
        Input Measurements Summary
        </div>
        """,
        unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Sepal Length",
            f"{sepal_length:.1f} cm"
        )
    with m2:
        st.metric(
            "Sepal Width",
            f"{sepal_width:.1f} cm"
        )
    with m3:
        st.metric(
            "Petal Length",
            f"{petal_length:.1f} cm"
        )
    with m4:
        st.metric(
            "Petal Width",
            f"{petal_width:.1f} cm"
        )
        
# ==========================================================
# SPECIES INTELLIGENCE
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
    Species Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            "Predicted Species",
            predicted_species
        )
    with col2:
        st.markdown(
            SPECIES_INFO[predicted_species]
        )

# ==========================================================
# DATASET ANALYTICS
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
    Dataset Analytics
    </div>
    """,
    unsafe_allow_html=True
)

analytics_left, analytics_right = st.columns(
    2,
    gap="large"
)

# ==========================================================
# SPECIES DISTRIBUTION
# ==========================================================

with analytics_left:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
            📈 Species Distribution
            </div>
            """,
            unsafe_allow_html=True
        )

        species_counts = (
            iris_df["species"]
            .value_counts()
            .reset_index()
        )

        species_counts.columns = [
            "Species",
            "Count"
        ]

        fig_distribution = px.bar(
            species_counts,
            x="Species",
            y="Count",
            text="Count",
            color="Species"
        )

        fig_distribution.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

with analytics_right:

    with st.container(border=True):

        st.markdown(
            """
            <div class="card-title">
            📊 Correlation Heatmap
            </div>
            """,
            unsafe_allow_html=True
        )

        corr_matrix = (
            iris_df
            .drop("species", axis=1)
            .corr()
        )

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto"
        )

        fig_heatmap.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

# ==========================================================
# FEATURE RELATIONSHIP ANALYSIS
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):

    st.markdown(
        """
        <div class="card-title">
        Feature Relationship Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    scatter_fig = px.scatter(
        iris_df,
        x="petal length (cm)",
        y="petal width (cm)",
        color="species",
        size="sepal length (cm)",
        hover_data=[
            "sepal width (cm)"
        ],
        template="plotly_dark",
        title=""
    )

    scatter_fig.update_layout(
        height=550
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True
    )

# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander(
    "📄 Dataset Preview (150 Records)",
    expanded=False
):

    st.dataframe(
        iris_df,
        use_container_width=True,
        hide_index=True
    )
    
st.markdown("<br><br>", unsafe_allow_html=True)

st.caption(
    "Built with Streamlit, Scikit-Learn and Plotly"
)

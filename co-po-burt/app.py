import streamlit as st

import pandas as pd

from src.io_utils import load_thresholds, load_targets

from src.nba_math import compute_po_attainment_nba

from src.burt import compute_burt_adjustments_from_students

from src.nlp_mapping import generate_co_po_mapping



st.set_page_config(page_title="CO–PO Attainment System", layout="wide")



st.title("CO–PO / PSO Attainment Dashboard")



st.sidebar.header("Mode")



mode = st.sidebar.radio(

    "Select Mode",

    ["NLP CO–PO Mapping", "PO/PSO Attainment Calculation"]

)



# --------------------

# Upload section

# --------------------

st.sidebar.header("Upload Files")



if mode == "NLP CO–PO Mapping":

    co_text_file = st.sidebar.file_uploader("CO Statements CSV", type=["csv"])

    po_text_file = st.sidebar.file_uploader("PO / PSO Statements CSV", type=["csv"])



    if not co_text_file or not po_text_file:

        st.info("⬅️ Upload CO and PO statement CSVs to generate mapping")

        st.stop()



elif mode == "PO/PSO Attainment Calculation":

    co_file = st.sidebar.file_uploader("CO Attainment CSV", type=["csv"])

    map_file = st.sidebar.file_uploader("CO → PO / PSO Mapping CSV", type=["csv"])

    threshold_file = st.sidebar.file_uploader("Thresholds CSV", type=["csv"])

    target_file = st.sidebar.file_uploader("Targets CSV", type=["csv"])



    if not all([co_file, map_file, threshold_file, target_file]):

        st.info("⬅️ Upload all required CSV files for attainment calculation")

        st.stop()



# --------------------

# NLP Mapping Mode

# --------------------

if mode == "NLP CO–PO Mapping":

    co_text_df = pd.read_csv(co_text_file, encoding="latin1")

    po_text_df = pd.read_csv(po_text_file, encoding="latin1")



    mapping_df = generate_co_po_mapping(co_text_df, po_text_df)



    st.subheader("Generated CO–PO / PSO Mapping (NLP)")

    st.dataframe(mapping_df, use_container_width=True)



    pivot = mapping_df.pivot(

        index="co",

        columns="outcome",

        values="weight"

    ).fillna(0).astype(int)



    st.subheader("CO × PO Matrix (0–3)")

    st.dataframe(pivot, use_container_width=True)



# --------------------

# PO/PSO Attainment Calculation Mode

# --------------------

elif mode == "PO/PSO Attainment Calculation":

    # --------------------

    # Load data

    # --------------------

    co_df = pd.read_csv(co_file)

    map_df = pd.read_csv(map_file)

    thresholds = load_thresholds(threshold_file)

    targets = load_targets(target_file)



    # Filters

    years = sorted(co_df["year"].unique())

    courses = sorted(co_df["course"].unique())

    att_types = sorted(co_df["attainment_type"].unique())



    col1, col2, col3 = st.columns(3)

    year = col1.selectbox("Year", years)

    course = col2.selectbox("Course", courses)

    att_type = col3.selectbox("Attainment Type", att_types)



    co_df = co_df[

        (co_df["year"] == year) &

        (co_df["course"] == course) &

        (co_df["attainment_type"] == att_type)

    ]

    map_df = map_df[map_df["course"] == course]



    # --------------------

    # Burt (optional)

    # --------------------

    use_burt = st.sidebar.checkbox("Use Burt Adjustment", value=False)

    student_file = None

    if use_burt:

        student_file = st.sidebar.file_uploader("Student CO Scores CSV", type=["csv"])

        if student_file is None:

            st.error("Student CO Scores CSV required for Burt mode")

            st.stop()

        stu_df = pd.read_csv(student_file)

        stu_df = stu_df[

            (stu_df["year"] == year) &

            (stu_df["course"] == course)

        ]

        assoc = compute_burt_adjustments_from_students(stu_df, thresholds)

    else:

        assoc = None



    # --------------------

    # Compute

    # --------------------

    results = compute_po_attainment_nba(

        co_attainment=co_df,

        mapping=map_df,

        thresholds=thresholds,

        targets=targets,

        attainment_type=att_type,

        assoc=assoc,

    )



    # --------------------

    # Display

    # --------------------

    st.subheader("CO Attainment (Used)")

    st.dataframe(results["co_attainment_used"], use_container_width=True)



    st.subheader("CO Attainment Levels")

    st.dataframe(results["co_report"], use_container_width=True)



    st.subheader("PO / PSO Attainment (Scale of 3)")

    st.dataframe(results["po_matrix_scale"], use_container_width=True)



    st.subheader("Target Achievement (≥ 1.4)")

    st.dataframe(results["po_matrix_target"], use_container_width=True)



    st.subheader("PO / PSO Attainment (%)")

    st.dataframe(results["po_matrix_pct"], use_container_width=True)



    st.success("✅ Computation complete")

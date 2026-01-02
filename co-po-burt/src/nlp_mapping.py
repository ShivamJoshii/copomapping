import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity





def detect_id_column(df, keywords):

    """

    Detect column containing IDs like CO1, PO1, PSO1

    """

    for col in df.columns:

        col_l = col.lower()

        if any(k in col_l for k in keywords):

            return col

    raise ValueError(f"Could not detect ID column. Columns found: {list(df.columns)}")





def detect_text_column(df, id_col):

    """

    Detect the text/statement column (anything except ID)

    """

    for col in df.columns:

        if col != id_col:

            return col

    raise ValueError("Could not detect text column")





def similarity_to_weight(sim):

    if sim >= 0.75:

        return 3

    if sim >= 0.55:

        return 2

    if sim >= 0.35:

        return 1

    return 0





def generate_co_po_mapping(co_df: pd.DataFrame, po_df: pd.DataFrame) -> pd.DataFrame:

    # ---- detect columns safely ----

    co_id_col = detect_id_column(co_df, ["co"])

    po_id_col = detect_id_column(po_df, ["po", "pso", "outcome"])



    co_text_col = detect_text_column(co_df, co_id_col)

    po_text_col = detect_text_column(po_df, po_id_col)



    # ---- clean data ----

    co_df = co_df.dropna(subset=[co_text_col])

    po_df = po_df.dropna(subset=[po_text_col])



    co_ids = co_df[co_id_col].astype(str).str.strip().tolist()

    po_ids = po_df[po_id_col].astype(str).str.strip().tolist()



    co_texts = co_df[co_text_col].astype(str).tolist()

    po_texts = po_df[po_text_col].astype(str).tolist()



    # ---- embeddings ----

    model = SentenceTransformer("all-MiniLM-L6-v2")



    co_emb = model.encode(co_texts, normalize_embeddings=True)

    po_emb = model.encode(po_texts, normalize_embeddings=True)



    sim_matrix = cosine_similarity(co_emb, po_emb)



    # ---- build mapping ----

    rows = []

    for i, co in enumerate(co_ids):

        for j, outcome in enumerate(po_ids):

            sim = float(sim_matrix[i, j])

            rows.append({

                "co": co,

                "outcome": outcome,

                "similarity": round(sim, 4),

                "weight": similarity_to_weight(sim)

            })



    return pd.DataFrame(rows)

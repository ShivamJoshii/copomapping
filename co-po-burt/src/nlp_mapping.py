import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity



def detect_id_column(df, keywords):

    """

    Detect column containing IDs like CO1, PO1, PSO1

    """

    # 1) exact matches first (most reliable)

    preferred = []

    for k in keywords:

        preferred += [k, f"{k}_id", f"{k}id", f"{k} no", f"{k}_no"]

    preferred = {p.lower() for p in preferred}



    for col in df.columns:

        if col.lower().strip() in preferred:

            return col



    # 2) fallback: substring match

    for col in df.columns:

        col_l = col.lower()

        if any(k in col_l for k in keywords):

            return col



    raise ValueError(f"Could not detect ID column. Columns found: {list(df.columns)}")





def detect_text_column(df, id_col):

    """

    Detect the text/statement column (anything except ID)

    """

    candidates = [c for c in df.columns if c != id_col]



    if not candidates:

        raise ValueError("Could not detect text column")



    # Drop columns that are obviously not text

    blacklist = {"weight", "similarity", "attainment", "score", "marks", "percentage"}

    candidates2 = []

    for c in candidates:

        if c.lower().strip() not in blacklist:

            candidates2.append(c)

    candidates = candidates2 or candidates



    # Choose column with greatest average string length

    best_col = None

    best_len = -1



    for c in candidates:

        s = df[c].dropna().astype(str).str.strip()

        if len(s) == 0:

            continue

        avg_len = s.str.len().mean()

        if avg_len > best_len:

            best_len = avg_len

            best_col = c



    if best_col is None:

        raise ValueError(f"Could not detect text column. Candidates were: {candidates}")



    return best_col





def similarity_to_weight(sim, t3=0.75, t2=0.50, t1=0.26):

    if sim >= t3:

        return 3

    if sim >= t2:

        return 2

    if sim >= t1:

        return 1

    return 0





# ---------- BERT embedding helpers ----------

_MODEL_NAME = "bert-base-uncased"

_tokenizer = None

_model = None



def _load_bert():

    global _tokenizer, _model

    if _tokenizer is None or _model is None:

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

        _model = AutoModel.from_pretrained(_MODEL_NAME)

        _model.eval()

    return _tokenizer, _model



@torch.no_grad()

def bert_encode_texts(texts, batch_size=16, max_length=128, device=None):

    """

    Returns L2-normalized sentence embeddings using mean pooling over token embeddings.

    """

    tokenizer, model = _load_bert()



    if device is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"



    model.to(device)



    all_embs = []



    for start in range(0, len(texts), batch_size):

        batch = texts[start : start + batch_size]



        enc = tokenizer(

            batch,

            padding=True,

            truncation=True,

            max_length=max_length,

            return_tensors="pt",

        )

        enc = {k: v.to(device) for k, v in enc.items()}



        out = model(**enc)  # last_hidden_state: (B, T, H)

        last_hidden = out.last_hidden_state

        attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)



        # mean pooling with mask

        masked = last_hidden * attention_mask

        summed = masked.sum(dim=1)

        counts = attention_mask.sum(dim=1).clamp(min=1e-9)

        mean_pooled = summed / counts  # (B, H)



        # L2 normalize

        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)



        all_embs.append(mean_pooled.cpu().numpy())



    return np.vstack(all_embs)





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



    # normalize text a bit

    co_texts = (

        co_df[co_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()

    )

    po_texts = (

        po_df[po_text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()

    )



    # ---- BERT embeddings ----

    co_emb = bert_encode_texts(co_texts, batch_size=16, max_length=128)

    po_emb = bert_encode_texts(po_texts, batch_size=16, max_length=128)



    sim_matrix = cosine_similarity(co_emb, po_emb)



    # ---- build mapping ----

    rows = []

    for i, co in enumerate(co_ids):

        for j, outcome in enumerate(po_ids):

            sim = float(sim_matrix[i, j])

            rows.append(

                {

                    "co": co,

                    "outcome": outcome,

                    "similarity": round(sim, 4),

                    "weight": similarity_to_weight(sim, t3=0.75, t2=0.50, t1=0.26),

                }

            )



    return pd.DataFrame(rows)

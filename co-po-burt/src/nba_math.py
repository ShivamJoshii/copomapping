from typing import Optional

import numpy as np

import pandas as pd





def pct_to_level(x: float, thresholds: dict) -> int:

    # thresholds: {3:0.70, 2:0.60, 1:0.50}

    if x >= thresholds[3]:

        return 3

    if x >= thresholds[2]:

        return 2

    if x >= thresholds[1]:

        return 1

    return 0





def compute_po_attainment_nba(

    co_attainment: pd.DataFrame,

    mapping: pd.DataFrame,

    thresholds: dict,

    targets: dict,

    attainment_type: str = "FINAL",

    assoc: Optional[pd.DataFrame] = None,

) -> dict:

    """

    co_attainment: year,course,co,attainment_type,value (0..1)

    mapping: course,co,outcome,weight (0..3)

    assoc (optional): course,co,assoc in [0,1] used to adjust weights

    """

    atype = attainment_type.upper().strip()

    co_use = co_attainment[co_attainment["attainment_type"] == atype].copy()

    if co_use.empty:

        raise ValueError(f"No CO attainment rows found for attainment_type={atype}")



    # Join CO attainment with mapping

    merged = mapping.merge(

        co_use[["year", "course", "co", "value"]],

        on=["course", "co"],

        how="inner",

    )

    if merged.empty:

        raise ValueError("Mapping and CO attainment do not overlap. Check course/co names.")



    # Optional Burt adjustment

    merged["effective_weight"] = merged["weight"].astype(float)

    if assoc is not None and not assoc.empty:

        # assoc is per (course,co)

        merged = merged.merge(assoc, on=["course", "co"], how="left")

        merged["assoc"] = merged["assoc"].fillna(1.0)

        merged["effective_weight"] = merged["effective_weight"] * merged["assoc"]



    # PO/PSO attainment per year, course, outcome

    # Formula: sum(value * w) / sum(w)  (ignore outcomes with sum(w)=0)

    merged["num"] = merged["value"] * merged["effective_weight"]

    agg = merged.groupby(["year", "course", "outcome"], as_index=False).agg(

        numerator=("num", "sum"),

        denom=("effective_weight", "sum")

    )

    agg["attainment_value"] = np.where(agg["denom"] > 0, agg["numerator"] / agg["denom"], 0.0)

    agg["attainment_pct"] = agg["attainment_value"] * 100.0



    scale_max = float(targets.get("scale_max", 3.0))

    target_level = float(targets.get("target_level", 1.4))



    # Scale of 3 (like your sheet): value * 3

    agg["attainment_scale"] = agg["attainment_value"] * scale_max

    agg["target_met"] = np.where(agg["attainment_scale"] >= target_level, "Y", "N")



    # CO-level reporting too

    co_rep = co_use.copy()

    co_rep["level"] = co_rep["value"].apply(lambda x: pct_to_level(float(x), thresholds))



    # Pivot matrix outputs for convenience

    po_matrix = agg.pivot_table(index=["year", "course"], columns="outcome", values="attainment_value", fill_value=0.0)

    po_matrix_pct = agg.pivot_table(index=["year", "course"], columns="outcome", values="attainment_pct", fill_value=0.0)

    po_matrix_scale = agg.pivot_table(index=["year", "course"], columns="outcome", values="attainment_scale", fill_value=0.0)

    po_matrix_target = agg.pivot_table(index=["year", "course"], columns="outcome", values="target_met", aggfunc="first", fill_value="N")



    return {

        "co_attainment_used": co_use,

        "co_report": co_rep,

        "merged_detail": merged,

        "po_long": agg,

        "po_matrix_value": po_matrix.reset_index(),

        "po_matrix_pct": po_matrix_pct.reset_index(),

        "po_matrix_scale": po_matrix_scale.reset_index(),

        "po_matrix_target": po_matrix_target.reset_index(),

    }

import numpy as np

import pandas as pd





def pct_to_level(x: float, thresholds: dict) -> int:

    if x >= thresholds[3]:

        return 3

    if x >= thresholds[2]:

        return 2

    if x >= thresholds[1]:

        return 1

    return 0





def compute_burt_adjustments_from_students(student_co_scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:

    """

    TRUE Burt needs many observations. Here: each student is an observation.

    We compute an association score per (course, co) that measures how strongly that CO

    tends to be in Level 3 vs others in that cohort.



    Then we return assoc per (course, co, outcome) later by copying that CO score

    onto all outcomes mapped from that CO (done in nba_math via merge).

    But since nba_math expects assoc per (course, co, outcome), we output only (course, co)

    then expand at merge time would be harder.



    So we output a CO-only assoc and caller can expand if needed.

    Easiest: output (course, co, outcome, assoc) = 1 for now? No.

    We'll output per (course, co, outcome) by requiring you to merge with mapping externally,

    BUT to keep run.py simple, we'll compute assoc per CO and broadcast inside this function

    only if mapping is available. Since we don't have mapping here, we return CO-only scores.

    run.py merges CO-only association to mapping inside nba_math by joining on (course,co) and

    then sets assoc for all outcomes.



    So this returns columns: course, co, assoc

    """



    df = student_co_scores.copy()

    df["level"] = df["co_pct"].astype(float).apply(lambda x: pct_to_level(x, thresholds))



    # Simple Burt-inspired strength proxy:

    # assoc = P(level==3) scaled into [0.5, 1.0] so it adjusts but doesn't destroy weights

    # (You can change scaling later.)

    grp = df.groupby(["course", "co"], as_index=False).agg(

        n=("level", "size"),

        n3=("level", lambda s: int((s == 3).sum())),

        n2=("level", lambda s: int((s == 2).sum())),

        n1=("level", lambda s: int((s == 1).sum())),

    )

    grp["p3"] = np.where(grp["n"] > 0, grp["n3"] / grp["n"], 0.0)



    # Map p3 to assoc weight in [0.7, 1.0]

    grp["assoc"] = 0.7 + 0.3 * grp["p3"]

    grp = grp[["course", "co", "assoc"]]



    # IMPORTANT:

    # This is the practical "association strength" that behaves like Burt would in your context.

    # If you later provide multi-year student-level CO levels, we can do a real Burt ZᵀZ and MCA.



    return grp

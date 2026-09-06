
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1_R10H1_external_lineage_bridge_audit_fix1.py

Purpose
-------
Audit the ACTUAL bridge from the current 9000-row external invariant feature
table back to an upstream table that preserves explicit model/state identity.

Important
---------
The previous global search found several 9000-row files with explicit keys,
but that alone does NOT prove that those rows correspond 1:1 to the current
external feature rows.

This script audits the known external lineage step-by-step:

R10A2_3 external_feature_lock
    -> R10A2_7 external representation lock
    -> R10A2_8 external RUR transform
    -> R10E1 external complete v3

and separately inspects:
R05D2 target_paot_probabilities
R05D3 model_case_prediction_manifest
R05D4 model_case_outcomes

No row-order matching is allowed.
No nearest-neighbour matching is used for recovery.
No labels are attached.

For each adjacent pair the script:
1) finds shared columns,
2) builds exact SHA256 row fingerprints from sample_id + shared VALUE columns,
3) checks uniqueness and set equality,
4) reports whether an exact 1:1 bridge exists.

It also reports all likely identity/state columns and low-cardinality metadata
that may preserve the missing 9-state row identity.
"""

import argparse
from pathlib import Path
import hashlib
from itertools import combinations
import numpy as np
import pandas as pd


DEFAULTS = {
    "a23": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_3_external_feature_lock_v1/"
        "external_feature_lock.csv"
    ),
    "a27": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_7_external_representation_lock_v1/"
        "neopolyp_external_R10A_feature_lock.csv"
    ),
    "a28": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10A2_8_external_rur_transform_v1/"
        "neopolyp_external_invariant_feature_table.csv"
    ),
    "e1v3": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R10E1_external_complete_feature_v3/"
        "external_complete_invariant_feature_table_v2.csv"
    ),
    "r05d2": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R05D2_neopolyp_prospective_paot_probability_lock_v1_fix1/"
        "target_paot_probabilities.csv"
    ),
    "r05d3": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R05D3_neopolyp_source_a1_prediction_lock_v1_fix1/"
        "model_case_prediction_manifest.csv"
    ),
    "r05d4": (
        "F:/MEDSEG_SAFETTA/outputs/"
        "Q1_R05D4_neopolyp_gt_reveal_paot_confirmatory_evaluation_v1/"
        "model_case_outcomes.csv"
    ),
}

IDENTITY_HINTS = (
    "sample", "case", "model", "state", "seed", "checkpoint",
    "family", "fold", "method", "adapt", "run", "trial",
)

LABEL_HINTS = (
    "outcome", "harm", "benefit", "label", "dice", "target",
)

NONVALUE_EXCLUDE = (
    "path", "sha", "timestamp", "time", "date",
)


def norm_sid(x):
    return str(x).strip().replace("\\", "/").lower()


def canonical(v):
    if pd.isna(v):
        return "nan"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return format(float(v), ".15g")
    s = str(v).strip()
    # normalize numeric-looking strings
    try:
        f = float(s)
        if np.isfinite(f):
            return format(f, ".15g")
    except Exception:
        pass
    return s.lower()


def fingerprint_df(df, cols):
    vals = []
    for _, row in df.iterrows():
        parts = [norm_sid(row["sample_id"])]
        parts.extend(canonical(row[c]) for c in cols)
        vals.append(
            hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        )
    return pd.Series(vals, index=df.index)


def schema_audit(name, df):
    print(f"\n===== SCHEMA: {name} =====")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("All columns:")
    for c in df.columns:
        print("  ", c)

    if "sample_id" in df.columns:
        sid = df["sample_id"].map(norm_sid)
        vc = sid.value_counts()
        print("Unique sample_id:", vc.size)
        print("Multiplicity min/max:", int(vc.min()), int(vc.max()))
        print("Multiplicity distribution:")
        print(vc.value_counts().sort_index().to_string())

    identity = [
        c for c in df.columns
        if any(h in c.lower() for h in IDENTITY_HINTS)
    ]
    labels = [
        c for c in df.columns
        if any(h in c.lower() for h in LABEL_HINTS)
    ]
    print("Identity-like columns:", identity)
    print("Label-like columns:", labels)

    print("\nLow-cardinality metadata candidates (2..100 unique):")
    low = []
    for c in df.columns:
        if c == "sample_id":
            continue
        try:
            n = df[c].nunique(dropna=False)
        except Exception:
            continue
        if 2 <= n <= 100:
            low.append((c, int(n), str(df[c].dtype)))
    if low:
        for item in sorted(low, key=lambda x: (x[1], x[0])):
            print("  ", item)
    else:
        print("  NONE")

    # Test obvious identity key combinations.
    candidates = [
        c for c in identity
        if c != "sample_id" and c not in labels
    ]
    print("\nCandidate unique keys with sample_id:")
    found = 0
    for r in (1, 2, 3):
        for comb in combinations(candidates, r):
            cols = ["sample_id"] + list(comb)
            try:
                u = df[cols].astype(str).drop_duplicates().shape[0]
            except Exception:
                continue
            if u == len(df):
                print("  UNIQUE:", " + ".join(cols))
                found += 1
                if found >= 20:
                    break
        if found >= 20:
            break
    if found == 0:
        print("  NONE")


def choose_shared_value_cols(a, b):
    shared = [c for c in a.columns if c in b.columns and c != "sample_id"]

    # Prefer non-label, non-identity actual feature/value columns.
    preferred = []
    fallback = []

    for c in shared:
        lc = c.lower()
        if any(h in lc for h in LABEL_HINTS):
            continue
        if any(h in lc for h in NONVALUE_EXCLUDE):
            continue

        # Identity metadata are useful for diagnostics, but not for proving
        # that feature rows themselves are identical.
        if any(h in lc for h in IDENTITY_HINTS):
            fallback.append(c)
            continue

        preferred.append(c)

    return shared, preferred, fallback


def bridge_audit(name_a, a, name_b, b):
    print(f"\n\n===== EXACT BRIDGE AUDIT: {name_a} -> {name_b} =====")

    if "sample_id" not in a.columns or "sample_id" not in b.columns:
        print("BRIDGE: IMPOSSIBLE (sample_id missing)")
        return {"status": "NO_SAMPLE_ID"}

    sid_a = set(a["sample_id"].map(norm_sid))
    sid_b = set(b["sample_id"].map(norm_sid))

    print("Rows:", len(a), "vs", len(b))
    print("Unique sample IDs:", len(sid_a), "vs", len(sid_b))
    print("Sample-ID set equal:", sid_a == sid_b)

    shared, preferred, identity_shared = choose_shared_value_cols(a, b)

    print("Shared columns total:", len(shared))
    print("Shared VALUE columns:", len(preferred))
    print("Shared identity-like columns:", identity_shared)
    print("Shared VALUE columns:")
    for c in preferred:
        print("  ", c)

    if not preferred:
        print("EXACT_1_TO_1_VALUE_BRIDGE: NOT_TESTABLE")
        return {
            "status": "NOT_TESTABLE",
            "shared_value_cols": 0,
        }

    fa = fingerprint_df(a, preferred)
    fb = fingerprint_df(b, preferred)

    ua = fa.nunique()
    ub = fb.nunique()
    sa = set(fa)
    sb = set(fb)
    inter = sa & sb

    print("Unique fingerprints:", ua, "vs", ub)
    print("Fingerprint intersection:", len(inter))
    print("A-only fingerprints:", len(sa - sb))
    print("B-only fingerprints:", len(sb - sa))
    print("Duplicated fingerprint rows A:", int(fa.duplicated(keep=False).sum()))
    print("Duplicated fingerprint rows B:", int(fb.duplicated(keep=False).sum()))

    exact = (
        len(a) == len(b)
        and ua == len(a)
        and ub == len(b)
        and sa == sb
    )

    print("EXACT_1_TO_1_VALUE_BRIDGE:", "YES" if exact else "NO")

    return {
        "status": "YES" if exact else "NO",
        "shared_value_cols": len(preferred),
        "shared_value_names": ";".join(preferred),
        "rows_a": len(a),
        "rows_b": len(b),
        "unique_fp_a": ua,
        "unique_fp_b": ub,
        "intersection": len(inter),
    }


def key_set_crosscheck(name_a, a, name_b, b, key):
    print(f"\n===== EXPLICIT KEY SET CHECK: {name_a} vs {name_b} =====")
    print("Key:", key)

    if not all(c in a.columns for c in key):
        print(name_a, "missing:", [c for c in key if c not in a.columns])
        return False
    if not all(c in b.columns for c in key):
        print(name_b, "missing:", [c for c in key if c not in b.columns])
        return False

    aa = a[key].astype(str).drop_duplicates()
    bb = b[key].astype(str).drop_duplicates()

    print(name_a, "unique keys:", len(aa), "/", len(a))
    print(name_b, "unique keys:", len(bb), "/", len(b))

    sa = set(map(tuple, aa.to_numpy()))
    sb = set(map(tuple, bb.to_numpy()))

    print("Key-set intersection:", len(sa & sb))
    print("A-only:", len(sa - sb))
    print("B-only:", len(sb - sa))
    print("KEY_SET_EQUAL:", sa == sb)

    return sa == sb


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    ap.add_argument(
        "--output_dir",
        default=(
            "F:/MEDSEG_SAFETTA/outputs/"
            "Q1_R10H1_external_lineage_bridge_audit_fix1_v1"
        ),
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== R10H1 EXTERNAL LINEAGE BRIDGE AUDIT FIX1 =====")
    print("ROW-ORDER MATCHING: FORBIDDEN")
    print("NEAREST-NEIGHBOUR RECOVERY: FORBIDDEN")
    print("LABEL ATTACHMENT: NONE")
    print("TRAINING: NONE")
    print("EVALUATION: NONE")

    dfs = {}
    for k in DEFAULTS:
        p = Path(getattr(args, k))
        print(f"{k}: exists={p.exists()}  path={p}")
        if not p.exists():
            raise FileNotFoundError(p)
        dfs[k] = pd.read_csv(p, low_memory=False)

    # Full schema audit: this is critical because prior global scan only
    # loaded selected columns and may have hidden useful state metadata.
    for k in ["a23", "a27", "a28", "e1v3", "r05d2", "r05d3", "r05d4"]:
        schema_audit(k, dfs[k])

    # Known external feature lineage.
    bridge_rows = []
    for ka, kb in [
        ("a23", "a27"),
        ("a27", "a28"),
        ("a28", "e1v3"),
        ("a23", "e1v3"),
    ]:
        r = bridge_audit(ka, dfs[ka], kb, dfs[kb])
        r["from"] = ka
        r["to"] = kb
        bridge_rows.append(r)

    # R05D2/R05D3/R05D4 should carry the same frozen explicit state key.
    for key in [
        ["sample_id", "model_state_id"],
        ["sample_id", "model_family", "training_seed"],
        ["sample_id", "model_state_id", "checkpoint_sha256"],
    ]:
        key_set_crosscheck("r05d2", dfs["r05d2"], "r05d3", dfs["r05d3"], key)
        key_set_crosscheck("r05d3", dfs["r05d3"], "r05d4", dfs["r05d4"], key)

    pd.DataFrame(bridge_rows).to_csv(
        out / "external_lineage_bridge_summary.csv",
        index=False,
    )

    print("\n===== FINAL BRIDGE DECISION =====")
    statuses = {(r["from"], r["to"]): r["status"] for r in bridge_rows}

    if (
        statuses.get(("a23", "a27")) == "YES"
        and statuses.get(("a27", "a28")) == "YES"
        and statuses.get(("a28", "e1v3")) == "YES"
    ):
        print("DECISION: EXTERNAL_FEATURE_CHAIN_EXACTLY_RECOVERABLE")
        print(
            "The current external rows can be traced exactly back to R10A2_3 "
            "without row-order assumptions."
        )
        print(
            "Inspect R10A2_3 schema above: if it contains a unique state key, "
            "the next step is an exact key join to R05D3/R05D4."
        )
    elif statuses.get(("a23", "e1v3")) == "YES":
        print("DECISION: EXTERNAL_FEATURE_CHAIN_DIRECT_BRIDGE_RECOVERABLE")
        print(
            "R10A2_3 directly bridges exactly to the current external table."
        )
    else:
        print("DECISION: EXTERNAL_FEATURE_CHAIN_NOT_FULLY_RECOVERABLE")
        print(
            "At least one transformation stage lost or changed the exact "
            "row-level feature identity. Do NOT attach labels yet."
        )
        print(
            "Use the schema/bridge output above to identify the earliest "
            "stage where row identity was lost, then rebuild from that stage "
            "with model_state_id retained."
        )

    print("\nOutput:", out)
    print("PASS: BRIDGE AUDIT COMPLETED")


if __name__ == "__main__":
    main()

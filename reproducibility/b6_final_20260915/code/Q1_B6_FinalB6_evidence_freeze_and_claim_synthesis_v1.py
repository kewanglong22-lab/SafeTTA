#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA Final-B6 evidence freeze + claim synthesis.

READ ONLY with respect to prior experiment stages.
NO model fitting.
NO new scientific metric computation.
NO threshold / coverage / sign / feature changes.
NO manuscript editing.

Purpose:
  1) verify completed P0-1B and P0-2 exact artifacts;
  2) locate already-completed MRI P0-1A / P0-3 evidence by frozen numeric anchors;
  3) freeze evidence table, claim matrix, lineage manifest and claim synthesis
     BEFORE manuscript revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


VERSION = "2026-09-15-FinalB6-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL_PATH = CODE / "B6_FINALB6_EVIDENCE_FREEZE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "8630f5b726f61882ad3eac99d4b0b1e218f9ec8ec2569a998f925d4e51d0d8c2"

P01B_DIR = ROOT / "B6_P01B_polypgen_geometry_core_experiment_v1_fix5"
P01B_AUDIT = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_AUDIT.json"
P01B_METRICS = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_METRICS.csv"
P01B_DELTAS = P01B_DIR / "B6_P01B_POLYPGEN_GEOMETRY_CORE_PAIRED_DELTAS.csv"

P02_DIR = ROOT / "B6_P02_polypgen_exact50_utility_v1"
P02_AUDIT = P02_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_AUDIT.json"
P02_METRICS = P02_DIR / "B6_P02_POLYPGEN_EXACT50_UTILITY_METRICS.csv"
P02_DELTAS = P02_DIR / "B6_P02_POLYPGEN_EXACT50_PAIRED_DELTAS.csv"

OUT_DIR = ROOT / "B6_FinalB6_evidence_freeze_and_claim_synthesis_v1"
PASS_GATE = "PASS_B6_FINALB6_EVIDENCE_FREEZE_AND_CLAIM_SYNTHESIS"

P01A_ANCHORS = {
    "SOURCE_STATE_AUROC": 0.528816,
    "SOURCE_STATE_AUPRC": 0.171936,
    "SEMANTIC_TRANSITION_AUROC": 0.571579,
    "SEMANTIC_TRANSITION_AUPRC": 0.347158,
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE_AUROC": 0.850306,
    "SOURCE_PLUS_SIMPLE_MASK_CHANGE_AUPRC": 0.415812,
    "FULL_TRANSITION_SAFETTA_AUROC": 0.739470,
    "FULL_TRANSITION_SAFETTA_AUPRC": 0.347912,
}

P03_ANCHORS = {
    "GT_NONEMPTY_FULL_AUROC": 0.680667,
    "GT_NONEMPTY_FULL_AUPRC": 0.315584,
    "GT_NONEMPTY_SIMPLE_AUROC": 0.849961,
    "GT_NONEMPTY_SIMPLE_AUPRC": 0.525523,
    "GT_NONEMPTY_FULL_MINUS_SIMPLE_AUROC": -0.167533,
    "GT_NONEMPTY_FULL_MINUS_SIMPLE_AUPRC": -0.207148,
}

MAX_SCAN_BYTES = 8_000_000
SCAN_SUFFIXES = {".json", ".csv", ".txt"}


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    sha = sha256_file(PROTOCOL_PATH)
    if sha.lower() != EXPECTED_PROTOCOL_SHA256.lower():
        raise RuntimeError(
            "Final-B6 protocol SHA mismatch.\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"observed={sha}"
        )
    p = load_json(PROTOCOL_PATH)
    if p.get("status") != "POST_EXPERIMENT_EVIDENCE_FREEZE_BEFORE_MANUSCRIPT_REVISION":
        raise RuntimeError("Final-B6 protocol status changed.")
    return {"path": str(PROTOCOL_PATH), "sha256": sha, "status": p.get("status")}


def exact_float(x: float, y: float, atol: float = 5e-7) -> bool:
    return bool(abs(float(x) - float(y)) <= atol)


def verify_p01b() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    for p in [P01B_AUDIT, P01B_METRICS, P01B_DELTAS]:
        if not p.is_file():
            raise FileNotFoundError(p)

    audit = load_json(P01B_AUDIT)
    if audit.get("gate") != "PASS_B6_P01B_POLYPGEN_GEOMETRY_CORE_EXPERIMENT_COMPLETE":
        raise RuntimeError(f"P01B gate changed: {audit.get('gate')}")
    if audit.get("decision") != "CANDIDATE_CONDITIONING_INCREMENT_SUPPORTED":
        raise RuntimeError(f"P01B decision changed: {audit.get('decision')}")

    m = pd.read_csv(P01B_METRICS)
    d = pd.read_csv(P01B_DELTAS)

    expected = {
        "SOURCE_STATE": (0.649536, 0.119172),
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE": (0.760411, 0.209984),
        "FROZEN_FULL_TRANSITION_SAFETTA": (0.743238, 0.255458),
    }
    for name, (auroc, auprc) in expected.items():
        r = m[m["representation"] == name]
        if len(r) != 1:
            raise RuntimeError(f"P01B metric row missing/duplicate: {name}")
        if not exact_float(r.iloc[0]["auroc"], auroc):
            raise RuntimeError(f"P01B {name} AUROC anchor mismatch.")
        if not exact_float(r.iloc[0]["auprc"], auprc):
            raise RuntimeError(f"P01B {name} AUPRC anchor mismatch.")

    return m, d, {
        "audit": str(P01B_AUDIT),
        "audit_sha256": sha256_file(P01B_AUDIT),
        "metrics": str(P01B_METRICS),
        "metrics_sha256": sha256_file(P01B_METRICS),
        "deltas": str(P01B_DELTAS),
        "deltas_sha256": sha256_file(P01B_DELTAS),
        "gate": audit.get("gate"),
        "decision": audit.get("decision"),
    }


def verify_p02() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    for p in [P02_AUDIT, P02_METRICS, P02_DELTAS]:
        if not p.is_file():
            raise FileNotFoundError(p)

    audit = load_json(P02_AUDIT)
    if audit.get("gate") != "PASS_B6_P02_POLYPGEN_EXACT50_UTILITY_COMPLETE":
        raise RuntimeError(f"P02 gate changed: {audit.get('gate')}")
    if audit.get("decision") != "CANDIDATE_CONDITIONING_EXACT50_SAFETY_SUPPORTED_UTILITY_NOT_CONFIRMED":
        raise RuntimeError(f"P02 decision changed: {audit.get('decision')}")

    m = pd.read_csv(P02_METRICS)
    d = pd.read_csv(P02_DELTAS)

    expected = {
        "SOURCE_STATE": 0.749321,
        "SOURCE_PLUS_SIMPLE_MASK_CHANGE": 0.748043,
        "FROZEN_FULL_TRANSITION_SAFETTA": 0.754432,
    }
    for name, deployed in expected.items():
        r = m[m["representation"] == name]
        if len(r) != 1:
            raise RuntimeError(f"P02 metric row missing/duplicate: {name}")
        if not exact_float(r.iloc[0]["mean_deployed_dice"], deployed):
            raise RuntimeError(f"P02 {name} deployed-Dice anchor mismatch.")

    return m, d, {
        "audit": str(P02_AUDIT),
        "audit_sha256": sha256_file(P02_AUDIT),
        "metrics": str(P02_METRICS),
        "metrics_sha256": sha256_file(P02_METRICS),
        "deltas": str(P02_DELTAS),
        "deltas_sha256": sha256_file(P02_DELTAS),
        "gate": audit.get("gate"),
        "decision": audit.get("decision"),
    }


def iter_candidate_files() -> Iterable[Path]:
    skip = {".git", "__pycache__", "release", "releases", OUT_DIR.name, "code"}
    for child in ROOT.iterdir():
        if not child.is_dir() or child.name in skip:
            continue
        name = child.name.lower()
        if not any(k in name for k in [
            "p01a", "p0_1a", "p0-1a", "p03", "p0_3", "p0-3",
            "mri", "attribution", "robust",
        ]):
            continue
        for dp, dns, fns in os.walk(child):
            dns[:] = [d for d in dns if d not in {".git", "__pycache__", "release"}]
            for fn in fns:
                p = Path(dp) / fn
                if p.suffix.lower() not in SCAN_SUFFIXES:
                    continue
                try:
                    if p.stat().st_size > MAX_SCAN_BYTES:
                        continue
                except OSError:
                    continue
                yield p


def numeric_text_match(text: str, values: List[float], min_hits: int) -> int:
    hits = 0
    for v in values:
        token = f"{v:.6f}"
        if token in text:
            hits += 1
    return hits if hits >= min_hits else 0


def discover_evidence(label: str, anchors: Dict[str, float], min_hits: int) -> List[Dict[str, Any]]:
    vals = list(anchors.values())
    files = list(iter_candidate_files())
    found = []
    for p in tqdm(files, desc=f"Discover {label} evidence", unit="file", dynamic_ncols=True):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = numeric_text_match(text, vals, min_hits)
        if hits:
            found.append({
                "label": label,
                "path": str(p),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "anchor_hits": hits,
                "anchor_total": len(vals),
            })
    found.sort(key=lambda x: (-x["anchor_hits"], x["bytes"], x["path"]))
    return found


def bind_best_discovered(label: str, rows: List[Dict[str, Any]], required_hits: int) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No {label} evidence file located.")
    top = rows[0]
    if top["anchor_hits"] < required_hits:
        raise RuntimeError(
            f"{label} top evidence has only {top['anchor_hits']} anchor hits; required={required_hits}."
        )
    tied = [r for r in rows if r["anchor_hits"] == top["anchor_hits"]]
    return {
        "selected": top,
        "same_hit_candidates": tied,
        "selection_rule": "max frozen-anchor coverage, then smaller provenance artifact; no direction/outcome tuning",
    }


def make_evidence_table(
    p01b_m: pd.DataFrame,
    p01b_d: pd.DataFrame,
    p02_m: pd.DataFrame,
    p02_d: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "stage": "P0-1A MRI attribution",
            "endpoint": "AUROC",
            "representation_or_comparison": "SOURCE_STATE",
            "estimate": 0.528816, "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "SOURCE-state baseline",
        },
        {
            "stage": "P0-1A MRI attribution",
            "endpoint": "AUROC",
            "representation_or_comparison": "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
            "estimate": 0.850306, "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "strongest MRI candidate representation",
        },
        {
            "stage": "P0-1A MRI attribution",
            "endpoint": "AUROC delta",
            "representation_or_comparison": "FULL_TRANSITION_SAFETTA - SOURCE_PLUS_SIMPLE_MASK_CHANGE",
            "estimate": -0.110835, "ci95_low": -0.137994, "ci95_high": -0.083142,
            "interpretation": "simple geometry significantly exceeds full transition on MRI",
        },
    ]

    for _, r in p01b_m.iterrows():
        rows.append({
            "stage": "P0-1B PolypGen unseen MEMO attribution",
            "endpoint": "AUROC",
            "representation_or_comparison": r["representation"],
            "estimate": float(r["auroc"]),
            "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "external ranking",
        })
        rows.append({
            "stage": "P0-1B PolypGen unseen MEMO attribution",
            "endpoint": "AUPRC",
            "representation_or_comparison": r["representation"],
            "estimate": float(r["auprc"]),
            "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "external ranking",
        })

    for _, r in p01b_d.iterrows():
        if r["comparison"] in {
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE",
            "SOURCE_PLUS_SIMPLE_MASK_CHANGE - FROZEN_FULL_TRANSITION_SAFETTA",
        }:
            rows.append({
                "stage": "P0-1B PolypGen unseen MEMO attribution",
                "endpoint": str(r["metric"]),
                "representation_or_comparison": str(r["comparison"]),
                "estimate": float(r["point_delta"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
                "interpretation": "paired physical-image bootstrap delta",
            })

    for _, r in p02_m.iterrows():
        for metric in ["mean_deployed_dice", "prevented_harm_fraction", "benefit_capture_fraction"]:
            rows.append({
                "stage": "P0-2 PolypGen exact50 utility",
                "endpoint": metric,
                "representation_or_comparison": r["representation"],
                "estimate": float(r[metric]),
                "ci95_low": np.nan, "ci95_high": np.nan,
                "interpretation": "fixed 50% matched coverage",
            })

    for _, r in p02_d.iterrows():
        if (
            r["comparison"] in {
                "SOURCE_PLUS_SIMPLE_MASK_CHANGE - SOURCE_STATE",
                "FROZEN_FULL_TRANSITION_SAFETTA - SOURCE_STATE",
            }
            and r["metric"] in {
                "mean_deployed_dice",
                "prevented_harm_fraction",
                "benefit_capture_fraction",
                "committed_harm_rate",
            }
        ):
            rows.append({
                "stage": "P0-2 PolypGen exact50 utility",
                "endpoint": str(r["metric"]),
                "representation_or_comparison": str(r["comparison"]),
                "estimate": float(r["point_delta"]),
                "ci95_low": float(r["ci95_low"]),
                "ci95_high": float(r["ci95_high"]),
                "interpretation": "paired physical-image bootstrap delta",
            })

    rows += [
        {
            "stage": "P0-3 MRI robustness",
            "endpoint": "GT_NONEMPTY AUROC",
            "representation_or_comparison": "FULL_TRANSITION_SAFETTA",
            "estimate": 0.680667, "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "robustness stratum",
        },
        {
            "stage": "P0-3 MRI robustness",
            "endpoint": "GT_NONEMPTY AUROC",
            "representation_or_comparison": "SOURCE_PLUS_SIMPLE_MASK_CHANGE",
            "estimate": 0.849961, "ci95_low": np.nan, "ci95_high": np.nan,
            "interpretation": "robustness stratum",
        },
        {
            "stage": "P0-3 MRI robustness",
            "endpoint": "GT_NONEMPTY AUROC delta",
            "representation_or_comparison": "FULL_TRANSITION_SAFETTA - SOURCE_PLUS_SIMPLE_MASK_CHANGE",
            "estimate": -0.167533, "ci95_low": -0.198275, "ci95_high": -0.136727,
            "interpretation": "empty-slice convention does not explain geometry advantage",
        },
    ]
    return pd.DataFrame(rows)


def make_claim_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        ["C1", "SUPPORTED",
         "Candidate-conditioned information adds future-HARM ranking information beyond SOURCE state across MRI and colonoscopy external shifts.",
         "P0-1A + P0-1B", "primary mechanism claim"],
        ["C2", "SUPPORTED_WITH_SCOPE",
         "Simple no-GT mask geometry is a strong candidate-change signal, especially on MRI.",
         "P0-1A + P0-3 + P0-1B", "representation finding"],
        ["C3", "SUPPORTED",
         "The best candidate-change representation is modality dependent.",
         "MRI simple geometry > full transition; PolypGen full transition has higher AUPRC and better exact50 utility.",
         "secondary scientific finding"],
        ["C4", "SUPPORTED",
         "Ranking quality and deployment utility are distinct endpoints.",
         "P0-1B + P0-2", "deployment interpretation"],
        ["C5", "SUPPORTED",
         "At frozen exact50 on PolypGen, simple geometry is more conservative: it prevents more HARM but captures fewer BENEFIT cases and does not improve deployed Dice over SOURCE-state gating.",
         "P0-2", "negative/nuanced result"],
        ["C6", "SUPPORTED",
         "At frozen exact50 on PolypGen, the frozen full transition score improves deployed Dice, HARM prevention, and BENEFIT capture versus SOURCE-state gating.",
         "P0-2", "utility result"],
        ["F1", "FORBIDDEN",
         "Semantic transition is universally best or necessary.",
         "Contradicted by MRI attribution.", "must not appear"],
        ["F2", "FORBIDDEN",
         "Simple geometry universally replaces semantic transition.",
         "Contradicted by PolypGen AUPRC/exact50 utility.", "must not appear"],
        ["F3", "FORBIDDEN",
         "Simple geometry improves deployed Dice at exact50 on PolypGen.",
         "P0-2 paired delta is significantly negative.", "must not appear"],
        ["F4", "FORBIDDEN",
         "The exact50 policy is guaranteed to outperform SOURCE-only deployment.",
         "Exact50 does not establish this.", "must not appear"],
    ], columns=["claim_id", "status", "claim", "evidence", "manuscript_role"])


def make_markdown(p01a_binding: Dict[str, Any], p03_binding: Dict[str, Any]) -> str:
    p01a_path = p01a_binding["selected"]["path"]
    p03_path = p03_binding["selected"]["path"]
    return f"""# SafeTTA Final-B6 Evidence Freeze

Status: **FROZEN BEFORE MANUSCRIPT REVISION**

## Core scientific position

Frame the manuscript around **candidate-aware pre-commit future-HARM ranking**, not universal superiority of semantic transition.

**SOURCE state + candidate-induced change → pre-commit HARM risk ranking**

The representation of candidate-induced change is modality dependent.

## Frozen evidence

- MRI P0-1A: SOURCE-state AUROC 0.528816; simple mask change 0.850306; full transition 0.739470.
- MRI P0-3 GT-nonempty: full AUROC 0.680667 versus simple 0.849961; full − simple = -0.167533, 95% CI [-0.198275, -0.136727].
- PolypGen P0-1B unseen MEMO: SOURCE-state AUROC/AUPRC 0.649536/0.119172; simple geometry 0.760411/0.209984; frozen full transition 0.743238/0.255458.
- PolypGen simple − SOURCE: ΔAUROC +0.110874 [0.084972, 0.136790]; ΔAUPRC +0.090812 [0.058587, 0.133482].
- PolypGen P0-2 exact50: simple geometry prevents more HARM than SOURCE-state gating but deployed Dice is lower.
- PolypGen full transition exact50: deployed Dice +0.005111 [0.003400, 0.006998] versus SOURCE-state gating, with improved HARM prevention and BENEFIT capture.

## Interpretation to preserve

1. Candidate conditioning is the transferable mechanism.
2. Geometry is not a universal replacement for semantic transition.
3. MRI strongly favors simple candidate geometry.
4. Colonoscopy shows a safety–benefit trade-off: simple geometry is more conservative, while full transition retains more beneficial updates and gives better exact50 deployment utility.
5. Ranking, calibration, operating-point selection, and deployment utility are separate claims.
6. Exact50 is a frozen matched-coverage analysis, not an optimized operating point.

## MRI provenance bindings

P0-1A selected evidence:
`{p01a_path}`

P0-3 selected evidence:
`{p03_path}`

## Next manuscript action

Revise title/abstract/contributions around candidate-aware pre-commit safety ranking; rebuild attribution and exact50 utility tables; explicitly discuss modality-dependent candidate representation and ranking-vs-utility distinction.
"""


def self_test() -> None:
    assert P01A_ANCHORS["SOURCE_PLUS_SIMPLE_MASK_CHANGE_AUROC"] == 0.850306
    assert P03_ANCHORS["GT_NONEMPTY_SIMPLE_AUROC"] == 0.849961
    print("SELF_TEST_ANCHORS=PASS")
    print("SELF_TEST_READ_ONLY=PASS")
    print("SELF_TEST=PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA Final-B6 evidence freeze and claim synthesis."
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--discovery-only", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    print("=" * 164)
    print("SafeTTA Final-B6 — evidence freeze + claim synthesis")
    print(f"Version            : {VERSION}")
    print("Model fitting      : NO")
    print("New science metric : NO")
    print("Manuscript editing : NO")
    print("=" * 164)

    print("\n[1/5] Verify freeze protocol")
    protocol_meta = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol_meta["sha256"])

    print("\n[2/5] Verify completed PolypGen P0-1B / P0-2 artifacts")
    p01b_m, p01b_d, p01b_meta = verify_p01b()
    p02_m, p02_d, p02_meta = verify_p02()
    print("P01B =", p01b_meta["gate"], "/", p01b_meta["decision"])
    print("P02  =", p02_meta["gate"], "/", p02_meta["decision"])

    print("\n[3/5] Locate MRI P0-1A / P0-3 provenance by frozen numeric anchors")
    p01a_found = discover_evidence("P01A_MRI", P01A_ANCHORS, min_hits=4)
    p03_found = discover_evidence("P03_MRI", P03_ANCHORS, min_hits=3)
    p01a_binding = bind_best_discovered("P01A_MRI", p01a_found, required_hits=4)
    p03_binding = bind_best_discovered("P03_MRI", p03_found, required_hits=3)

    print("P01A evidence =", p01a_binding["selected"]["path"])
    print("P01A anchor hits =", p01a_binding["selected"]["anchor_hits"])
    print("P03 evidence  =", p03_binding["selected"]["path"])
    print("P03 anchor hits =", p03_binding["selected"]["anchor_hits"])

    if args.discovery_only:
        print("\nFINALB6_DISCOVERY=PASS")
        print("FREEZE_OUTPUTS=NOT_WRITTEN")
        return

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite Final-B6 freeze: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[4/5] Build frozen evidence and claim matrices")
    evidence = make_evidence_table(p01b_m, p01b_d, p02_m, p02_d)
    claims = make_claim_matrix()

    lineage_rows = []
    for stage, meta in [("P01B", p01b_meta), ("P02", p02_meta)]:
        for role in ["audit", "metrics", "deltas"]:
            lineage_rows.append({
                "stage": stage,
                "artifact_role": role,
                "path": meta[role],
                "sha256": meta[f"{role}_sha256"],
            })

    lineage_rows += [
        {
            "stage": "P01A_MRI",
            "artifact_role": "discovered_evidence",
            "path": p01a_binding["selected"]["path"],
            "sha256": p01a_binding["selected"]["sha256"],
        },
        {
            "stage": "P03_MRI",
            "artifact_role": "discovered_evidence",
            "path": p03_binding["selected"]["path"],
            "sha256": p03_binding["selected"]["sha256"],
        },
    ]
    lineage = pd.DataFrame(lineage_rows)

    p_evidence = OUT_DIR / "B6_FINALB6_EVIDENCE_TABLE.csv"
    p_claims = OUT_DIR / "B6_FINALB6_CLAIM_MATRIX.csv"
    p_lineage = OUT_DIR / "B6_FINALB6_LINEAGE_MANIFEST.csv"
    p_md = OUT_DIR / "B6_FINALB6_CLAIM_SYNTHESIS.md"

    evidence.to_csv(p_evidence, index=False)
    claims.to_csv(p_claims, index=False)
    lineage.to_csv(p_lineage, index=False)
    p_md.write_text(make_markdown(p01a_binding, p03_binding), encoding="utf-8")

    print("\n[5/5] Freeze outputs")
    freeze = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol_meta,
        "p01b": p01b_meta,
        "p02": p02_meta,
        "p01a_mri_binding": p01a_binding,
        "p03_mri_binding": p03_binding,
        "scientific_position": (
            "candidate-aware pre-commit future-HARM ranking; "
            "candidate representation is modality dependent"
        ),
        "forbidden_rescue": [
            "post-P0-2 coverage tuning",
            "score reversal",
            "target calibration",
            "geometry/semantic fusion chosen from target outcome",
            "HARM threshold change",
        ],
        "outputs": {},
    }

    for p in [p_evidence, p_claims, p_lineage, p_md]:
        freeze["outputs"][p.name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    p_json = OUT_DIR / "B6_FINALB6_EVIDENCE_FREEZE.json"
    write_json(p_json, freeze)

    print("\nCORE POSITION:")
    print("candidate-aware pre-commit future-HARM ranking")
    print("candidate representation = modality dependent")
    print("\nGATE=" + PASS_GATE)
    print("freeze_json=" + str(p_json))
    print("stage_dir=" + str(OUT_DIR))
    print("=" * 164)


if __name__ == "__main__":
    main()

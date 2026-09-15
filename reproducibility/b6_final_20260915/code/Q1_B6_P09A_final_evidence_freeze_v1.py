#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SafeTTA B6-P09A — Final evidence freeze

Purpose:
- Bind completed P01-P08 evidence stages.
- Verify key PASS gates where available.
- Hash paper-facing reports/audits/summaries.
- Emit a claim matrix and an evidence manifest.
- NO new scientific metrics, fitting, inference, calibration, or tuning.

This is a reproducibility/release-preparation stage only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any


VERSION = "2026-09-15-B6-P09A-v1"

ROOT = Path(r"F:\MEDSEG_SAFETTA")
CODE = ROOT / "code"

PROTOCOL = CODE / "B6_P09_FINAL_EVIDENCE_FREEZE_PROTOCOL_v1.json"
EXPECTED_PROTOCOL_SHA256 = "3989ee752ed7dc461572a1544c8d55f1f597f85f6eba4cffaa6f1eaf82d7ce76"

OUT_DIR = ROOT / "B6_P09A_final_evidence_freeze_v1"
PASS_GATE = "PASS_B6_P09A_FINAL_EVIDENCE_FREEZE"

STAGES = [
    {
        "id": "FinalB3_MRI",
        "patterns": ["FinalB3_mri_action_specific_outcome_and_future_harm_v1"],
        "required": True,
        "expected_gate": "PASS_FINALB3_MRI_ACTION_OUTCOME_AND_FUTURE_HARM_EVALUATION_COMPLETE",
    },
    {
        "id": "B6_P01A_MRI_ATTRIBUTION",
        "patterns": ["B6_P01A_mri_matched_representation_attribution_v1"],
        "required": True,
        "expected_gate": None,
    },
    {
        "id": "B6_P01B_POLYPGEN_ATTRIBUTION",
        "patterns": ["B6_P01B_polypgen_geometry_core_experiment_v1_fix5"],
        "required": True,
        "expected_gate": "PASS_B6_P01B_POLYPGEN_GEOMETRY_CORE_EXPERIMENT_COMPLETE",
    },
    {
        "id": "B6_P02_POLYPGEN_UTILITY",
        "patterns": ["B6_P02_polypgen_exact50_utility_v1"],
        "required": True,
        "expected_gate": "PASS_B6_P02_POLYPGEN_EXACT50_UTILITY_COMPLETE",
    },
    {
        "id": "B6_P02B_POLYPGEN_ABSOLUTE",
        "patterns": ["B6_P02B_polypgen_exact50_absolute_references_v1"],
        "required": True,
        "expected_gate": "PASS_B6_P02B_POLYPGEN_EXACT50_ABSOLUTE_REFERENCES",
    },
    {
        "id": "B6_P03_MRI_ROBUSTNESS",
        "patterns": ["B6_P03_mri_empty_foreground_sourcequality_robustness_v1"],
        "required": True,
        "expected_gate": None,
    },
    {
        "id": "B6_P05B_RUNTIME",
        "patterns": ["B6_P05B*", "*P05B*runtime*"],
        "required": True,
        "expected_gate": "PASS_B6_P05B_RUNTIME_SCOPE_REPORTING_AUDIT",
    },
    {
        "id": "B6_P06B_MRI_PATIENT_UTILITY",
        "patterns": ["B6_P06B_mri_transition_patient_utility_execution_v1_fix1"],
        "required": True,
        "expected_gate": "PASS_B6_P06B_MRI_TRANSITION_PATIENT_UTILITY_COMPLETE",
    },
    {
        "id": "B6_P07B_MARGIN_SENSITIVITY",
        "patterns": ["B6_P07B_fixed_score_harm_margin_sensitivity_v1"],
        "required": True,
        "expected_gate": "PASS_B6_P07B_FIXED_SCORE_HARM_MARGIN_SENSITIVITY_COMPLETE",
    },
    {
        "id": "B6_P08B_STATISTICAL_CLOSURE",
        "patterns": ["B6_P08B_final_statistical_closure_v1"],
        "required": True,
        "expected_gate": "PASS_B6_P08B_FINAL_STATISTICAL_CLOSURE_COMPLETE",
    },
]

# Only paper-facing evidence files are hashed here. Large replay caches and
# bootstrap-replicate tables are intentionally excluded from the compact
# release manifest; their paths remain available inside audit JSONs.
INCLUDE_NAME_TOKENS = (
    "REPORT", "AUDIT", "SUMMARY", "METRIC", "POINT",
    "EVENT_SUPPORT", "CLAIM", "FREEZE", "GATE",
)
EXCLUDE_NAME_TOKENS = (
    "REPLICATES", "BOUND_PATIENT_UTILITY_ROWS", "MASKS", "PACKBITS",
    "FEATURE", "PREDICTION", "OUTCOMES", "MODELCASE",
)

SUPPORTED_CLAIMS = [
    (
        "Candidate-conditioned information adds future-HARM ranking signal beyond SOURCE-state-only information.",
        "B6_P01A_MRI_ATTRIBUTION;B6_P01B_POLYPGEN_ATTRIBUTION;B6_P08B_STATISTICAL_CLOSURE",
    ),
    (
        "The ranking signal transfers across external domain and held-out/unseen action shifts.",
        "FinalB3_MRI;B6_P01B_POLYPGEN_ATTRIBUTION;B6_P07B_MARGIN_SENSITIVITY",
    ),
    (
        "Representation effectiveness is modality/endpoint dependent; simple geometry is especially strong on MRI.",
        "B6_P01A_MRI_ATTRIBUTION;B6_P03_MRI_ROBUSTNESS;B6_P06B_MRI_PATIENT_UTILITY",
    ),
    (
        "Fixed-budget ranking can improve safety allocation without necessarily improving final Dice over SOURCE-only deployment.",
        "B6_P02_POLYPGEN_UTILITY;B6_P02B_POLYPGEN_ABSOLUTE;B6_P06B_MRI_PATIENT_UTILITY",
    ),
    (
        "NeoPolyp shared transition has significant macro-action AUPRC gain over SOURCE_Q66_ONLY, while macro-action AUROC increment is not statistically significant.",
        "B6_P08B_STATISTICAL_CLOSURE",
    ),
    (
        "Full pre-commit runtime is dominated by candidate TTA execution.",
        "B6_P05B_RUNTIME",
    ),
]

FORBIDDEN_CLAIMS = [
    "Semantic transition is universally superior.",
    "SafeTTA universally improves segmentation Dice.",
    "SafeTTA exact50 outperforms SOURCE-only deployment.",
    "PolypGen exact50 full transition improves Dice over matched random.",
    "MRI is zero-shot cross-task transfer.",
    "The complete full-transition pre-commit pipeline is lightweight.",
    "The target-domain sigmoid output is calibrated probability.",
    "Historical PolypGen/PROMISE12 analyses are pristine prospective validation.",
]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_protocol() -> Dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    got = sha256_file(PROTOCOL)
    if got != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"P09 protocol SHA mismatch expected={EXPECTED_PROTOCOL_SHA256} observed={got}"
        )
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = "FROZEN_AFTER_P08B_BEFORE_PUBLIC_RELEASE_SYNC_AND_MANUSCRIPT_REWRITE"
    if payload.get("status") != expected:
        raise RuntimeError(f"P09 protocol status changed: {payload.get('status')}")
    return {
        "path": str(PROTOCOL),
        "sha256": got,
        "status": payload["status"],
    }


def resolve_stage(stage: Dict[str, Any]) -> Path | None:
    hits: List[Path] = []
    seen = set()
    for pattern in stage["patterns"]:
        for p in sorted(ROOT.glob(pattern)):
            if not p.is_dir():
                continue
            k = str(p.resolve()).lower()
            if k not in seen:
                seen.add(k)
                hits.append(p)
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]

    # Prefer exact first pattern if it exists; otherwise fail rather than guess.
    exact = ROOT / stage["patterns"][0]
    if exact.is_dir():
        return exact
    raise RuntimeError(
        f"Ambiguous stage {stage['id']}: {[str(x) for x in hits]}"
    )


def compact_files(stage_dir: Path) -> List[Path]:
    files = []
    for p in sorted(stage_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.upper()
        if any(tok in name for tok in EXCLUDE_NAME_TOKENS):
            continue
        if p.suffix.lower() not in {".json", ".txt", ".csv"}:
            continue
        if any(tok in name for tok in INCLUDE_NAME_TOKENS):
            files.append(p)

    # Always include top-level JSON/TXT if compact selector found nothing.
    if not files:
        for p in sorted(stage_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".json", ".txt"}:
                files.append(p)

    return files


def text_contains_gate(stage_dir: Path, gate: str) -> bool:
    for p in stage_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".json"}:
            continue
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if gate in s:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SafeTTA P09A final evidence freeze."
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        assert len(STAGES) >= 9
        assert any(x["id"] == "B6_P08B_STATISTICAL_CLOSURE" for x in STAGES)
        assert any("universally superior" in x for x in FORBIDDEN_CLAIMS)
        print("SELF_TEST_STAGE_REGISTRY=PASS")
        print("SELF_TEST_CLAIM_MATRIX=PASS")
        print("SELF_TEST=PASS")
        return

    print("=" * 172)
    print("SafeTTA B6-P09A — Final evidence freeze")
    print("Version              :", VERSION)
    print("New scientific metric: NO")
    print("Model/inference      : NO")
    print("Refit/calibration    : NO")
    print("Threshold/coverage tuning: NO")
    print("=" * 172)

    protocol = verify_protocol()
    print("PROTOCOL_SHA256 =", protocol["sha256"])

    stage_rows = []
    evidence_rows = []
    failures = []

    print("\n[1/3] Resolve final evidence stages")
    for stage in STAGES:
        p = resolve_stage(stage)
        if p is None:
            if stage["required"]:
                failures.append(f"MISSING_STAGE:{stage['id']}")
            stage_rows.append({
                "stage_id": stage["id"],
                "resolved": False,
                "path": None,
                "expected_gate": stage["expected_gate"],
                "gate_found": False,
            })
            print(stage["id"], "=> MISSING")
            continue

        gate_found = True
        if stage["expected_gate"]:
            gate_found = text_contains_gate(p, stage["expected_gate"])
            if not gate_found:
                failures.append(f"GATE_NOT_FOUND:{stage['id']}")

        stage_rows.append({
            "stage_id": stage["id"],
            "resolved": True,
            "path": str(p),
            "expected_gate": stage["expected_gate"],
            "gate_found": bool(gate_found),
        })
        print(
            stage["id"], "=>", p,
            "| gate =", "PASS" if gate_found else "NOT_FOUND"
        )

        for f in compact_files(p):
            evidence_rows.append({
                "stage_id": stage["id"],
                "path": str(f),
                "bytes": int(f.stat().st_size),
                "sha256": sha256_file(f),
            })

    if failures:
        print("\nFAILURES")
        for x in failures:
            print(" -", x)
        raise RuntimeError(
            "P09A fail-closed: required final evidence is incomplete. "
            "Do not freeze/release yet."
        )

    if OUT_DIR.exists():
        raise FileExistsError(
            f"Never overwrite final freeze: {OUT_DIR}\n"
            "Use _fix1 only for implementation repair."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n[2/3] Write evidence + claim manifests")
    p_stages = OUT_DIR / "B6_P09A_STAGE_REGISTRY.csv"
    p_evidence = OUT_DIR / "B6_P09A_EVIDENCE_SHA256_MANIFEST.csv"
    p_claims = OUT_DIR / "B6_P09A_CLAIM_MATRIX.csv"

    with p_stages.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        w.writeheader()
        w.writerows(stage_rows)

    with p_evidence.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["stage_id", "path", "bytes", "sha256"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(evidence_rows)

    claim_rows = []
    for claim, evidence in SUPPORTED_CLAIMS:
        claim_rows.append({
            "status": "SUPPORTED",
            "claim": claim,
            "evidence_stage_ids": evidence,
        })
    for claim in FORBIDDEN_CLAIMS:
        claim_rows.append({
            "status": "FORBIDDEN",
            "claim": claim,
            "evidence_stage_ids": "",
        })

    with p_claims.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["status", "claim", "evidence_stage_ids"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(claim_rows)

    freeze = {
        "status": "PASS",
        "gate": PASS_GATE,
        "version": VERSION,
        "protocol": protocol,
        "stages": stage_rows,
        "evidence_file_count": len(evidence_rows),
        "supported_claims": [
            {"claim": c, "evidence_stage_ids": e}
            for c, e in SUPPORTED_CLAIMS
        ],
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "next": [
            "Public GitHub synchronization",
            "New final paper tag/commit",
            "Manuscript rewrite against this frozen claim matrix",
        ],
    }
    p_freeze = OUT_DIR / "B6_P09A_FINAL_EVIDENCE_FREEZE.json"
    p_freeze.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("stage_registry =", p_stages)
    print("evidence_manifest =", p_evidence)
    print("claim_matrix =", p_claims)
    print("freeze_json =", p_freeze)

    print("\n[3/3] Freeze summary")
    print("resolved_stages =", sum(int(x["resolved"]) for x in stage_rows))
    print("paper_facing_hashed_files =", len(evidence_rows))
    print("supported_claims =", len(SUPPORTED_CLAIMS))
    print("forbidden_claims =", len(FORBIDDEN_CLAIMS))
    print("NEW_PERFORMANCE_EXPERIMENTS_ALLOWED = NO")
    print("NEXT = GitHub final sync/tag, then manuscript rewrite")
    print("GATE=" + PASS_GATE)
    print("=" * 172)


if __name__ == "__main__":
    main()
